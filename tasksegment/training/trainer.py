from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Mapping, Optional, Union
import torch
import numpy as np   
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..configs import (
    DEFAULT_DENSE_GUIDANCE_LOSS_CFG,
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_POSTPROCESS_CFG,
    DEFAULT_FG_RETRIEVAL_ALPHA,
    DEFAULT_TASK_IMPORTANCE_CFG,
    DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    DEFAULT_TRAIN_LOSS_WEIGHT_CFG,
)
from ..data.datasets import MultiOrganDataset
from ..text.bank import make_Ftext_batch
from .augmentations import augment_episode_medical
from .evaluation import evaluate
from .losses import (
    boundary_weighted_ce_loss,
    compute_dense_guidance_supervision,
    compute_task_importance,
    dice_loss,
    get_boundary_weight,
)
from .retrieval import EMATaskMemoryBank, build_support_bank, make_balanced_domain_schedule, select_task_tokens_for_query

def _stack_output_dicts(output_list: List[Dict[str, torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    keys = [
        "pred_masks",
        "dense_guidance_map",
        "refined_task_tokens",
    ]
    out: Dict[str, Optional[torch.Tensor]] = {}
    for key in keys:
        vals = [x[key] for x in output_list if x.get(key, None) is not None]
        out[key] = torch.cat(vals, dim=0) if len(vals) > 0 else None
    return out


def _resolve_float_cfg(
    user_cfg: Optional[Mapping[str, float]],
    default_cfg: Mapping[str, float],
) -> Dict[str, float]:
    cfg = dict(default_cfg)
    if user_cfg is not None:
        cfg.update({key: float(value) for key, value in user_cfg.items()})
    return cfg


def train(
    model: torch.nn.Module,
    train_dataset: MultiOrganDataset,
    val_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    num_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    patience: int = 8,
    max_support_per_domain: Optional[int] = 128,
    val_ensemble_k: int = 8,
    save_path: str = "best_model.pth",
    episodes_per_epoch: Optional[int] = None,
    grad_accum_steps: int = 4,
    ema_bank_momentum: float = 0.99,
    ema_alpha_cfg: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    boundary_loss_weight: float = 0.5,
    text_retrieval_weight: float = DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    fg_retrieval_alpha: float = DEFAULT_FG_RETRIEVAL_ALPHA,
    model_config: Optional[Dict[str, Any]] = None,
    warmup_epochs: int = 3,
    dense_guidance_loss_cfg: Optional[Dict[str, float]] = None,
    train_loss_weight_cfg: Optional[Mapping[str, float]] = None,
    task_importance_cfg: Optional[Mapping[str, float]] = None,
    retrieval_train_enabled: bool =True,  # 🌟 恢复这个参数以兼容外层脚本传参
    **kwargs,  # 🌟 吸收任何其他外层脚本传来的冗余参数防止报错
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    ce_loss = nn.CrossEntropyLoss()
    domains = list(train_dataset.organ_to_positive_indices.keys())
    best_val_dice, patience_counter = -1.0, 0
    ema_memory_bank = EMATaskMemoryBank(momentum=ema_bank_momentum)

    modality_map = {
        "thyroid": "thyroid",
        "TN3K": "thyroid",
        "BUSI_WHU": "BUSI",
        "BUS-BRA": "BUSI",
    }

    train_loss_weight_cfg = _resolve_float_cfg(train_loss_weight_cfg, DEFAULT_TRAIN_LOSS_WEIGHT_CFG)
    task_importance_cfg = _resolve_float_cfg(task_importance_cfg, DEFAULT_TASK_IMPORTANCE_CFG)
    raw_aux_weight = train_loss_weight_cfg["raw_aux"]
    token_consistency_weight = train_loss_weight_cfg["token_consistency"]
    raw_dense_aux_weight = train_loss_weight_cfg["raw_dense_aux"]
    dense_main_weight = train_loss_weight_cfg["dense_main"]

    if ema_alpha_cfg is None:
        ema_alpha_cfg = dict(DEFAULT_EMA_ALPHA_CFG)
    if model_config is None:
        model_config = dict(DEFAULT_MODEL_CONFIG)
    if dense_guidance_loss_cfg is None:
        dense_guidance_loss_cfg = dict(DEFAULT_DENSE_GUIDANCE_LOSS_CFG)

    print(f"📋 训练域列表: {domains}")
    print(
        f"📋 Retrieval-aware train: {retrieval_train_enabled} | warmup_epochs={warmup_epochs} | "
        f"raw_aux_weight={raw_aux_weight} | raw_dense_aux_weight={raw_dense_aux_weight} | "
        f"token_consistency_weight={token_consistency_weight} | dense_main_weight={dense_main_weight}"
    )
    print(f"📋 Dense guidance supervision cfg: {dense_guidance_loss_cfg}")

    def compute_segmentation_loss(pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        boundary_map = get_boundary_weight(target_mask.unsqueeze(1).float(), 3, 5.0)
        return (
            ce_loss(pred_logits, target_mask)
            + dice_loss(pred_logits, target_mask)
            + boundary_loss_weight * boundary_weighted_ce_loss(pred_logits, target_mask, boundary_map)
        )

    for epoch in range(num_epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            for p in optimizer.param_groups:
                p["lr"] = lr * float(epoch + 1) / float(warmup_epochs)

        epoch_support_bank = None
        use_retrieval_training = bool(retrieval_train_enabled and epoch >= warmup_epochs)
        if use_retrieval_training:
            model.eval()
            with torch.no_grad():
                epoch_support_bank = build_support_bank(
                    model=model,
                    support_dataset=train_dataset,
                    device=device,
                    text_bank=text_bank,
                    max_support_per_domain=max_support_per_domain,
                )
            model.train()
            print(f"[Epoch {epoch}] 已构建训练期 support bank（retrieval-aware 开启）")
        else:
            model.train()
            print(f"[Epoch {epoch}] 使用 raw episode training（warmup 阶段）")

        total_iters = int(episodes_per_epoch) if episodes_per_epoch else max(1, len(train_dataset) // (2 * batch_size))
        loop = tqdm(range(total_iters), leave=True)
        optimizer.zero_grad(set_to_none=True)

        epoch_loss, epoch_train_dice_hard = 0.0, 0.0
        domain_loss = {domain: 0.0 for domain in domains}
        domain_dice = {domain: 0.0 for domain in domains}
        domain_count = {domain: 0 for domain in domains}
        domain_schedule = make_balanced_domain_schedule(domains, total_iters)

        for step in loop:
            domain = domain_schedule[step]
            support_indices, query_indices = train_dataset.sample_episode_indices(domain, num_support=batch_size, num_query=batch_size)

            xs = torch.stack([train_dataset[i][0] for i in support_indices]).to(device)
            ys = torch.stack([train_dataset[i][1] for i in support_indices]).unsqueeze(1).float().to(device)
            xq = torch.stack([train_dataset[i][0] for i in query_indices]).to(device)
            yq = torch.stack([train_dataset[i][1] for i in query_indices]).to(device)

            modality = modality_map.get(domain, "generic")
            xs, ys = augment_episode_medical(xs, ys, modality=modality)
            xq, yq_aug = augment_episode_medical(xq, yq.unsqueeze(1).float(), modality=modality)
            yq = yq_aug.squeeze(1).long()

            organ_text = text_bank.get(domain)
            Ftext = make_Ftext_batch(organ_text, batch_size) if organ_text is not None else None

            raw_out = model(xs=xs, ys=ys, Ftext=Ftext, xq=xq)
            raw_pred = raw_out["pred_masks"]
            raw_task_tokens = raw_out["task_tokens"]

            ema_memory_bank.update(domain, raw_task_tokens, importance=compute_task_importance(raw_pred, yq, ys, cfg=task_importance_cfg))

            raw_seg_loss = compute_segmentation_loss(raw_pred, yq)
            raw_dense_loss = compute_dense_guidance_supervision(raw_out, yq.unsqueeze(1).float(), dense_guidance_loss_cfg)

            if use_retrieval_training and epoch_support_bank is not None:
                retrieval_task_tokens_list = []
                retrieval_ema_alphas = []

                for b in range(xq.shape[0]):
                    xq_b = xq[b:b + 1]
                    with torch.no_grad():
                        task_tokens_b, _query_pyramid_b, debug_b = select_task_tokens_for_query(
                            model=model,
                            query_img=xq_b,
                            organ_name=domain,
                            support_bank=epoch_support_bank,
                            device=device,
                            ensemble_k=val_ensemble_k,
                            fg_retrieval_alpha=fg_retrieval_alpha,
                            text_retrieval_weight=text_retrieval_weight,
                            text_bank=text_bank,
                            ema_memory_bank=ema_memory_bank,
                            ema_alpha_cfg=ema_alpha_cfg,
                            return_debug=True,
                        )
                    retrieval_task_tokens_list.append(task_tokens_b.detach())
                    retrieval_ema_alphas.append(float(debug_b.get("ema_alpha", 0.0) or 0.0))

                retrieval_task_tokens = torch.cat(retrieval_task_tokens_list, dim=0)
                retrieval_out = model.segment_with_task(
                    xq=xq,
                    task_tokens=retrieval_task_tokens,
                    output_size=xq.shape[-2:],
                )

                pred = retrieval_out["pred_masks"]

                seg_loss = compute_segmentation_loss(pred, yq)
                dense_main_loss = compute_dense_guidance_supervision(retrieval_out, yq.unsqueeze(1).float(), dense_guidance_loss_cfg)
                raw_aux_loss = raw_aux_weight * raw_seg_loss
                raw_aux_dense_loss = raw_dense_aux_weight * raw_dense_loss
                token_consistency_loss = token_consistency_weight * F.mse_loss(raw_task_tokens, retrieval_task_tokens.detach())
                mean_ema_alpha = float(np.mean(retrieval_ema_alphas)) if len(retrieval_ema_alphas) > 0 else 0.0
            else:
                pred = raw_pred
                retrieval_task_tokens = raw_task_tokens
                seg_loss = raw_seg_loss
                dense_main_loss = raw_dense_loss
                raw_aux_loss = pred.new_zeros(())
                raw_aux_dense_loss = pred.new_zeros(())
                token_consistency_loss = pred.new_zeros(())
                mean_ema_alpha = 0.0

            loss = (
                seg_loss
                + dense_main_weight * dense_main_loss
                + raw_aux_loss
                + raw_aux_dense_loss
                + token_consistency_loss
            )

            (loss / grad_accum_steps).backward()
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == total_iters):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                prob = torch.softmax(pred, dim=1)[:, 1]
                pred_binary = (prob > 0.5).float()
                target_fg = (yq == 1).float()
                intersection = (pred_binary * target_fg).sum(dim=(1, 2))
                union = pred_binary.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
                batch_dice_hard = ((2.0 * intersection + 1.0) / (union + 1.0)).mean().item()

            epoch_loss += loss.item()
            epoch_train_dice_hard += batch_dice_hard
            domain_loss[domain] += loss.item()
            domain_dice[domain] += batch_dice_hard
            domain_count[domain] += 1

            loop.set_description(
                f"Epoch {epoch} Loss: {loss.item():.4f} | Dice: {batch_dice_hard:.4f} | "
                f"G={float(dense_main_loss.item()) if torch.is_tensor(dense_main_loss) else float(dense_main_loss):.3f} | "
                f"EMA α={mean_ema_alpha:.3f} | {domain}"
            )

        iter_count = max(1, total_iters)
        print(f"\nEpoch {epoch} Summary:")
        print(f" [Train] Avg Loss: {epoch_loss / iter_count:.4f} | Avg HardDice: {epoch_train_dice_hard / iter_count:.4f}")
        for dom in domains:
            if domain_count[dom] > 0:
                count = domain_count[dom]
                print(f"  [{dom}] Loss: {domain_loss[dom] / count:.4f} | HardDice: {domain_dice[dom] / count:.4f} (Sampled: {count})")

        val_metrics = evaluate(
            model,
            val_dataset,
            train_dataset,
            device,
            text_bank,
            max_support_per_domain=max_support_per_domain,
            ensemble_k=val_ensemble_k,
            fg_retrieval_alpha=fg_retrieval_alpha,
            text_retrieval_weight=text_retrieval_weight,
            ema_memory_bank=ema_memory_bank,
            ema_alpha_cfg=ema_alpha_cfg,
            domain_thresholds=dict(DEFAULT_DOMAIN_THRESHOLDS),
            postprocess_cfg=dict(DEFAULT_POSTPROCESS_CFG),
            use_postprocess=True,
        )

        if epoch + 1 > warmup_epochs:
            scheduler.step(val_metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(f" [Val | Postprocessed Domain-Balanced] Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | HD95: {val_metrics['hd95']:.2f} | LR: {current_lr:.2e}")
        for organ_name, organ_metrics in val_metrics["per_domain"].items():
            print(f"  [{organ_name}] Dice: {organ_metrics['dice']:.4f} | IoU: {organ_metrics['iou']:.4f} | HD95: {organ_metrics['hd95']:.2f}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                    "ema_memory_bank": ema_memory_bank.state_dict(),
                    "model_config": dict(model_config),
                    "train_config": {
                        "image_size": tuple(getattr(train_dataset, "image_size", (512, 512))),
                        "ema_alpha_cfg": dict(ema_alpha_cfg),
                        "fg_retrieval_alpha": float(fg_retrieval_alpha),
                        "text_retrieval_weight": float(text_retrieval_weight),
                        "retrieval_scoring": "global+foreground+text",
                        "warmup_epochs": int(warmup_epochs),
                        "retrieval_train_enabled": bool(retrieval_train_enabled),
                        "raw_aux_weight": float(raw_aux_weight),
                        "raw_dense_aux_weight": float(raw_dense_aux_weight),
                        "dense_main_weight": float(dense_main_weight),
                        "token_consistency_weight": float(token_consistency_weight),
                        "train_loss_weight_cfg": dict(train_loss_weight_cfg),
                        "task_importance_cfg": dict(task_importance_cfg),
                        "dense_guidance_loss_cfg": dict(dense_guidance_loss_cfg),
                        "domain_thresholds": dict(DEFAULT_DOMAIN_THRESHOLDS),
                        "postprocess_cfg": dict(DEFAULT_POSTPROCESS_CFG),
                    },
                },
                save_path,
            )
            print(f" 🎉 Saved (Best: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f" 📉 Val Dice did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered after epoch {epoch}.")
                break

        print("-" * 80)
    print()