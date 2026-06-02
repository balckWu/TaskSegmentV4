from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .configs import (
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_EMA_ALPHA_CFG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_POSTPROCESS_CFG,
    DEFAULT_FG_RETRIEVAL_ALPHA,
    DEFAULT_TEXT_RETRIEVAL_WEIGHT,
)
from .data.datasets import MultiOrganDataset
from .models.segmentation_model import TaskSegmentModel
from .training.evaluation import logits_to_postprocessed_binary_mask
from .training.metrics import binary_dice, binary_hd95, binary_iou
from .training.retrieval import EMATaskMemoryBank, build_support_bank, select_task_tokens_for_query


def _safe_torch_load(path: str, map_location: Union[str, torch.device]) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _validate_checkpoint_schema(checkpoint: object, model_path: str) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint schema 错误: {model_path} 不是 dict。")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint schema 错误: {model_path} 缺少 model_state_dict。")
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError(f"checkpoint schema 错误: model_state_dict 不是 dict。")
    optional_dict_keys = ["model_config", "train_config", "ema_memory_bank"]
    for key in optional_dict_keys:
        if key in checkpoint and not isinstance(checkpoint[key], dict):
            raise ValueError(f"checkpoint schema 错误: {key} 应该是 dict。")
    return checkpoint




def _debug_value_to_jsonable(value: Any) -> Any:
    """Convert tensors / numpy arrays in debug dict to JSON-friendly values."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_debug_value_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _debug_value_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(_debug_value_to_jsonable(value), ensure_ascii=False)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Preserve first-row order, then append any extra keys discovered later.
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_case_rankings(save_dir: str, case_records: List[Dict[str, Any]], top_k: int = 20) -> None:
    """Save all per-case metrics and per-domain worst-case rankings."""
    _write_csv(os.path.join(save_dir, "per_case_metrics.csv"), case_records)

    organs = sorted({str(row["organ"]) for row in case_records})
    for organ in organs:
        sub = [row for row in case_records if row["organ"] == organ]
        worst_dice = sorted(sub, key=lambda r: float(r["dice"]))[:top_k]
        worst_hd95 = sorted(sub, key=lambda r: float(r["hd95"]), reverse=True)[:top_k]
        largest_fp = sorted(sub, key=lambda r: int(r["fp_area"]), reverse=True)[:top_k]
        largest_fn = sorted(sub, key=lambda r: int(r["fn_area"]), reverse=True)[:top_k]
        _write_csv(os.path.join(save_dir, f"{organ}_worst_dice_top{top_k}.csv"), worst_dice)
        _write_csv(os.path.join(save_dir, f"{organ}_worst_hd95_top{top_k}.csv"), worst_hd95)
        _write_csv(os.path.join(save_dir, f"{organ}_largest_fp_top{top_k}.csv"), largest_fp)
        _write_csv(os.path.join(save_dir, f"{organ}_largest_fn_top{top_k}.csv"), largest_fn)

    global_worst_dice = sorted(case_records, key=lambda r: float(r["dice"]))[:top_k]
    global_worst_hd95 = sorted(case_records, key=lambda r: float(r["hd95"]), reverse=True)[:top_k]
    _write_csv(os.path.join(save_dir, f"global_worst_dice_top{top_k}.csv"), global_worst_dice)
    _write_csv(os.path.join(save_dir, f"global_worst_hd95_top{top_k}.csv"), global_worst_hd95)


def denormalize_for_vis(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np = ((img_np + 1.0) * 127.5).clip(0, 255)
    return img_np.astype(np.uint8)


def to_heatmap_np(x: torch.Tensor) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = x.squeeze().detach().cpu().numpy()
    x = np.nan_to_num(x)
    return np.clip(x, 0.0, 1.0)


def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, EMATaskMemoryBank, Dict[str, Any]]:
    checkpoint = _validate_checkpoint_schema(_safe_torch_load(model_path, map_location=device), model_path)
    cfg = dict(DEFAULT_MODEL_CONFIG)
    cfg.update(checkpoint.get("model_config", {}))
    cfg.pop("decoder_num_classes", None)
    model = TaskSegmentModel(**cfg).to(device)
    raw_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    compatible_state = {}
    skipped_shape_mismatch = []
    for key, value in raw_state.items():
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape_mismatch.append(key)
            continue
        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys or skipped_shape_mismatch:
        print(
            "[info] checkpoint 以 compatible non-strict 方式加载："
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)} "
            f"shape_skipped={len(skipped_shape_mismatch)}"
        )
    model.eval()
    ema_bank = EMATaskMemoryBank()
    if "ema_memory_bank" in checkpoint:
        ema_bank.load_state_dict(checkpoint["ema_memory_bank"])
    return model, ema_bank, checkpoint.get("train_config", {})


def save_interpretability_panel(
    save_path: str,
    img_np: np.ndarray,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    out_dict: Dict[str, torch.Tensor],
    dice_val: float,
    alpha: float,
    hd95_val: Optional[float] = None,
) -> None:
    fg_prob = torch.softmax(out_dict["pred_masks"], dim=1)[:, 1].squeeze(0).detach().cpu().numpy()

    # Error map: 0=background, 1=TP, 2=FP, 3=FN.
    # This makes over-segmentation and missed lesion regions visible at a glance.
    gt_bool = gt_np.astype(bool)
    pred_bool = pred_np.astype(bool)
    err = np.zeros_like(gt_np, dtype=np.uint8)
    err[np.logical_and(pred_bool, gt_bool)] = 1
    err[np.logical_and(pred_bool, ~gt_bool)] = 2
    err[np.logical_and(~pred_bool, gt_bool)] = 3

    panels = [
        ("Image", img_np, "gray"),
        ("GT", gt_np, "gray"),
        ("Pred", pred_np, "gray"),
        ("Error 1=TP 2=FP 3=FN", err, "viridis"),
        ("FG prob", fg_prob, "viridis"),
        ("Guidance", to_heatmap_np(out_dict.get("dense_guidance_map")), "magma"),
    ]
    n = len(panels)
    plt.figure(figsize=(3.2 * n, 3.4))
    for i, (title, arr, cmap) in enumerate(panels, 1):
        plt.subplot(1, n, i)
        if arr is None:
            arr = np.zeros_like(gt_np, dtype=np.float32)
        plt.imshow(arr, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    hd95_text = "nan" if hd95_val is None else f"{hd95_val:.2f}"
    plt.suptitle(f"Dice={dice_val:.3f} | HD95={hd95_text} | EMA α={alpha:.3f}", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def predict_all_organs(
    model: torch.nn.Module,
    test_dataset: MultiOrganDataset,
    train_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    ema_bank: EMATaskMemoryBank,
    save_dir: str = "./vis",
    max_support_per_domain: int = 128,
    ensemble_k: int = 8,
    temperature: float = 0.15,
    coarse_pool_size: int = 16,
    retrieval_alpha: float = DEFAULT_FG_RETRIEVAL_ALPHA,
    text_retrieval_weight: float = DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    ema_alpha_cfg: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[Dict[str, Union[int, bool]]] = None,
    use_postprocess: bool = True,
    num_vis_per_organ: int = 5,
    vis_data_indices: Optional[List[int]] = None,
    guidance_scale: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    os.makedirs(save_dir, exist_ok=True)
    if guidance_scale is not None:
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "dense_guidance_strength"):
            raise ValueError("model.decoder.dense_guidance_strength not found; cannot apply guidance_scale.")
        model.decoder.dense_guidance_strength = float(guidance_scale)
    active_guidance_scale = float(getattr(getattr(model, "decoder", None), "dense_guidance_strength", 0.0))
    if ema_alpha_cfg is None:
        ema_alpha_cfg = dict(DEFAULT_EMA_ALPHA_CFG)
    if domain_thresholds is None:
        domain_thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS)
    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CFG)

    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    case_records: List[Dict[str, Any]] = []
    vis_data_idx_set = set(int(x) for x in (vis_data_indices or []))

    with torch.inference_mode():
        support_bank = build_support_bank(
            model=model,
            support_dataset=train_dataset,
            device=device,
            text_bank=text_bank,
            max_support_per_domain=max_support_per_domain,
        )

        for organ in list(test_dataset.organ_to_indices.keys()):
            organ_dir = os.path.join(save_dir, organ)
            os.makedirs(organ_dir, exist_ok=True)
            indices = test_dataset.get_organ_samples(organ, num_samples=len(test_dataset.organ_to_indices[organ]))
            dices, ious, hd95s = [], [], []
            for sample_i, data_idx in enumerate(tqdm(indices, desc=f"测试 {organ}")):
                img, mask, _ = test_dataset[data_idx]
                xq = img.unsqueeze(0).to(device)
                yq = mask.unsqueeze(0).long().to(device)

                task_tokens, query_pyramid, debug = select_task_tokens_for_query(
                    model=model,
                    query_img=xq,
                    organ_name=organ,
                    support_bank=support_bank,
                    device=device,
                    ensemble_k=ensemble_k,
                    temperature=temperature,
                    coarse_pool_size=coarse_pool_size,
                    fg_retrieval_alpha=retrieval_alpha,
                    text_retrieval_weight=text_retrieval_weight,
                    text_bank=text_bank,
                    ema_memory_bank=ema_bank,
                    ema_alpha_cfg=ema_alpha_cfg,
                    return_debug=True,
                )
                out = model.segment_with_task(query_feats=query_pyramid, task_tokens=task_tokens, output_size=xq.shape[-2:])
                logits = out["pred_masks"]

                pred = logits_to_postprocessed_binary_mask(
                    logits,
                    organ_name=organ,
                    domain_thresholds=domain_thresholds,
                    postprocess_cfg=postprocess_cfg,
                    use_postprocess=use_postprocess,
                )
                target = yq.squeeze(0).detach().cpu().numpy().astype(np.uint8)

                dice_val = binary_dice(pred, target)
                iou_val = binary_iou(pred, target)
                hd95_val = binary_hd95(pred, target)
                dices.append(dice_val)
                ious.append(iou_val)
                hd95s.append(hd95_val)

                pred_bool = pred.astype(bool)
                target_bool = target.astype(bool)
                tp_area = int(np.logical_and(pred_bool, target_bool).sum())
                fp_area = int(np.logical_and(pred_bool, ~target_bool).sum())
                fn_area = int(np.logical_and(~pred_bool, target_bool).sum())
                gt_area = int(target_bool.sum())
                pred_area = int(pred_bool.sum())
                precision = tp_area / max(tp_area + fp_area, 1)
                recall = tp_area / max(tp_area + fn_area, 1)

                selected_indices = debug.get("selected_indices")
                selected_scores = debug.get("selected_scores")
                case_records.append({
                    "organ": organ,
                    "sample_i": int(sample_i),
                    "data_idx": int(data_idx),
                    "dice": float(dice_val),
                    "iou": float(iou_val),
                    "hd95": float(hd95_val),
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "tp_area": tp_area,
                    "fp_area": fp_area,
                    "fn_area": fn_area,
                    "precision": float(precision),
                    "recall": float(recall),
                    "pred_gt_area_ratio": float(pred_area / max(gt_area, 1)),
                    "threshold": float(domain_thresholds.get(organ, domain_thresholds.get("default", 0.5))),
                    "use_postprocess": bool(use_postprocess),
                    "guidance_scale": float(active_guidance_scale),
                    "ema_alpha": float(debug.get("ema_alpha", 0.0) or 0.0),
                    "selected_indices": _safe_json_dumps(selected_indices),
                    "selected_scores": _safe_json_dumps(selected_scores),
                })

                should_save_default_vis = sample_i < num_vis_per_organ
                should_save_target_vis = int(data_idx) in vis_data_idx_set
                if should_save_default_vis or should_save_target_vis:
                    if should_save_target_vis:
                        file_name = (
                            f"{organ}_dataidx_{int(data_idx):04d}_sample_{int(sample_i):03d}"
                            f"_dice_{dice_val:.3f}_hd95_{hd95_val:.1f}.png"
                        )
                    else:
                        file_name = f"{organ}_{sample_i:03d}.png"
                    save_interpretability_panel(
                        os.path.join(organ_dir, file_name),
                        img_np=denormalize_for_vis(xq),
                        gt_np=target,
                        pred_np=pred,
                        out_dict=out,
                        dice_val=dice_val,
                        hd95_val=hd95_val,
                        alpha=float(debug.get("ema_alpha", 0.0) or 0.0),
                    )
            results[organ] = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "hd95": float(np.mean(hd95s))}
            print(f"[{organ}] Dice={results[organ]['dice']:.4f} IoU={results[organ]['iou']:.4f} HD95={results[organ]['hd95']:.2f}")

    _save_case_rankings(save_dir, case_records, top_k=20)
    found_vis_targets = {int(row["data_idx"]) for row in case_records if int(row["data_idx"]) in vis_data_idx_set}
    missing_vis_targets = sorted(vis_data_idx_set - found_vis_targets)
    if vis_data_idx_set:
        print(f"[info] targeted data_idx visualization saved for: {sorted(found_vis_targets)}")
        if missing_vis_targets:
            print(f"[warning] requested data_idx not found in test set: {missing_vis_targets}")
    print(f"[info] per-case metrics saved to: {os.path.join(save_dir, 'per_case_metrics.csv')}")
    return results


def predict(*args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
    return predict_all_organs(*args, **kwargs)