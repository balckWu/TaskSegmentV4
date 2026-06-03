from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..configs import DEFAULT_EMA_ALPHA_CFG, DEFAULT_FG_RETRIEVAL_ALPHA, DEFAULT_TEXT_RETRIEVAL_WEIGHT
from ..text.bank import make_Ftext_batch
from .losses import normalize_descriptor, split_task_tokens


def make_balanced_domain_schedule(domains: List[str], total_iters: int) -> List[str]:
    if len(domains) == 0:
        raise ValueError("domains cannot be empty")
    reps, rem = total_iters // len(domains), total_iters % len(domains)
    schedule = []
    for _ in range(reps):
        chunk = list(domains)
        random.shuffle(chunk)
        schedule.extend(chunk)
    if rem > 0:
        tail = list(domains)
        random.shuffle(tail)
        schedule.extend(tail[:rem])
    return schedule


class EMATaskMemoryBank:
    def __init__(self, momentum: float = 0.99):
        self.momentum, self.bank = float(momentum), {}

    def has(self, class_id: str) -> bool:
        return class_id in self.bank

    def get(self, class_id: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if class_id not in self.bank:
            return None
        return self.bank[class_id].to(device) if device else self.bank[class_id]

    @torch.no_grad()
    def update(self, class_id: str, task_tokens: torch.Tensor, importance: Optional[torch.Tensor] = None) -> None:
        weights = importance / importance.sum().clamp_min(1e-6) if importance is not None else torch.full((task_tokens.shape[0],), 1.0 / task_tokens.shape[0], device=task_tokens.device)
        prototype = (task_tokens.detach() * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True).cpu()
        self.bank[class_id] = self.momentum * self.bank[class_id] + (1.0 - self.momentum) * prototype if class_id in self.bank else prototype

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().cpu() for k, v in self.bank.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.bank = {k: v.clone().cpu() for k, v in state_dict.items()}

    def summary(self) -> str:
        return ", ".join([f"{k}: {tuple(v.shape)}" for k, v in self.bank.items()]) if self.bank else "empty"


def select_support_indices(dataset, organ_name, max_support_per_domain: Optional[int], seed: int = 42):
    indices = dataset.organ_to_positive_indices.get(organ_name, [])
    if not indices:
        return []
    if max_support_per_domain is None or len(indices) <= max_support_per_domain:
        return list(indices)
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(indices, max_support_per_domain, replace=False).tolist())


@torch.no_grad()
def build_support_bank(
    model,
    support_dataset,
    device,
    text_bank,
    max_support_per_domain: Optional[int] = 128,
    seed: int = 42,
):
    model.eval()
    support_bank = defaultdict(list)
    for organ_name in support_dataset.organ_to_positive_indices.keys():
        organ_text = text_bank.get(organ_name)
        for idx in select_support_indices(support_dataset, organ_name, max_support_per_domain, seed):
            img, mask, _ = support_dataset[idx]
            xs = img.unsqueeze(0).to(device)
            ys = mask.unsqueeze(0).unsqueeze(1).float().to(device)
            Ftext = make_Ftext_batch(organ_text, 1) if organ_text is not None else None

            support_pyramid = model.encode_image_pyramid(xs)
            task_dict = model.encode_task(xs=xs, ys=ys, Ftext=Ftext, support_pyramid=support_pyramid)
            fg_token, _, text_tokens = split_task_tokens(task_dict["task_tokens"], model.task_encoder.num_query_tokens)

            fallback_text = torch.zeros(1, model.task_encoder.hidden_dim).cpu()
            text_descriptor = normalize_descriptor(text_tokens.mean(dim=1)).detach().cpu() if text_tokens.shape[1] > 0 else fallback_text

            support_bank[organ_name].append({
                "task_tokens": task_dict["task_tokens"].detach().cpu(),
                "global_descriptor": normalize_descriptor(task_dict["support_feats"].mean(dim=(2, 3))).detach().cpu(),
                "fg_descriptor": normalize_descriptor(fg_token.squeeze(1)).detach().cpu(),
                "text_descriptor": text_descriptor,
                "index": idx,
            })
    return support_bank


def _resolve_fixed_ema_alpha(organ_name: str, ema_alpha_cfg: Optional[Dict[str, Union[float, Dict[str, float]]]]) -> float:
    default_alpha = DEFAULT_EMA_ALPHA_CFG.get(organ_name, 0.30)
    if ema_alpha_cfg is None:
        raw = default_alpha
    else:
        raw = ema_alpha_cfg.get(organ_name, default_alpha)
    if isinstance(raw, dict):
        raw = raw.get("prior", default_alpha)
    return float(np.clip(raw, 0.0, 1.0))


def get_projected_text_descriptor(model, text_feats, device):
    if text_feats is None:
        return None
    if text_feats.ndim == 2:
        text_feats = text_feats.unsqueeze(0)
    projected = model.task_encoder.project_text_tokens(text_feats.to(device))
    if projected is None:
        return None
    return normalize_descriptor(projected.mean(dim=1))


def select_task_tokens_for_query(
    model,
    query_img,
    organ_name,
    support_bank,
    device,
    ensemble_k: int = 8,
    coarse_pool_size: int = 16,
    fg_retrieval_alpha: float = DEFAULT_FG_RETRIEVAL_ALPHA,
    text_retrieval_weight: float = DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    temperature: float = 0.15,
    text_bank: Optional[Dict[str, torch.Tensor]] = None,
    ema_memory_bank: Optional[EMATaskMemoryBank] = None,
    ema_alpha_cfg=None,
    return_debug: bool = False,
    exclude_indices: Optional[set] = None,
    query_pyramid=None, # 新增参数支持预计算特征传入
):
    if query_pyramid is None:
        if query_img is None:
            raise ValueError("Either query_img or query_pyramid must be provided.")
        query_pyramid = model.encode_query(query_img)
    query_feats = query_pyramid["bottleneck"]
    retrieved_task_tokens, selected_scores, selected_indices = None, None, None

    if organ_name in support_bank and len(support_bank[organ_name]) > 0:
        if exclude_indices is not None:
            entries = [e for e in support_bank[organ_name] if e.get("index") not in exclude_indices]
        else:
            entries = support_bank[organ_name]

        if len(entries) > 0:
            query_global = normalize_descriptor(query_feats.mean(dim=(2, 3)))
            query_text = None
            if text_bank is not None and organ_name in text_bank:
                organ_text = text_bank[organ_name].to(device)
                query_text = get_projected_text_descriptor(model, organ_text, device)

            bank_global = torch.cat([e["global_descriptor"] for e in entries], dim=0).to(device)
            coarse_sims = F.cosine_similarity(query_global, bank_global, dim=1)
            if query_text is not None:
                bank_text = torch.cat([e["text_descriptor"] for e in entries], dim=0).to(device)
                coarse_sims = (1.0 - text_retrieval_weight) * coarse_sims + text_retrieval_weight * F.cosine_similarity(query_text, bank_text, dim=1)

            coarse_candidate_indices = torch.topk(
                coarse_sims,
                k=min(len(entries), max(coarse_pool_size, ensemble_k * 4)),
                largest=True,
            ).indices.tolist()

            coarse_task_tokens = entries[coarse_candidate_indices[0]]["task_tokens"].to(device)
            coarse_out = model.segment_with_task(
                query_feats=query_pyramid,
                task_tokens=coarse_task_tokens,
                output_size=query_img.shape[-2:] if query_img is not None else query_feats.shape[-2:],
            )
            coarse_pred = coarse_out["pred_masks"]
            pseudo_prob = torch.softmax(coarse_pred, dim=1)[:, 1:2]
            pseudo_prob_lowres = F.interpolate(pseudo_prob, size=query_feats.shape[-2:], mode="bilinear", align_corners=False)
            query_fg = normalize_descriptor(model.task_encoder.masked_average(query_feats, pseudo_prob_lowres))
            bank_fg = torch.cat([e["fg_descriptor"] for e in entries], dim=0).to(device)

            cand_global = coarse_sims[coarse_candidate_indices]
            cand_fg = F.cosine_similarity(query_fg, bank_fg[coarse_candidate_indices], dim=1)
            if query_text is not None:
                bank_text_all = torch.cat([e["text_descriptor"] for e in entries], dim=0).to(device)
                cand_text = F.cosine_similarity(query_text, bank_text_all[coarse_candidate_indices], dim=1)
            else:
                cand_text = torch.zeros_like(cand_fg)

            selected_scores_all = (
                (1.0 - text_retrieval_weight)
                * ((1.0 - fg_retrieval_alpha) * cand_global + fg_retrieval_alpha * cand_fg)
                + text_retrieval_weight * cand_text
            )

            topk = min(ensemble_k, len(coarse_candidate_indices))
            selected_local = torch.topk(selected_scores_all, k=topk, largest=True).indices.tolist()
            selected_indices = [coarse_candidate_indices[i] for i in selected_local]
            selected_scores = selected_scores_all[selected_local]
            weights = torch.softmax(selected_scores / max(temperature, 1e-4), dim=0).view(-1, 1, 1)
            retrieved_task_tokens = (torch.stack([entries[i]["task_tokens"].squeeze(0) for i in selected_indices], dim=0).to(device) * weights).sum(dim=0, keepdim=True)

    ema_task_tokens = ema_memory_bank.get(organ_name, device=device) if ema_memory_bank and ema_memory_bank.has(organ_name) else None
    ema_alpha = None

    if ema_task_tokens is not None and retrieved_task_tokens is not None:
        ema_alpha = _resolve_fixed_ema_alpha(organ_name, ema_alpha_cfg)
        task_tokens = ema_alpha * ema_task_tokens + (1.0 - ema_alpha) * retrieved_task_tokens
    elif ema_task_tokens is not None:
        ema_alpha = 1.0
        task_tokens = ema_task_tokens
    elif retrieved_task_tokens is not None:
        ema_alpha = 0.0
        task_tokens = retrieved_task_tokens
    else:
        raise ValueError(f"No EMA prototype or support retrieval available for organ/domain: {organ_name}")

    debug = {
        "ema_alpha": ema_alpha,
        "selected_scores": selected_scores.detach().cpu() if selected_scores is not None else None,
        "selected_indices": selected_indices,
    }
    return (task_tokens, query_pyramid, debug) if return_debug else (task_tokens, query_pyramid)