from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from ..data.datasets import MultiOrganDataset
from ..configs import (
    DEFAULT_DOMAIN_THRESHOLDS,
    DEFAULT_POSTPROCESS_CFG,
    DEFAULT_FG_RETRIEVAL_ALPHA,
    DEFAULT_TEXT_RETRIEVAL_WEIGHT,
)
from .metrics import binary_dice, binary_hd95, binary_iou, compute_hard_dice, compute_hausdorff_distance95, compute_iou
from .retrieval import EMATaskMemoryBank, build_support_bank, select_task_tokens_for_query


def postprocess_mask(
    pred: np.ndarray,
    min_area: int = 64,
    keep_largest: bool = True,
    fill_holes: bool = True,
    closing_kernel: int = 3,
) -> np.ndarray:
    pred = (pred > 0).astype(np.uint8)
    if fill_holes:
        pred = binary_fill_holes(pred).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    if num_labels <= 1:
        out = pred
    else:
        areas = stats[:, cv2.CC_STAT_AREA]
        valid_ids = [i for i in range(1, num_labels) if areas[i] >= min_area]
        if len(valid_ids) == 0:
            out = pred
        else:
            out = np.zeros_like(pred)
            if keep_largest:
                best_id = max(valid_ids, key=lambda i: areas[i])
                out[labels == best_id] = 1
            else:
                for i in valid_ids:
                    out[labels == i] = 1
    if closing_kernel and closing_kernel > 1:
        kernel = np.ones((closing_kernel, closing_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(np.uint8)


def logits_to_postprocessed_binary_mask(
    logits: torch.Tensor,
    organ_name: str,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[Dict[str, Union[int, bool]]] = None,
    use_postprocess: bool = True,
) -> np.ndarray:
    if domain_thresholds is None:
        domain_thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS)
    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CFG)

    tau = float(domain_thresholds.get(organ_name, 0.50))
    fg_prob = torch.softmax(logits, dim=1)[:, 1].squeeze(0).detach().cpu().numpy()
    pred_bin = (fg_prob >= tau).astype(np.uint8)
    if use_postprocess:
        pred_bin = postprocess_mask(pred_bin, **postprocess_cfg)
    return pred_bin.astype(np.uint8)


def evaluate(
    model: torch.nn.Module,
    query_dataset: MultiOrganDataset,
    support_dataset: MultiOrganDataset,
    device: torch.device,
    text_bank: Dict[str, torch.Tensor],
    max_support_per_domain: Optional[int] = 128,
    ensemble_k: int = 8,
    fg_retrieval_alpha: float = DEFAULT_FG_RETRIEVAL_ALPHA,
    text_retrieval_weight: float = DEFAULT_TEXT_RETRIEVAL_WEIGHT,
    ema_memory_bank: Optional[EMATaskMemoryBank] = None,
    ema_alpha_cfg: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    domain_thresholds: Optional[Dict[str, float]] = None,
    postprocess_cfg: Optional[Dict[str, Union[int, bool]]] = None,
    use_postprocess: bool = True,
) -> Dict[str, Any]:
    model.eval()
    domain_metrics: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    if domain_thresholds is None:
        domain_thresholds = dict(DEFAULT_DOMAIN_THRESHOLDS)
    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CFG)

    with torch.inference_mode():
        support_bank = build_support_bank(model, support_dataset, device, text_bank, max_support_per_domain)

        for idx in tqdm(range(len(query_dataset)), desc="Validating", leave=False):
            img, label, organ_name = query_dataset[idx]
            xq = img.unsqueeze(0).to(device)
            yq = label.unsqueeze(0).to(device)
            task_tokens, query_pyramid = select_task_tokens_for_query(
                model,
                xq,
                organ_name,
                support_bank,
                device,
                ensemble_k,
                fg_retrieval_alpha=fg_retrieval_alpha,
                text_retrieval_weight=text_retrieval_weight,
                text_bank=text_bank,
                ema_memory_bank=ema_memory_bank,
                ema_alpha_cfg=ema_alpha_cfg,
            )
            pred_out = model.segment_with_task(query_feats=query_pyramid, task_tokens=task_tokens, output_size=xq.shape[-2:])
            pred = pred_out["pred_masks"]

            if use_postprocess:
                pred_bin = logits_to_postprocessed_binary_mask(
                    pred,
                    organ_name=organ_name,
                    domain_thresholds=domain_thresholds,
                    postprocess_cfg=postprocess_cfg,
                    use_postprocess=True,
                )
                target_bin = yq.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                domain_metrics[organ_name]["dice"].append(binary_dice(pred_bin, target_bin))
                domain_metrics[organ_name]["iou"].append(binary_iou(pred_bin, target_bin))
                domain_metrics[organ_name]["hd95"].append(binary_hd95(pred_bin, target_bin))
            else:
                domain_metrics[organ_name]["dice"].append(compute_hard_dice(pred, yq).item())
                domain_metrics[organ_name]["iou"].append(compute_iou(pred, yq).item())
                domain_metrics[organ_name]["hd95"].append(compute_hausdorff_distance95(pred, yq))

    per_domain = {
        k: {"dice": float(np.mean(v["dice"])), "iou": float(np.mean(v["iou"])), "hd95": float(np.mean(v["hd95"]))}
        for k, v in domain_metrics.items()
    }
    if not per_domain:
        raise RuntimeError("No validation results were produced.")
    return {
        "dice": float(np.mean([x["dice"] for x in per_domain.values()])),
        "iou": float(np.mean([x["iou"] for x in per_domain.values()])),
        "hd95": float(np.mean([x["hd95"] for x in per_domain.values()])),
        "per_domain": per_domain,
    }