from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from ..configs import DEFAULT_TASK_IMPORTANCE_CFG
from .metrics import compute_per_sample_hard_dice


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    target_fg = (target == 1).float()
    intersection = (probs * target_fg).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def get_boundary_weight(mask: torch.Tensor, dilation: int = 2, boundary_weight: float = 5.0) -> torch.Tensor:
    # Force odd kernel size so max-pooling preserves spatial dimensions.
    if dilation % 2 == 0:
        dilation += 1
        
    mask_float = mask.float()
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask_float.dtype, device=mask_float.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask_float.dtype, device=mask_float.device).view(1, 1, 3, 3)
    edge_map = torch.sqrt(F.conv2d(mask_float, sobel_x, padding=1) ** 2 + F.conv2d(mask_float, sobel_y, padding=1) ** 2)
    if dilation > 1:
        edge_map = F.max_pool2d(edge_map, kernel_size=dilation, stride=1, padding=dilation // 2)
    edge_map = (edge_map > 0.1).float()
    return torch.ones_like(mask_float) + (boundary_weight - 1.0) * edge_map


def boundary_weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    target_onehot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2).float()
    if weight_map.ndim == 3:
        weight_map = weight_map.unsqueeze(1)
    if weight_map.ndim != 4:
        raise ValueError(f"weight_map must have shape [B, H, W] or [B, 1, H, W], but got {tuple(weight_map.shape)}")
    if weight_map.shape[1] == 1:
        weight_map = weight_map.expand(-1, target_onehot.shape[1], -1, -1)
    elif weight_map.shape[1] != target_onehot.shape[1]:
        raise ValueError(f"weight_map channel dimension error")
    return (-target_onehot * F.log_softmax(logits, dim=1) * weight_map).mean()


# 🌟 恢复：特征检索时必需的归一化函数
def normalize_descriptor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def split_task_tokens(task_tokens: torch.Tensor, num_visual_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return task_tokens[:, :1, :], task_tokens[:, 1:1 + num_visual_tokens, :], task_tokens[:, 1 + num_visual_tokens:, :]


def _soft_dice_from_probs(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = prob.float()
    target = target.float()
    intersection = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x is None:
        return None
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode=mode)
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False)


# 🌟 恢复：之前不小心删漏的辅助函数
def _build_boundary_target(mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
    weight = get_boundary_weight(mask, dilation=dilation, boundary_weight=5.0)
    boundary = (weight > 1.0).float()
    return boundary


def compute_dense_guidance_targets(target_mask: torch.Tensor, loss_cfg: Dict[str, float]) -> Dict[str, torch.Tensor]:
    del loss_cfg
    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    fg = (target_mask > 0).float()
    return {"guidance": fg}


# 🌟 修复保留：不再使用会导致 NaN 的 clamp 反模式，拥抱稳定的 BCEWithLogits
def _binary_map_loss(pred_logits: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_map)
    # 计算 Dice 时临时映射为概率
    prob_map = torch.sigmoid(pred_logits)
    dice = _soft_dice_from_probs(prob_map, target_map)
    return bce + dice


def compute_dense_guidance_supervision(output_dict: Dict[str, torch.Tensor], target_mask: torch.Tensor, loss_cfg: Dict[str, float]) -> torch.Tensor:
    key_map = {
        "guidance": "dense_guidance_map",
    }
    targets = compute_dense_guidance_targets(target_mask, loss_cfg)
    total = None
    for loss_key, out_key in key_map.items():
        weight = float(loss_cfg.get(loss_key, 0.0))
        pred_map = output_dict.get(out_key, None)
        if weight <= 0.0 or pred_map is None:
            continue
        tgt = _resize_like(targets[loss_key], pred_map, mode="nearest")
        loss = _binary_map_loss(pred_map, tgt)
        total = weight * loss if total is None else total + weight * loss
    if total is None:
        return target_mask.new_zeros(())
    return total


def compute_task_importance(
    pred_logits: torch.Tensor, query_target: torch.Tensor, support_mask: torch.Tensor,
    eps: float = 1e-6, cfg: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    cfg = DEFAULT_TASK_IMPORTANCE_CFG if cfg is None else cfg
    dice_weight = float(cfg.get("dice_weight", DEFAULT_TASK_IMPORTANCE_CFG["dice_weight"]))
    confidence_weight = float(cfg.get("confidence_weight", DEFAULT_TASK_IMPORTANCE_CFG["confidence_weight"]))
    support_size_weight = float(cfg.get("support_size_weight", DEFAULT_TASK_IMPORTANCE_CFG["support_size_weight"]))
    support_size_norm = float(cfg.get("support_size_norm", DEFAULT_TASK_IMPORTANCE_CFG["support_size_norm"]))
    min_importance = float(cfg.get("min_importance", DEFAULT_TASK_IMPORTANCE_CFG["min_importance"]))

    with torch.no_grad():
        per_sample_dice = compute_per_sample_hard_dice(pred_logits, query_target, eps=1.0)
        confidence = torch.softmax(pred_logits, dim=1).amax(dim=1).mean(dim=(1, 2))
        support_size_score = torch.clamp(
            support_mask.float().mean(dim=(1, 2, 3)) / support_size_norm,
            max=1.0,
        )
        importance = (
            dice_weight * per_sample_dice
            + confidence_weight * confidence
            + support_size_weight * support_size_score
        ).clamp_min(min_importance)
        return importance / importance.sum().clamp_min(eps)