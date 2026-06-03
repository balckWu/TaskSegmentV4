from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _max_possible_hd_from_shape(shape) -> float:
    """Return image diagonal as the finite HD penalty for empty/non-empty mismatch."""
    if len(shape) < 2:
        return 0.0
    h, w = int(shape[-2]), int(shape[-1])
    return float(np.hypot(h, w))


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Return 2D object boundary pixels for HD95 computation."""
    mask = (mask > 0).astype(bool)
    if not mask.any():
        return mask
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    surface = mask ^ eroded
    if not surface.any():
        surface = mask
    return surface


def _surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Bidirectional surface-to-surface distances in pixel units."""
    pred_surface = _binary_surface(pred)
    target_surface = _binary_surface(target)
    dist_to_target_surface = distance_transform_edt(~target_surface)
    dist_to_pred_surface = distance_transform_edt(~pred_surface)
    pred_to_target = dist_to_target_surface[pred_surface]
    target_to_pred = dist_to_pred_surface[target_surface]
    return np.concatenate([pred_to_target, target_to_pred]).astype(np.float32)


def binary_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th percentile Hausdorff distance between mask boundaries.

    Empty/empty returns 0.0. Empty/non-empty uses the image diagonal as a finite
    penalty, matching the previous HD behavior in this project.
    """
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return _max_possible_hd_from_shape(target.shape)
    distances = _surface_distances(pred, target)
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def compute_hausdorff_distance95(logits, target) -> float:
    """Mean symmetric HD95 over a batch, computed on predicted/target boundaries."""
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
    target_labels = target.detach().cpu().numpy()
    batch_hd95 = []
    for i in range(pred_labels.shape[0]):
        batch_hd95.append(binary_hd95(pred_labels[i], target_labels[i]))
    return float(np.mean(batch_hd95))


def compute_hard_dice(logits, target, eps=1.0):
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1)
    pred_fg = (pred_labels == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_per_sample_hard_dice(logits, target, eps=1.0):
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1)
    pred_fg = (pred_labels == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def compute_iou(logits, target, eps=1e-8):
    """Mean foreground IoU with correct empty/empty handling.

    If prediction and target are both empty for a sample, IoU is 1.0.
    If exactly one is empty, IoU is 0.0.
    """
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1)
    pred_fg = (pred_labels == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum(dim=(1, 2))
    union = pred_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2)) - intersection
    iou = torch.where(
        union > 0,
        intersection / (union + eps),
        torch.ones_like(union),
    )
    return iou.mean()


def compute_hausdorff_distance(logits, target):
    """Symmetric Hausdorff distance with image-diagonal empty mismatch penalty."""
    pred_labels = torch.softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
    target_labels = target.detach().cpu().numpy()
    batch_hd = []
    for i in range(pred_labels.shape[0]):
        p = (pred_labels[i] > 0).astype(np.uint8)
        t = (target_labels[i] > 0).astype(np.uint8)
        if p.sum() == 0 and t.sum() == 0:
            batch_hd.append(0.0)
            continue
        if p.sum() == 0 or t.sum() == 0:
            batch_hd.append(_max_possible_hd_from_shape(t.shape))
            continue
        dist_to_target = distance_transform_edt(1 - t)
        dist_to_pred = distance_transform_edt(1 - p)
        batch_hd.append(max(dist_to_target[p > 0].max(), dist_to_pred[t > 0].max()))
    return float(np.mean(batch_hd))


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1.0) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * inter + eps) / (union + eps)


def binary_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter = float((pred * target).sum())
    union = float(pred.sum() + target.sum() - inter)
    if union == 0.0:
        return 1.0
    return inter / (union + eps)


def binary_hd(pred: np.ndarray, target: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return _max_possible_hd_from_shape(target.shape)
    dist_to_target = distance_transform_edt(1 - target)
    dist_to_pred = distance_transform_edt(1 - pred)
    return float(max(dist_to_target[pred > 0].max(), dist_to_pred[target > 0].max()))
