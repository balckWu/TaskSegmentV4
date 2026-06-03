from __future__ import annotations

import os
from typing import Dict, Optional, Union

import torch


def _safe_torch_load(path: str, map_location: Union[str, torch.device] = "cpu"):
    """Load trusted tensor/checkpoint files with weights_only when available."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch does not support weights_only.
        return torch.load(path, map_location=map_location)


def load_text_bank(
    text_dir: str,
    device: torch.device,
    expected_text_dim: Optional[int] = None,
    strict_dim: bool = True,
) -> Dict[str, torch.Tensor]:
    mapping = {
        "thyroid": "text_features_thyroid.pt",
        "TN3K": "text_features_TN3K.pt",
        "BUSI_WHU": "text_features_BUSI_WHU.pt",
        "BUS-BRA": "text_features_BUS-BRA.pt",
        "OTU": "text_features_OTU.pt",       # 新加入的 OTU
        "prostate": "text_features_prostate.pt"  # 新加入的 前列腺
    }
    text_bank: Dict[str, torch.Tensor] = {}
    if not os.path.exists(text_dir):
        return text_bank

    for domain, fname in mapping.items():
        path = os.path.join(text_dir, fname)
        if not os.path.exists(path):
            continue

        obj = _safe_torch_load(path, map_location="cpu")
        if not isinstance(obj, dict) or "text_features" not in obj:
            raise ValueError(
                f"文本特征文件 schema 错误: {path}. "
                "期望是包含 key='text_features' 的 dict。"
            )

        feats = obj["text_features"]
        if not torch.is_tensor(feats):
            raise TypeError(f"文本特征 {path} 的 text_features 不是 torch.Tensor。")

        feats = feats.float()
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        if feats.ndim == 3:
            feats = feats.reshape(-1, feats.shape[-1])
        if feats.ndim != 2:
            raise ValueError(f"文本特征 {path} 维度非法: shape={tuple(feats.shape)}，期望 [N, D]。")

        if expected_text_dim is not None and feats.shape[-1] != int(expected_text_dim):
            msg = (
                f"文本特征维度不匹配: domain={domain}, file={path}, "
                f"got={feats.shape[-1]}, expected={int(expected_text_dim)}. "
                "请重新生成 text_features，或用 --text-dim 指定正确维度。"
            )
            if strict_dim:
                raise ValueError(msg)
            print(f"[警告] {msg}")
            continue

        text_bank[domain] = feats.to(device)
    return text_bank


def make_Ftext_batch(text_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
    return text_feats.unsqueeze(0).repeat(batch_size, 1, 1)
