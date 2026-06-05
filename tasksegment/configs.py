from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

DEFAULT_MODEL_CONFIG = {
    "use_ftext": True,
    "hidden_dim": 1024,
    "num_query_tokens": 4,
    "context_upscale": 4,
    "dropout": 0.1,
    "text_dim": 1024,
    "task_norm_type": "group",
    "group_norm_groups": 16,
    "encoder_batchnorm": False,
    "fg_token_weight": 0.3,
    "use_dense_text_guidance": True,
    "dense_guidance_dim": 256,
    "dense_guidance_strength": 0.75,
    "guidance_dropout_prob": 0.15,
    "guidance_scale_min": 0.50,
    "guidance_scale_max": 1.00,
    "guidance_clamp_max": 0.80,
    "text_group_summary_count": 3,
}

# 🌟 新增了 OTU 和 prostate 的 EMA 权重
DEFAULT_EMA_ALPHA_CFG = {
    "thyroid": 0.40,
    "TN3K": 0.40,
    "BUSI_WHU": 0.35,
    "BUS-BRA": 0.35,
    "OTU": 0.40,
    "prostate": 0.40,
}

DEFAULT_TEXT_RETRIEVAL_WEIGHT = 0.40
DEFAULT_FG_RETRIEVAL_ALPHA = 0.65

DEFAULT_TASK_IMPORTANCE_CFG = {
    "dice_weight": 0.60,
    "confidence_weight": 0.25,
    "support_size_weight": 0.15,
    "support_size_norm": 0.08,
    "min_importance": 0.05,
}

DEFAULT_TRAIN_LOSS_WEIGHT_CFG = {
    "raw_aux": 0.25,
    "token_consistency": 0.05,
    "raw_dense_aux": 0.25,
    "dense_main": 0.50,
}

# 🌟 新增了 OTU 和 prostate 的阈值
DEFAULT_DOMAIN_THRESHOLDS = {
    "thyroid": 0.50,
    "TN3K": 0.50,
    "BUSI_WHU": 0.50,
    "BUS-BRA": 0.50,
    "OTU": 0.50,
    "prostate": 0.50,
}

DEFAULT_POSTPROCESS_CFG = {
    "min_area": 64,
    "keep_largest": False,
    "fill_holes": True,
    "closing_kernel": 3,
}

DEFAULT_DENSE_GUIDANCE_LOSS_CFG = {
    "guidance": 0.50,
}


@dataclass
class DataConfig:
    data_root: str = "./data"
    image_size: Tuple[int, int] = (512, 512)
    # 🌟 在这里将 OTU 和 prostate 加入了核心训练域
    domains: Tuple[str, ...] = ("thyroid", "TN3K", "BUSI_WHU", "BUS-BRA", "OTU", "prostate")

    def roots(self) -> Dict[str, str]:
        return {domain: os.path.join(self.data_root, domain) for domain in self.domains}


@dataclass
class TrainConfig:
    num_epochs: int = 150
    batch_size: int = 2
    lr: float = 5e-5
    weight_decay: float = 1e-4
    patience: int = 10
    max_support_per_domain: int = 128
    val_ensemble_k: int = 8
    episodes_per_epoch: int = 1500
    grad_accum_steps: int = 4
    ema_bank_momentum: float = 0.99
    warmup_epochs: int = 3
    boundary_loss_weight: float = 0.5
    train_loss_weight_cfg: dict = field(default_factory=lambda: dict(DEFAULT_TRAIN_LOSS_WEIGHT_CFG))
    task_importance_cfg: dict = field(default_factory=lambda: dict(DEFAULT_TASK_IMPORTANCE_CFG))
    save_path: str = "USTS_guidance_reg.pth"
    text_dir: str = "./text_features"
    model_config: dict = field(default_factory=lambda: dict(DEFAULT_MODEL_CONFIG))