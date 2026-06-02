from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import build_2d_norm

class TaskEncoder2D(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        num_query_tokens=4,
        context_upscale=4,
        num_heads=8,
        dropout=0.1,
        use_ftext=True,
        text_dim=1024,
        norm_type="group",
        group_norm_groups=16,
    ):
        super().__init__()
        if hidden_dim % (context_upscale ** 2) != 0:
            raise ValueError("hidden_dim must be divisible by context_upscale^2 for PixelShuffle.")

        self.hidden_dim = int(hidden_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.context_upscale = int(context_upscale)
        self.use_ftext = bool(use_ftext)

        self.context_in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        shuffled_dim = hidden_dim // (context_upscale ** 2)

        self.context_fuse_conv = nn.Sequential(
            nn.Conv2d(shuffled_dim + 1, shuffled_dim, kernel_size=1),
            build_2d_norm(shuffled_dim, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        if self.use_ftext:
            self.ftext_proj = nn.Linear(text_dim, hidden_dim)
            self.ftext_norm = nn.LayerNorm(hidden_dim)

        self.query_norm1 = nn.LayerNorm(hidden_dim)
        self.query_norm2 = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.query_dropout = nn.Dropout(dropout)

        self.fg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _ensure_4d_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"support_mask must have 3 or 4 dims, but got shape {tuple(mask.shape)}")
        return mask

    @staticmethod
    def masked_average(feats: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        masked = feats * mask
        denom = mask.sum(dim=(2, 3), keepdim=False).clamp_min(eps)
        pooled = masked.sum(dim=(2, 3), keepdim=False) / denom
        return pooled

    @staticmethod
    def _unwrap_feats(feats):
        return feats["bottleneck"] if isinstance(feats, dict) else feats

    def project_text_tokens(self, text_tokens: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_ftext or text_tokens is None:
            return None
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(1)
        return self.ftext_norm(self.ftext_proj(text_tokens.float()))

    def forward(self, support_feats, support_mask, text_tokens=None):
        support_feats = self._unwrap_feats(support_feats)
        support_mask = self._ensure_4d_mask(support_mask).float()

        support_mask_lowres = F.interpolate(
            support_mask,
            size=support_feats.shape[-2:],
            mode="nearest",
        )

        raw_foreground = self.masked_average(support_feats, support_mask_lowres)
        foreground_token = self.fg_proj(raw_foreground).unsqueeze(1)

        high_res_feats = F.pixel_shuffle(self.context_in_proj(support_feats), self.context_upscale)
        high_res_mask = F.interpolate(
            support_mask,
            size=high_res_feats.shape[-2:],
            mode="nearest",
        )

        fused = torch.cat([high_res_feats, high_res_mask], dim=1)
        fused = self.context_fuse_conv(fused)
        fused = F.pixel_unshuffle(fused, self.context_upscale)
        fused_flat = fused.flatten(2).transpose(1, 2)

        batch_size = support_feats.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        ctx_tokens, _ = self.cross_attn(query_tokens, fused_flat, fused_flat)
        query_tokens = self.query_norm1(query_tokens + self.query_dropout(ctx_tokens))

        ffn_out = self.query_ffn(query_tokens)
        query_tokens = self.query_norm2(query_tokens + self.query_dropout(ffn_out))

        task_tokens = torch.cat([foreground_token, query_tokens], dim=1)
        
        # 🌟 彻底放行完整的 text_tokens 到下游
        projected_text_tokens = self.project_text_tokens(text_tokens)
        if projected_text_tokens is not None:
            task_tokens = torch.cat([task_tokens, projected_text_tokens], dim=1)
            
        return task_tokens