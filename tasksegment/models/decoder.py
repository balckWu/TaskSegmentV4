from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import build_2d_norm
from .dense_guidance import ConvNormAct, TextInducedDenseGuidance2D


class SkipFusionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, task_dim, norm_type="group", group_norm_groups=16):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            build_2d_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            build_2d_norm(out_channels, norm_type=norm_type, group_norm_groups=group_norm_groups),
        )
        self.skip_gate = nn.Sequential(
            nn.Linear(task_dim, out_channels),
            nn.Sigmoid(),
        )
        self.fuse = ConvNormAct(
            out_channels * 2,
            out_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x, skip, task_summary, spatial_guidance=None, spatial_strength: float = 1.0):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.skip_proj(skip)
        gate = self.skip_gate(task_summary).unsqueeze(-1).unsqueeze(-1)
        skip = skip * (1.0 + gate)

        if spatial_guidance is not None:
            if spatial_guidance.shape[-2:] != skip.shape[-2:]:
                spatial_guidance = F.interpolate(
                    spatial_guidance,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            spatial_guidance = spatial_guidance.clamp(0.0, 1.0)
            skip = skip * (1.0 + spatial_strength * spatial_guidance)
            x = x * (1.0 + spatial_strength * spatial_guidance)

        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class IrisMaskDecoder2D(nn.Module):
    """
    Task-conditioned multi-scale decoder.

    Uses query-to-task cross-attention, task-conditioned skip fusion, and dense text guidance.
    Dense guidance is regularized during training with dropout, random scaling, and bounded modulation.
    """

    def __init__(
        self,
        hidden_dim=1024,
        num_heads=8,
        dropout=0.1,
        fg_token_weight=0.65,
        num_query_tokens=4,
        use_ftext=True,
        norm_type="group",
        group_norm_groups=16,
        encoder_filters=(64, 128, 256, 512, 1024),
        use_dense_text_guidance=True,
        dense_guidance_dim=256,
        dense_guidance_strength=0.75,
        guidance_dropout_prob=0.15,
        guidance_scale_min=0.50,
        guidance_scale_max=1.00,
        guidance_clamp_max=0.80,
        text_group_summary_count=3,
    ):
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.use_ftext = bool(use_ftext)
        self.dropout = nn.Dropout(dropout)
        self.use_dense_text_guidance = bool(use_dense_text_guidance and use_ftext)
        self.dense_guidance_strength = float(dense_guidance_strength)
        self.guidance_dropout_prob = float(guidance_dropout_prob)
        self.guidance_scale_min = float(guidance_scale_min)
        self.guidance_scale_max = float(guidance_scale_max)
        self.guidance_clamp_max = float(guidance_clamp_max)
        self.text_group_summary_count = int(text_group_summary_count)

        self.query_to_task = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)

        self.task_summary_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        f1, f2, f3, f4, f5 = encoder_filters
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(f5, f5, kernel_size=3, padding=1, bias=False),
            build_2d_norm(f5, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
        )

        if self.use_dense_text_guidance:
            self.dense_guidance = TextInducedDenseGuidance2D(
                in_channels=f5,
                text_dim=hidden_dim,
                guidance_dim=dense_guidance_dim,
                text_group_summary_count=text_group_summary_count,
                norm_type=norm_type,
                group_norm_groups=group_norm_groups,
            )
        else:
            self.dense_guidance = None

        self.up4 = SkipFusionBlock(f5, f4, f4, hidden_dim, norm_type=norm_type, group_norm_groups=group_norm_groups)
        self.up3 = SkipFusionBlock(f4, f3, f3, hidden_dim, norm_type=norm_type, group_norm_groups=group_norm_groups)
        self.up2 = SkipFusionBlock(f3, f2, f2, hidden_dim, norm_type=norm_type, group_norm_groups=group_norm_groups)
        self.up1 = SkipFusionBlock(f2, f1, f1, hidden_dim, norm_type=norm_type, group_norm_groups=group_norm_groups)

        self.head = nn.Sequential(
            nn.Conv2d(f1, f1, kernel_size=3, padding=1, bias=False),
            build_2d_norm(f1, norm_type=norm_type, group_norm_groups=group_norm_groups),
            nn.GELU(),
            nn.Conv2d(f1, 2, kernel_size=1),
        )
        self.fg_token_weight = float(fg_token_weight)

    def _split_task_tokens(self, task_tokens):
        foreground_token = task_tokens[:, 0, :]
        visual_start = 1
        visual_end = 1 + self.num_query_tokens
        visual_context_tokens = task_tokens[:, visual_start:visual_end, :]
        text_tokens = task_tokens[:, visual_end:, :] if self.use_ftext and task_tokens.shape[1] > visual_end else None
        return foreground_token, visual_context_tokens, text_tokens

    @staticmethod
    def _unpack_query_feats(query_feats):
        if isinstance(query_feats, dict):
            return (
                query_feats["bottleneck"],
                query_feats.get("skip1", None),
                query_feats.get("skip2", None),
                query_feats.get("skip3", None),
                query_feats.get("skip4", None),
            )
        return query_feats, None, None, None, None

    def _summarize_task(self, task_tokens):
        foreground_token, visual_context_tokens, text_tokens = self._split_task_tokens(task_tokens)
        visual_summary = (
            visual_context_tokens.mean(dim=1)
            if visual_context_tokens.shape[1] > 0
            else foreground_token
        )

        fused_summary = self.task_summary_proj(torch.cat([foreground_token, visual_summary], dim=1))
        task_summary = self.fg_token_weight * foreground_token + (1.0 - self.fg_token_weight) * fused_summary
        return task_summary, text_tokens

    def forward(self, query_feats, task_tokens):
        bottleneck, skip1, skip2, skip3, skip4 = self._unpack_query_feats(query_feats)
        query_flat = bottleneck.flatten(2).transpose(1, 2)

        query_update, _ = self.query_to_task(query_flat, task_tokens, task_tokens)
        query_flat = self.query_norm(query_flat + self.dropout(query_update))
        x = query_flat.transpose(1, 2).reshape_as(bottleneck)
        x = self.bottleneck_proj(x)

        task_summary, text_tokens = self._summarize_task(task_tokens)

        dense_guidance_lowres = None
        dense_guidance_for_mod = None
        guidance_strength = float(self.dense_guidance_strength)

        if self.dense_guidance is not None and text_tokens is not None and text_tokens.shape[1] > 0:
            dense_out = self.dense_guidance(x, text_tokens)
            dense_guidance_lowres = dense_out["guidance_map"]

            # Training-only regularization: keep the dense guidance branch, but prevent the
            # decoder from becoming fully dependent on an occasionally mislocalized cue.
            # During inference this block is deterministic and uses dense_guidance_strength.
            if self.training:
                if torch.rand((), device=x.device).item() < self.guidance_dropout_prob:
                    guidance_strength = 0.0
                else:
                    scale_min = min(self.guidance_scale_min, self.guidance_scale_max)
                    scale_max = max(self.guidance_scale_min, self.guidance_scale_max)
                    rand_scale = torch.empty((), device=x.device).uniform_(scale_min, scale_max).item()
                    guidance_strength = guidance_strength * float(rand_scale)

            dense_guidance_for_mod = dense_guidance_lowres.clamp(0.0, self.guidance_clamp_max)

            x = x + guidance_strength * dense_out["guidance_feats"]
            x = x * (1.0 + guidance_strength * dense_guidance_for_mod)

        x = self.up4(x, skip4, task_summary, spatial_guidance=dense_guidance_for_mod, spatial_strength=guidance_strength) if skip4 is not None else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x, skip3, task_summary, spatial_guidance=dense_guidance_for_mod, spatial_strength=guidance_strength) if skip3 is not None else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x, skip2, task_summary, spatial_guidance=dense_guidance_for_mod, spatial_strength=guidance_strength) if skip2 is not None else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x, skip1, task_summary, spatial_guidance=dense_guidance_for_mod, spatial_strength=guidance_strength) if skip1 is not None else F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return {
            "logits": self.head(x),
            "dense_guidance_lowres": dense_guidance_lowres,
            "task_tokens": task_tokens,
        }
