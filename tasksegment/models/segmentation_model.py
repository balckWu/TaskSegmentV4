from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UNetEncoder2D
from .task_encoder import TaskEncoder2D
from .decoder import IrisMaskDecoder2D


class TaskSegmentModel(nn.Module):
    """Task-conditioned segmentation model."""

    def __init__(
        self,
        use_ftext=True,
        hidden_dim=1024,
        num_query_tokens=4,
        context_upscale=4,
        dropout=0.1,
        text_dim=1024,
        task_norm_type="group",
        group_norm_groups=16,
        encoder_batchnorm=True,
        fg_token_weight=0.65,
        use_dense_text_guidance=True,
        dense_guidance_dim=256,
        dense_guidance_strength=0.75,
        guidance_dropout_prob=0.15,
        guidance_scale_min=0.50,
        guidance_scale_max=1.00,
        guidance_clamp_max=0.80,
        text_group_summary_count=5,
    ):
        super().__init__()
        self.encoder = UNetEncoder2D(is_batchnorm=encoder_batchnorm)

        encoder_bottleneck_dim = self.encoder.filters[-1]
        if hidden_dim != encoder_bottleneck_dim:
            raise ValueError(
                f"hidden_dim={hidden_dim} must equal encoder bottleneck channels={encoder_bottleneck_dim} "
                "unless explicit projection layers are added. Please keep hidden_dim=1024."
            )

        self.use_ftext = bool(use_ftext)

        self.task_encoder = TaskEncoder2D(
            hidden_dim=hidden_dim,
            num_query_tokens=num_query_tokens,
            context_upscale=context_upscale,
            dropout=dropout,
            use_ftext=use_ftext,
            text_dim=text_dim,
            norm_type=task_norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.decoder = IrisMaskDecoder2D(
            hidden_dim=hidden_dim,
            num_query_tokens=num_query_tokens,
            use_ftext=use_ftext,
            dropout=dropout,
            fg_token_weight=fg_token_weight,
            norm_type=task_norm_type,
            group_norm_groups=group_norm_groups,
            encoder_filters=tuple(self.encoder.filters),
            use_dense_text_guidance=use_dense_text_guidance,
            dense_guidance_dim=dense_guidance_dim,
            dense_guidance_strength=dense_guidance_strength,
            guidance_dropout_prob=guidance_dropout_prob,
            guidance_scale_min=guidance_scale_min,
            guidance_scale_max=guidance_scale_max,
            guidance_clamp_max=guidance_clamp_max,
            text_group_summary_count=text_group_summary_count,
        )

    def encode_image(self, x):
        return self.encoder.forward_features(x)["bottleneck"]

    def encode_image_pyramid(self, x):
        return self.encoder.forward_features(x)

    def encode_task(self, xs, ys, Ftext=None, support_feats=None, support_pyramid=None):
        # 🌟 修复: 极简清晰的特征提取链条
        if support_pyramid is None and support_feats is None:
            support_pyramid = self.encoder.forward_features(xs)

        if support_feats is None:
            support_feats = support_pyramid["bottleneck"]

        task_tokens = self.task_encoder(support_feats, ys.float(), text_tokens=Ftext)
        return {
            "support_feats": support_feats,
            "support_pyramid": support_pyramid,
            "task_tokens": task_tokens,
        }

    def encode_query(self, xq):
        return self.encoder.forward_features(xq)

    @staticmethod
    def _maybe_resize_map(x: torch.Tensor | None, output_size):
        if x is None or output_size is None:
            return x
        if x.shape[-2:] == output_size:
            return x
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

    def segment_with_task(self, xq=None, task_tokens=None, query_feats=None, output_size=None):
        if task_tokens is None:
            raise ValueError("task_tokens must be provided.")

        if xq is not None:
            query_pyramid = self.encoder.forward_features(xq)
        elif query_feats is not None:
            query_pyramid = query_feats if isinstance(query_feats, dict) else {"bottleneck": query_feats}
        else:
            raise ValueError("Either xq or query_feats must be provided.")

        decoder_out = self.decoder(query_pyramid, task_tokens)
        logits = decoder_out["logits"]

        if output_size is None and xq is not None:
            output_size = xq.shape[-2:]
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return {
            "pred_masks": logits,
            "query_feats": query_pyramid["bottleneck"],
            "query_pyramid": query_pyramid,
            "dense_guidance_map": self._maybe_resize_map(decoder_out.get("dense_guidance_lowres"), output_size),
            "refined_task_tokens": decoder_out["task_tokens"],
        }

    def forward(self, xs, ys, Ftext=None, xq=None):
        task_dict = self.encode_task(xs=xs, ys=ys, Ftext=Ftext)
        pred_dict = self.segment_with_task(xq=xq, task_tokens=task_dict["task_tokens"])
        return {
            "pred_masks": pred_dict["pred_masks"],
            "task_tokens": task_dict["task_tokens"],
            "support_feats": task_dict["support_feats"],
            "support_pyramid": task_dict["support_pyramid"],
            "query_feats": pred_dict["query_feats"],
            "query_pyramid": pred_dict["query_pyramid"],
            "dense_guidance_map": pred_dict["dense_guidance_map"],
            "refined_task_tokens": pred_dict["refined_task_tokens"],
        }