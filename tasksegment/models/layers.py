from __future__ import annotations

import torch
import torch.nn as nn
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super().__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


def build_2d_norm(num_channels: int, norm_type: str = "group", group_norm_groups: int = 16) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = min(group_norm_groups, num_channels)
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")
