from __future__ import annotations

import torch
import torch.nn as nn

from .layers import unetConv2


class UNetEncoder2D(nn.Module):
    """Plain 2D U-Net encoder that exposes bottleneck and multi-scale skip features."""

    def __init__(
        self,
        feature_scale: int = 1,
        in_channels: int = 1,
        is_batchnorm: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_scale = int(feature_scale)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.filters = filters

        self.pre_conv = nn.Conv2d(self.in_channels, filters[0], kernel_size=3, padding=1)

        self.conv1 = unetConv2(filters[0], filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(2)

        self.center = unetConv2(filters[3], filters[4], is_batchnorm)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pre_conv(x)

        conv1 = self.conv1(x)
        x1 = self.maxpool1(conv1)

        conv2 = self.conv2(x1)
        x2 = self.maxpool2(conv2)

        conv3 = self.conv3(x2)
        x3 = self.maxpool3(conv3)

        conv4 = self.conv4(x3)
        x4 = self.maxpool4(conv4)

        bottleneck = self.center(x4)

        return {
            "skip1": conv1,
            "skip2": conv2,
            "skip3": conv3,
            "skip4": conv4,
            "bottleneck": bottleneck,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["bottleneck"]

