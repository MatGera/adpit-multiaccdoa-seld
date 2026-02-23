"""Squeeze-and-Excitation (SE) block for channel attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Applies channel-wise attention by:
    1. Global average pooling (squeeze)
    2. Two FC layers with bottleneck (excitation)
    3. Sigmoid gating

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: (B, C, H, W) → (B, C)
        scale = self.squeeze(x).view(b, c)
        # Excite: (B, C) → (B, C)
        scale = self.excitation(scale).view(b, c, 1, 1)
        # Scale
        return x * scale
