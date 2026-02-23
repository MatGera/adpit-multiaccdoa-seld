"""Multi-ACCDOA output head: C classes x T tracks x 3 Cartesian DOA coordinates."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiACCDOAHead(nn.Module):
    """Multi-ACCDOA output head.

    Maps Conformer output to Multi-ACCDOA format:
    (B, T', d_model) -> (B, T', C, T_tracks, 3)

    where:
    - C = number of sound event classes
    - T_tracks = number of tracks per class (default 3)
    - 3 = Cartesian DOA coordinates (x, y, z)

    The vector norm ||v|| encodes detection confidence in [0, 1].
    Zero vectors indicate inactive tracks.

    Args:
        d_model: Input dimension from Conformer.
        num_classes: Number of event classes (C).
        num_tracks: Number of tracks per class (T).
    """

    def __init__(self, d_model: int = 512, num_classes: int = 13, num_tracks: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_tracks = num_tracks
        self.output_dim = num_classes * num_tracks * 3

        self.fc = nn.Linear(d_model, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Conformer output, shape (B, T', d_model).

        Returns:
            Multi-ACCDOA output, shape (B, T', C, T_tracks, 3).
            No activation — this is a regression output.
        """
        out = self.fc(x)  # (B, T', C*T*3)
        b, t, _ = out.shape
        out = out.reshape(b, t, self.num_classes, self.num_tracks, 3)
        return out
