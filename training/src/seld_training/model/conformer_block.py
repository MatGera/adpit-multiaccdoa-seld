"""Conformer block: Feed-Forward + Multi-Head Self-Attention + Convolution module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    """Single Conformer block following the Macaron-Net structure.

    Architecture: FFN(1/2) -> MHSA -> Conv -> FFN(1/2) -> LayerNorm

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        conv_kernel_size: Depthwise convolution kernel size.
        ff_expansion: Feed-forward expansion factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        conv_kernel_size: int = 51,
        ff_expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # First FFN (half-step)
        self.ffn1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ffn1_norm = nn.LayerNorm(d_model)

        # Multi-Head Self-Attention
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mhsa_norm = nn.LayerNorm(d_model)

        # Convolution module
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.conv_norm = nn.LayerNorm(d_model)

        # Second FFN (half-step)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, d_model).

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        # FFN half-step 1
        residual = x
        x = self.ffn1_norm(x)
        x = residual + 0.5 * self.dropout(self.ffn1(x))

        # MHSA
        residual = x
        x = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(x, x, x)
        x = residual + self.dropout(attn_out)

        # Convolution module
        residual = x
        x = self.conv_norm(x)
        x = residual + self.dropout(self.conv_module(x))

        # FFN half-step 2
        residual = x
        x = self.ffn2_norm(x)
        x = residual + 0.5 * self.dropout(self.ffn2(x))

        # Final LayerNorm
        x = self.final_norm(x)

        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with expansion and GLU-like activation."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConvolutionModule(nn.Module):
    """Convolution module with pointwise + depthwise + pointwise pattern."""

    def __init__(self, d_model: int, kernel_size: int = 51, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B, T, D) -> (B, T, D)."""
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x
