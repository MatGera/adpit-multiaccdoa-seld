"""ADPIT (Auxiliary Duplicating Permutation Invariant Training) loss for Multi-ACCDOA."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn


class ADPITLoss(nn.Module):
    """Auxiliary Duplicating Permutation Invariant Training loss.

    For Multi-ACCDOA output (B, T, C, T_tracks, 3):

    For each class c:
      1. Compute all T_tracks! permutations of track assignments
      2. For each permutation, compute MSE between matched pred-target pairs
      3. Select permutation with minimum total loss (PIT)
      4. Add auxiliary loss: for unassigned pred tracks, penalize ||pred|| > 0
         This forces inactive tracks toward zero, eliminating ghost vectors.

    Total loss = mean over classes of (PIT_loss + lambda * auxiliary_suppression)

    Args:
        num_tracks: Number of tracks per class (T).
        aux_lambda: Weight for auxiliary suppression loss.
    """

    def __init__(self, num_tracks: int = 3, aux_lambda: float = 0.1) -> None:
        super().__init__()
        self.num_tracks = num_tracks
        self.aux_lambda = aux_lambda

        # Pre-compute all permutations of track indices
        self._perms = list(itertools.permutations(range(num_tracks)))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ADPIT loss.

        Args:
            pred: Predicted Multi-ACCDOA, shape (B, T, C, T_tracks, 3).
            target: Ground truth Multi-ACCDOA, shape (B, T, C, T_tracks, 3).

        Returns:
            Scalar loss tensor.
        """
        b, t, c, n_tracks, _ = pred.shape
        assert n_tracks == self.num_tracks

        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        for cls_idx in range(c):
            pred_cls = pred[:, :, cls_idx]    # (B, T, T_tracks, 3)
            target_cls = target[:, :, cls_idx]  # (B, T, T_tracks, 3)

            # Determine which target tracks are active
            # Active track: ||target_vector|| > 0
            target_norms = torch.norm(target_cls, dim=-1)  # (B, T, T_tracks)

            # PIT: find best permutation
            best_pit_loss = None

            for perm in self._perms:
                perm_pred = pred_cls[:, :, list(perm)]  # Permuted predictions
                # MSE between matched pairs, only for active targets
                diff = perm_pred - target_cls
                pair_loss = (diff ** 2).sum(dim=-1)  # (B, T, T_tracks)

                # Weight by active tracks
                active_mask = (target_norms > 0).float()
                weighted_loss = (pair_loss * active_mask).sum() / (active_mask.sum() + 1e-8)

                if best_pit_loss is None or weighted_loss < best_pit_loss:
                    best_pit_loss = weighted_loss

            # Auxiliary suppression: penalize inactive predictions
            # For tracks where target is zero, pred should also be zero
            inactive_mask = (target_norms == 0).float()  # (B, T, T_tracks)
            pred_norms = torch.norm(pred_cls, dim=-1)  # (B, T, T_tracks)
            aux_loss = (pred_norms * inactive_mask).sum() / (inactive_mask.sum() + 1e-8)

            total_loss = total_loss + best_pit_loss + self.aux_lambda * aux_loss

        return total_loss / c
