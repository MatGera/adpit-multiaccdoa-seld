"""SELD evaluation metrics — DCASE official metrics implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_seld_metrics(
    pred: NDArray[np.float32],
    target: NDArray[np.float32],
    threshold: float = 0.5,
    angular_threshold_deg: float = 20.0,
) -> dict[str, float]:
    """Compute SELD evaluation metrics.

    Metrics (as used in DCASE Challenge Task 3):
    - F1: Detection F1-score (location-dependent)
    - LE: Localization Error in degrees (angular error)
    - LR: Localization Recall
    - SELD-error: Combined metric (lower is better)
        SELD_error = (1-F1 + LE/180 + 1-LR) / 3

    Args:
        pred: Predictions, shape (T, C, T_tracks, 3).
        target: Ground truth, shape (T, C, T_tracks, 3).
        threshold: Detection threshold on vector norm.
        angular_threshold_deg: Angular threshold for location-dependent detection.

    Returns:
        Dict with keys: f1, le_deg, lr, seld_error.
    """
    tp = 0
    fp = 0
    fn = 0
    total_angular_error = 0.0
    num_localized = 0

    t, c, n_tracks, _ = pred.shape

    for t_idx in range(t):
        for c_idx in range(c):
            # Get active predictions and targets
            pred_active = []
            for track in range(n_tracks):
                vec = pred[t_idx, c_idx, track]
                norm = np.linalg.norm(vec)
                if norm > threshold:
                    direction = vec / norm
                    pred_active.append(direction)

            target_active = []
            for track in range(n_tracks):
                vec = target[t_idx, c_idx, track]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    direction = vec / norm
                    target_active.append(direction)

            # Match predictions to targets (greedy, by angular distance)
            matched_pred = set()
            matched_target = set()

            for t_i, t_dir in enumerate(target_active):
                best_p_i = -1
                best_angle = float("inf")

                for p_i, p_dir in enumerate(pred_active):
                    if p_i in matched_pred:
                        continue
                    angle = _angular_distance_deg(p_dir, t_dir)
                    if angle < best_angle:
                        best_angle = angle
                        best_p_i = p_i

                if best_p_i >= 0 and best_angle < angular_threshold_deg:
                    tp += 1
                    matched_pred.add(best_p_i)
                    matched_target.add(t_i)
                    total_angular_error += best_angle
                    num_localized += 1
                else:
                    fn += 1

            fp += len(pred_active) - len(matched_pred)

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    le_deg = total_angular_error / num_localized if num_localized > 0 else 180.0
    lr = num_localized / (tp + fn) if (tp + fn) > 0 else 0.0

    seld_error = (1 - f1 + le_deg / 180.0 + 1 - lr) / 3.0

    return {
        "f1": round(f1, 4),
        "le_deg": round(le_deg, 2),
        "lr": round(lr, 4),
        "seld_error": round(seld_error, 4),
    }


def _angular_distance_deg(v1: NDArray, v2: NDArray) -> float:
    """Compute angular distance between two unit vectors in degrees."""
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
