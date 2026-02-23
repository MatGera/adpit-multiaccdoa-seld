"""Multi-ACCDOA post-processing: decode output tensor to prediction list."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Prediction:
    """A single SELD prediction."""

    class_name: str
    confidence: float  # norm of ACCDOA vector, [0, 1]
    vector: tuple[float, float, float]  # (x, y, z) unit DOA


def decode_multi_accdoa(
    output: NDArray[np.float32],
    class_names: list[str],
    threshold: float = 0.5,
) -> list[Prediction]:
    """Decode Multi-ACCDOA tensor to prediction list.

    Multi-ACCDOA output format: (C_classes, T_tracks, 3)
    For each class c, for each track t:
        vector = output[c, t, :]
        norm = ||vector||
        if norm > threshold:
            direction = vector / norm  (unit DOA vector)
            confidence = norm
            emit Prediction(class=c, confidence=norm, direction=direction)

    The vector norm encodes detection confidence: ||v|| in [0, 1].
    This solves the polyphony collapse (ghost vector problem) of single-ACCDOA
    by maintaining T=3 independent tracks per class.

    Args:
        output: Multi-ACCDOA tensor, shape (C, T, 3).
        class_names: List of class name strings, length C.
        threshold: Minimum vector norm to emit a prediction.

    Returns:
        List of Prediction objects for active detections.
    """
    assert output.ndim == 3, f"Expected 3D tensor (C, T, 3), got shape {output.shape}"
    num_classes, num_tracks, _ = output.shape
    assert len(class_names) == num_classes, (
        f"class_names length {len(class_names)} != output classes {num_classes}"
    )

    predictions: list[Prediction] = []

    for c in range(num_classes):
        for t in range(num_tracks):
            vec = output[c, t, :]  # (3,)
            norm = float(np.linalg.norm(vec))

            if norm > threshold:
                # Normalize to unit vector for direction
                direction = vec / norm
                predictions.append(
                    Prediction(
                        class_name=class_names[c],
                        confidence=round(min(norm, 1.0), 4),
                        vector=(
                            round(float(direction[0]), 4),
                            round(float(direction[1]), 4),
                            round(float(direction[2]), 4),
                        ),
                    )
                )

    return predictions
