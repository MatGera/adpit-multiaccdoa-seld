"""Acoustic intensity vector extraction from FOA signal."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seld_common.audio_utils import compute_intensity_vectors


def extract_intensity_vectors(
    foa: NDArray[np.float32],
    n_fft: int = 512,
    hop_length: int = 240,
) -> NDArray[np.float32]:
    """Extract acoustic intensity vectors from FOA signal.

    The intensity vector at each time-frequency bin encodes the
    direction of energy flow: I = Re(W* . [X, Y, Z]).

    Args:
        foa: FOA B-format audio, shape (4, num_samples).
        n_fft: FFT window size.
        hop_length: Hop length.

    Returns:
        Intensity vectors, shape (3, T, F) for [Ix, Iy, Iz].
    """
    return compute_intensity_vectors(foa, n_fft=n_fft, hop_length=hop_length)
