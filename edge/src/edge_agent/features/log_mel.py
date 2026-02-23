"""Log-Mel spectrogram extraction for FOA channels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seld_common.audio_utils import compute_log_mel_spectrogram


def extract_log_mel_foa(
    foa: NDArray[np.float32],
    sample_rate: int = 24000,
    n_fft: int = 512,
    hop_length: int = 240,
    n_mels: int = 128,
) -> NDArray[np.float32]:
    """Extract log-mel spectrograms for all 4 FOA channels.

    Args:
        foa: FOA B-format audio, shape (4, num_samples).
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel filter banks.

    Returns:
        Log-mel spectrograms, shape (4, T, n_mels).
    """
    mels = []
    for ch in range(4):
        mel = compute_log_mel_spectrogram(
            foa[ch],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mels.append(mel)

    return np.stack(mels, axis=0)  # (4, T, n_mels)
