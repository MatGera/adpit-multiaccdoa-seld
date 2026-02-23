"""SALSA (Spatial Cue-Augmented Log-Spectrogram) feature computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_salsa_features(
    foa: NDArray[np.float32],
    n_fft: int = 512,
    hop_length: int = 240,
    sample_rate: int = 24000,
) -> NDArray[np.float32]:
    """Compute SALSA features from FOA signal.

    SALSA features are normalized inter-channel phase differences (IPDs)
    computed between the omnidirectional channel W and directional channels X, Y, Z.

    SALSA preserves more frequency bins and spatial cues than standard log-mel,
    which is critical for polyphonic SELD.

    Args:
        foa: FOA B-format audio, shape (4, num_samples).
        n_fft: FFT window size.
        hop_length: Hop length.
        sample_rate: Audio sample rate.

    Returns:
        SALSA features, shape (3, T, F) for normalized IPD between W and [X, Y, Z].
    """
    # Compute STFT for each FOA channel
    num_frames = 1 + (foa.shape[1] - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float32)

    stft_channels = []
    for ch in range(4):
        frames = np.lib.stride_tricks.sliding_window_view(foa[ch], n_fft)[::hop_length]
        if len(frames) > num_frames:
            frames = frames[:num_frames]
        stft = np.fft.rfft(frames * window, n=n_fft, axis=-1)
        stft_channels.append(stft)

    w_stft = stft_channels[0]  # Reference channel (omnidirectional)

    # Compute normalized inter-channel phase differences
    # SALSA_c = angle(X_c * conj(W)) normalized by frequency
    salsa_features = []
    freq_bins = np.arange(1, w_stft.shape[-1])  # Skip DC
    # Normalization factor: 2*pi*f*d/c (speed of sound)
    freq_hz = freq_bins * sample_rate / n_fft

    for ch in range(1, 4):  # X, Y, Z channels
        cross_spec = stft_channels[ch][:, 1:] * np.conj(w_stft[:, 1:])
        ipd = np.angle(cross_spec)

        # Normalize by frequency to get consistent spatial cues
        # across frequency bands
        norm_factor = 2 * np.pi * freq_hz / 343.0  # speed of sound = 343 m/s
        norm_factor = np.clip(norm_factor, 1e-6, None)
        normalized_ipd = ipd / norm_factor[np.newaxis, :]

        # Pad DC bin back
        padded = np.zeros((num_frames, w_stft.shape[-1]), dtype=np.float32)
        padded[:, 1:] = normalized_ipd

        salsa_features.append(padded)

    return np.stack(salsa_features, axis=0)  # (3, T, F)
