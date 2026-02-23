"""FOA encoding and feature extraction utilities shared across edge and training."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Tetrahedral microphone array encoding matrix (4 capsules -> FOA W,X,Y,Z)
# Based on standard Soundfield/Ambisonic B-format encoding
TETRA_ENCODING_MATRIX = np.array(
    [
        [0.5, 0.5, 0.5, 0.5],  # W (omnidirectional)
        [0.5, 0.5, -0.5, -0.5],  # X (front-back)
        [0.5, -0.5, 0.5, -0.5],  # Y (left-right)
        [0.5, -0.5, -0.5, 0.5],  # Z (up-down)
    ],
    dtype=np.float32,
)


def encode_foa(
    raw_channels: NDArray[np.float32],
    encoding_matrix: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Encode raw multi-channel audio to First-Order Ambisonics (FOA) B-format.

    Args:
        raw_channels: Raw audio array, shape (num_channels, num_samples).
        encoding_matrix: Custom encoding matrix. Uses tetrahedral default if None.

    Returns:
        FOA B-format array, shape (4, num_samples) for [W, X, Y, Z].
    """
    if encoding_matrix is None:
        encoding_matrix = TETRA_ENCODING_MATRIX

    num_channels = raw_channels.shape[0]
    assert encoding_matrix.shape[1] == num_channels, (
        f"Encoding matrix expects {encoding_matrix.shape[1]} channels, "
        f"got {num_channels}"
    )

    return encoding_matrix @ raw_channels


def compute_intensity_vectors(
    foa: NDArray[np.float32],
    n_fft: int = 512,
    hop_length: int = 240,
) -> NDArray[np.float32]:
    """Compute acoustic intensity vectors from FOA signal.

    The intensity vector at each time-frequency bin is:
      I = Re(W* . [X, Y, Z])

    Args:
        foa: FOA B-format signal, shape (4, num_samples).
        n_fft: FFT size.
        hop_length: Hop length for STFT.

    Returns:
        Intensity vectors, shape (3, T, F) for [Ix, Iy, Iz].
    """
    # STFT of each FOA channel
    stft_list = []
    for ch in range(4):
        # Use numpy rfft for simplicity (real FFT)
        num_frames = 1 + (foa.shape[1] - n_fft) // hop_length
        frames = np.lib.stride_tricks.sliding_window_view(foa[ch], n_fft)[::hop_length]
        if len(frames) > num_frames:
            frames = frames[:num_frames]
        window = np.hanning(n_fft).astype(np.float32)
        stft = np.fft.rfft(frames * window, n=n_fft, axis=-1)
        stft_list.append(stft)

    w_stft = stft_list[0]  # (T, F)
    x_stft = stft_list[1]
    y_stft = stft_list[2]
    z_stft = stft_list[3]

    # Intensity: I = Re(conj(W) * [X, Y, Z])
    w_conj = np.conj(w_stft)
    ix = np.real(w_conj * x_stft)  # (T, F)
    iy = np.real(w_conj * y_stft)
    iz = np.real(w_conj * z_stft)

    return np.stack([ix, iy, iz], axis=0)  # (3, T, F)


def compute_log_mel_spectrogram(
    audio: NDArray[np.float32],
    sample_rate: int = 24000,
    n_fft: int = 512,
    hop_length: int = 240,
    n_mels: int = 128,
) -> NDArray[np.float32]:
    """Compute log-mel spectrogram for a single channel.

    Args:
        audio: Single-channel audio, shape (num_samples,).
        sample_rate: Audio sample rate.
        n_fft: FFT size.
        hop_length: Hop length.
        n_mels: Number of mel filter banks.

    Returns:
        Log-mel spectrogram, shape (T, n_mels).
    """
    # Compute STFT magnitude
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length]
    if len(frames) > num_frames:
        frames = frames[:num_frames]
    window = np.hanning(n_fft).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=-1))

    # Create mel filterbank
    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels)
    mel_spec = magnitude @ mel_fb.T  # (T, n_mels)

    # Log scale
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    return log_mel


def _mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int
) -> NDArray[np.float32]:
    """Create a mel-scale filterbank matrix."""
    fmin = 0.0
    fmax = sample_rate / 2.0

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        start = bin_points[i]
        center = bin_points[i + 1]
        end = bin_points[i + 2]

        # Rising slope
        for j in range(start, center):
            if center > start:
                filterbank[i, j] = (j - start) / (center - start)
        # Falling slope
        for j in range(center, end):
            if end > center:
                filterbank[i, j] = (end - j) / (end - center)

    return filterbank
