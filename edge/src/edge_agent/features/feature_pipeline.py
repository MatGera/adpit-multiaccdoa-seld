"""Combined feature extraction pipeline: Log-Mel + SALSA + Intensity Vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from edge_agent.config import EdgeConfig
from edge_agent.features.log_mel import extract_log_mel_foa
from edge_agent.features.salsa import compute_salsa_features
from edge_agent.features.intensity_vectors import extract_intensity_vectors


class FeaturePipeline:
    """Extracts all input features for the SELD model.

    Combines:
    - Log-Mel spectrograms for 4 FOA channels: (4, T, F_mel)
    - Intensity Vectors: (3, T, F)
    - SALSA features (optional): (3, T, F)

    Output tensor shape: (C_feat, T, F) where C_feat = 7 (without SALSA) or 10 (with SALSA).
    """

    def __init__(self, config: EdgeConfig, use_salsa: bool = True) -> None:
        self.sample_rate = config.sample_rate
        self.n_fft = 512
        self.hop_length = 240
        self.n_mels = 128
        self.use_salsa = use_salsa

    def extract(self, foa: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract combined features from FOA audio.

        Args:
            foa: FOA B-format audio, shape (4, num_samples).

        Returns:
            Feature tensor, shape (1, C_feat, T, F) ready for inference.
            C_feat = 7 (log-mel + IV) or 10 (log-mel + IV + SALSA).
        """
        # Log-Mel spectrograms: (4, T, n_mels)
        log_mel = extract_log_mel_foa(
            foa,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Intensity Vectors: (3, T, F)
        iv = extract_intensity_vectors(
            foa,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Match time dimension (T might differ slightly)
        min_t = min(log_mel.shape[1], iv.shape[1])
        log_mel = log_mel[:, :min_t, :]
        iv = iv[:, :min_t, :]

        # Truncate/pad frequency dim for IV to match n_mels
        if iv.shape[2] > self.n_mels:
            iv = iv[:, :, : self.n_mels]
        elif iv.shape[2] < self.n_mels:
            pad_width = self.n_mels - iv.shape[2]
            iv = np.pad(iv, ((0, 0), (0, 0), (0, pad_width)))

        features_list = [log_mel, iv]  # (4, T, F) + (3, T, F) = 7 channels

        if self.use_salsa:
            salsa = compute_salsa_features(
                foa,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                sample_rate=self.sample_rate,
            )
            salsa = salsa[:, :min_t, :]
            if salsa.shape[2] > self.n_mels:
                salsa = salsa[:, :, : self.n_mels]
            elif salsa.shape[2] < self.n_mels:
                pad_width = self.n_mels - salsa.shape[2]
                salsa = np.pad(salsa, ((0, 0), (0, 0), (0, pad_width)))
            features_list.append(salsa)  # +3 channels = 10 total

        combined = np.concatenate(features_list, axis=0)  # (C_feat, T, F)

        # Add batch dimension: (1, C_feat, T, F)
        return combined[np.newaxis, ...].astype(np.float32)
