"""First-Order Ambisonics (FOA) encoding from raw microphone array channels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seld_common.audio_utils import encode_foa, TETRA_ENCODING_MATRIX


class FOAEncoder:
    """Encodes raw multi-channel microphone input to FOA B-format (W, X, Y, Z).

    Supports:
    - 4-channel tetrahedral arrays (standard encoding matrix)
    - 8-channel spherical arrays (custom encoding matrix required)
    """

    def __init__(self, num_channels: int, encoding_matrix: NDArray[np.float32] | None = None) -> None:
        self.num_channels = num_channels

        if encoding_matrix is not None:
            self.encoding_matrix = encoding_matrix
        elif num_channels == 4:
            self.encoding_matrix = TETRA_ENCODING_MATRIX
        else:
            raise ValueError(
                f"No default encoding matrix for {num_channels} channels. "
                "Provide a custom encoding_matrix."
            )

    def encode(self, raw_audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Encode raw multi-channel audio to FOA B-format.

        Args:
            raw_audio: Raw audio, shape (num_channels, num_samples).

        Returns:
            FOA B-format, shape (4, num_samples) for [W, X, Y, Z].
        """
        assert raw_audio.shape[0] == self.num_channels, (
            f"Expected {self.num_channels} channels, got {raw_audio.shape[0]}"
        )
        return encode_foa(raw_audio, self.encoding_matrix)
