"""CIC and FIR decimation filters for PDM → PCM conversion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CICDecimationFilter:
    """Cascaded Integrator-Comb (CIC) decimation filter.

    Used to convert PDM (1-bit high-frequency) to PCM audio.
    CIC filters are multiplier-free, ideal for edge hardware.

    Args:
        decimation_factor: Decimation ratio (e.g., 64 for 3.072 MHz PDM → 48 kHz PCM).
        num_stages: Number of CIC stages (order). Higher = sharper rolloff but more gain.
    """

    def __init__(self, decimation_factor: int = 64, num_stages: int = 4) -> None:
        self.R = decimation_factor
        self.N = num_stages
        # CIC gain = R^N — need to normalize output
        self.gain = float(self.R**self.N)

    def process(self, pdm_signal: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply CIC decimation to a PDM signal.

        Args:
            pdm_signal: Input PDM signal (1-bit values as float: -1.0 or +1.0).

        Returns:
            Decimated PCM signal.
        """
        signal = pdm_signal.copy()

        # Integrator stages (cumulative sum, N times)
        for _ in range(self.N):
            signal = np.cumsum(signal)

        # Downsample
        signal = signal[:: self.R]

        # Comb stages (difference, N times)
        for _ in range(self.N):
            delayed = np.zeros_like(signal)
            delayed[1:] = signal[:-1]
            signal = signal - delayed

        # Normalize by CIC gain
        signal = signal / self.gain

        return signal.astype(np.float32)


class FIRCompensationFilter:
    """FIR compensation filter to flatten CIC passband droop.

    Applied after CIC decimation to compensate for the sinc^N
    frequency response of the CIC filter.
    """

    def __init__(self, num_taps: int = 31) -> None:
        self.num_taps = num_taps
        self._coefficients = self._design_filter()

    def _design_filter(self) -> NDArray[np.float32]:
        """Design a simple low-pass FIR filter using windowed sinc method."""
        n = self.num_taps
        # Cutoff at 0.45 * Nyquist (leaves room for transition band)
        cutoff = 0.45
        half = n // 2
        h = np.zeros(n, dtype=np.float32)

        for i in range(n):
            if i == half:
                h[i] = 2 * cutoff
            else:
                x = i - half
                h[i] = np.sin(2 * np.pi * cutoff * x) / (np.pi * x)

        # Apply Hamming window
        window = np.hamming(n).astype(np.float32)
        h *= window

        # Normalize
        h /= np.sum(h)

        return h

    def process(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply FIR compensation filter."""
        return np.convolve(signal, self._coefficients, mode="same").astype(np.float32)
