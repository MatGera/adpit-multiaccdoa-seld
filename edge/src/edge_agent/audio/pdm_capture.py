"""PDM audio capture via ALSA/sounddevice, with decimation to PCM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import structlog

from edge_agent.config import EdgeConfig
from edge_agent.audio.decimation import CICDecimationFilter

logger = structlog.get_logger(__name__)


class AudioCapture:
    """Captures multi-channel audio from MEMS microphone array.

    Handles PDM → PCM conversion via decimation filter.
    Supports mock mode for development without hardware.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.num_channels = config.num_channels
        self.sample_rate = config.sample_rate
        self.frame_samples = config.frame_samples
        self.mock_audio = config.mock_audio

        # CIC decimation filter for PDM → PCM
        self.decimation = CICDecimationFilter(
            decimation_factor=64,  # PDM oversampling ratio
            num_stages=4,
        )

        if not self.mock_audio:
            self._init_audio_device()

    def _init_audio_device(self) -> None:
        """Initialize the audio device via sounddevice/ALSA."""
        try:
            import sounddevice as sd

            self._sd = sd
            # Verify device exists
            device_info = sd.query_devices(self.config.audio_device)
            logger.info(
                "audio_device_initialized",
                device=self.config.audio_device,
                channels=device_info.get("max_input_channels", 0),
                sample_rate=self.sample_rate,
            )
        except Exception:
            logger.exception("audio_device_init_failed")
            raise

    def read_frame(self) -> NDArray[np.float32]:
        """Read one frame of multi-channel audio.

        Returns:
            Audio array, shape (num_channels, frame_samples).
        """
        if self.mock_audio:
            return self._generate_mock_audio()

        # Read from ALSA device
        data = self._sd.rec(
            frames=self.frame_samples,
            samplerate=self.sample_rate,
            channels=self.num_channels,
            dtype="float32",
            blocking=True,
        )
        # sounddevice returns (samples, channels), we want (channels, samples)
        return data.T.copy()

    def _generate_mock_audio(self) -> NDArray[np.float32]:
        """Generate synthetic test audio (sine waves from different directions)."""
        t = np.linspace(0, self.config.frame_length_ms / 1000, self.frame_samples, dtype=np.float32)

        # Generate a 1 kHz tone with slight phase differences to simulate DOA
        freq = 1000.0
        audio = np.zeros((self.num_channels, self.frame_samples), dtype=np.float32)
        for ch in range(self.num_channels):
            phase_offset = ch * np.pi / (2 * self.num_channels)
            audio[ch] = 0.1 * np.sin(2 * np.pi * freq * t + phase_offset)

        # Add noise
        audio += 0.01 * np.random.randn(*audio.shape).astype(np.float32)

        return audio
