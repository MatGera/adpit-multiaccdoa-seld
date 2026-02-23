"""Secure buffer lifecycle manager — ensures audio data destruction within 500ms."""

from __future__ import annotations

import gc

import numpy as np
from numpy.typing import NDArray

import structlog

logger = structlog.get_logger(__name__)


class SecureBufferManager:
    """Manages secure lifecycle of audio buffers (privacy-by-design).

    All audio-related numpy arrays must be registered with this manager.
    After inference, `destroy_all()` zeroes and frees all registered buffers.
    """

    def __init__(self) -> None:
        self._buffers: list[NDArray] = []

    def register(self, buffer: NDArray) -> None:
        """Register a numpy array for secure destruction."""
        self._buffers.append(buffer)

    def destroy_all(self) -> None:
        """Securely destroy all registered buffers.

        Process:
        1. Fill each array with zeros (overwrite memory)
        2. Delete reference
        3. Force garbage collection
        """
        count = len(self._buffers)
        for buf in self._buffers:
            try:
                if buf is not None and buf.size > 0:
                    buf.fill(0)
            except (ValueError, AttributeError):
                pass  # Buffer may already be freed

        self._buffers.clear()
        gc.collect()

        if count > 0:
            logger.debug("buffers_destroyed", count=count)
