"""TensorRT engine loader and inference for SELD model on Jetson Orin."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import structlog

from edge_agent.config import EdgeConfig

logger = structlog.get_logger(__name__)


class InferenceEngine:
    """Loads and runs a TensorRT engine for SELD inference.

    On Jetson Orin (production): uses TensorRT Python API.
    On x86 (development): falls back to PyTorch CPU model.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.engine_path = Path(config.tensorrt_engine_path)
        self._trt_engine = None
        self._trt_context = None
        self._pytorch_model = None
        self._use_tensorrt = False

    def load(self) -> None:
        """Load the inference engine (TensorRT or PyTorch fallback)."""
        if self.engine_path.exists() and self.engine_path.suffix == ".engine":
            self._load_tensorrt()
        else:
            self._load_pytorch_fallback()

    def _load_tensorrt(self) -> None:
        """Load TensorRT engine from serialized .engine file."""
        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(self.engine_path, "rb") as f:
                engine_data = f.read()

            self._trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self._trt_context = self._trt_engine.create_execution_context()
            self._use_tensorrt = True

            logger.info("tensorrt_engine_loaded", path=str(self.engine_path))

        except ImportError:
            logger.warning("tensorrt_not_available, falling back to PyTorch")
            self._load_pytorch_fallback()

    def _load_pytorch_fallback(self) -> None:
        """Load PyTorch model for development/debugging."""
        try:
            import torch

            # Create a dummy model that outputs the right shape
            # In production, this would load a real checkpoint
            self._pytorch_model = _create_dummy_model(
                self.config.num_classes, self.config.num_tracks
            )
            self._use_tensorrt = False
            logger.info("pytorch_fallback_loaded")

        except ImportError:
            logger.error("neither_tensorrt_nor_pytorch_available")
            raise RuntimeError("No inference backend available")

    def infer(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run inference on extracted features.

        Args:
            features: Input tensor, shape (1, C_feat, T, F).

        Returns:
            Multi-ACCDOA output, shape (C_classes, T_tracks, 3).
        """
        if self._use_tensorrt:
            return self._infer_tensorrt(features)
        else:
            return self._infer_pytorch(features)

    def _infer_tensorrt(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run TensorRT inference."""
        import tensorrt as trt

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise RuntimeError("pycuda required for TensorRT inference")

        # Allocate device memory
        d_input = cuda.mem_alloc(features.nbytes)
        cuda.memcpy_htod(d_input, features)

        # Get output shape from engine
        output_shape = self._trt_engine.get_tensor_shape(
            self._trt_engine.get_tensor_name(1)
        )
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Execute
        self._trt_context.set_tensor_address(
            self._trt_engine.get_tensor_name(0), int(d_input)
        )
        self._trt_context.set_tensor_address(
            self._trt_engine.get_tensor_name(1), int(d_output)
        )
        self._trt_context.execute_async_v3(stream_handle=0)

        # Copy output back
        cuda.memcpy_dtoh(output, d_output)

        # Free device memory
        d_input.free()
        d_output.free()

        # Reshape: (B, T', C, T_tracks, 3) → take first batch, average over time
        # → (C, T_tracks, 3)
        if output.ndim == 5:
            output = output[0].mean(axis=0)  # Average over time frames
        elif output.ndim == 4:
            output = output[0].mean(axis=0)

        return output

    def _infer_pytorch(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run PyTorch inference (development fallback)."""
        import torch

        with torch.no_grad():
            x = torch.from_numpy(features)
            y = self._pytorch_model(x)
            output = y.numpy()

        # Shape: (B, T', C, T_tracks, 3) → average over time → (C, T_tracks, 3)
        if output.ndim == 5:
            output = output[0].mean(axis=0)
        elif output.ndim == 4:
            output = output[0].mean(axis=0)

        return output

    def unload(self) -> None:
        """Release engine resources."""
        self._trt_engine = None
        self._trt_context = None
        self._pytorch_model = None
        logger.info("inference_engine_unloaded")


def _create_dummy_model(num_classes: int, num_tracks: int):
    """Create a dummy PyTorch model for development."""
    import torch
    import torch.nn as nn

    class DummySELD(nn.Module):
        def __init__(self, c: int, t: int) -> None:
            super().__init__()
            self.c = c
            self.t = t
            self.fc = nn.Linear(128, c * t * 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            # Simple global average pooling
            x = x.mean(dim=(2, 3))  # (B, C_feat) → (B, C_feat)
            x = torch.zeros(b, 128)  # dummy
            x = self.fc(x)
            return x.reshape(b, 1, self.c, self.t, 3)

    return DummySELD(num_classes, num_tracks).eval()
