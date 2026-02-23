"""Export PyTorch SELD model to ONNX format."""

from __future__ import annotations

from pathlib import Path

import torch
import structlog

logger = structlog.get_logger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    opset: int = 17,
    input_shape: tuple[int, ...] = (1, 7, 100, 128),
) -> Path:
    """Export a PyTorch model to ONNX.

    Args:
        model: PyTorch model (must be in eval mode).
        output_path: Path to save the ONNX model.
        opset: ONNX opset version.
        input_shape: Dummy input shape (B, C_feat, T, F).

    Returns:
        Path to the exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "time_frames"},
            "output": {0: "batch_size", 1: "time_frames"},
        },
    )

    logger.info("onnx_exported", path=str(output_path), opset=opset)
    return output_path
