"""Build TensorRT engine from ONNX model (runs on Jetson or GPU machine)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp16",
    workspace_mb: int = 2048,
    calibration_data_dir: str | Path | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX model using trtexec.

    This function calls the trtexec command-line tool, which must be
    available on the system (included in JetPack/TensorRT installations).

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Path to save the TensorRT engine.
        precision: "fp16", "int8", or "fp32".
        workspace_mb: GPU workspace memory in MB.
        calibration_data_dir: Path to calibration data for INT8 (required if precision="int8").

    Returns:
        Path to the built engine.
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
        if calibration_data_dir:
            cmd.append(f"--calib={calibration_data_dir}")
        else:
            raise ValueError("INT8 precision requires calibration_data_dir")
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")

    logger.info(
        "building_tensorrt_engine",
        onnx=str(onnx_path),
        engine=str(engine_path),
        precision=precision,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(
            "trtexec_failed",
            returncode=result.returncode,
            stderr=result.stderr[-500:] if result.stderr else "",
        )
        raise RuntimeError(f"trtexec failed with code {result.returncode}")

    logger.info("tensorrt_engine_built", path=str(engine_path))
    return engine_path
