#!/usr/bin/env bash
set -euo pipefail

# Build TensorRT engine from ONNX model for Jetson Orin
# This script should be run ON the Jetson device or inside a JetPack container

ONNX_MODEL="${1:-models/seld_model.onnx}"
OUTPUT_ENGINE="${2:-models/seld_fp16.engine}"
PRECISION="${3:-fp16}"
WORKSPACE_MB="${4:-2048}"

echo "=== Building TensorRT Engine ==="
echo "  ONNX Model:  $ONNX_MODEL"
echo "  Output:      $OUTPUT_ENGINE"
echo "  Precision:   $PRECISION"
echo "  Workspace:   ${WORKSPACE_MB}MB"

if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found: $ONNX_MODEL"
    exit 1
fi

TRTEXEC_ARGS=(
    "--onnx=$ONNX_MODEL"
    "--saveEngine=$OUTPUT_ENGINE"
    "--workspace=$WORKSPACE_MB"
)

case "$PRECISION" in
    fp16)
        TRTEXEC_ARGS+=("--fp16")
        ;;
    int8)
        CALIB_DATA="${5:-data/calibration}"
        if [ ! -d "$CALIB_DATA" ]; then
            echo "ERROR: INT8 calibration data directory not found: $CALIB_DATA"
            exit 1
        fi
        TRTEXEC_ARGS+=("--int8" "--calib=$CALIB_DATA")
        ;;
    fp32)
        # No extra flags needed
        ;;
    *)
        echo "ERROR: Unknown precision: $PRECISION (use fp16, int8, or fp32)"
        exit 1
        ;;
esac

# Add timing info
TRTEXEC_ARGS+=("--verbose")

echo "Running: trtexec ${TRTEXEC_ARGS[*]}"
trtexec "${TRTEXEC_ARGS[@]}"

echo ""
echo "=== TensorRT engine built successfully ==="
echo "  Engine: $OUTPUT_ENGINE"
ls -lh "$OUTPUT_ENGINE"
