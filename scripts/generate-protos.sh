#!/usr/bin/env bash
set -euo pipefail

PROTO_DIR="packages/proto/src"
PYTHON_OUT="packages/proto/generated/python"
TS_OUT="packages/proto/generated/typescript"

echo "=== Generating protobuf stubs ==="

mkdir -p "$PYTHON_OUT" "$TS_OUT"

# Generate Python stubs
echo "--- Python gRPC stubs ---"
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$PYTHON_OUT" \
    --grpc_python_out="$PYTHON_OUT" \
    --pyi_out="$PYTHON_OUT" \
    "$PROTO_DIR"/*.proto

# Fix imports in generated Python files (relative imports)
for f in "$PYTHON_OUT"/*_pb2_grpc.py; do
    if [ -f "$f" ]; then
        sed -i 's/^import \(.*\)_pb2/from . import \1_pb2/' "$f"
    fi
done

# Generate TypeScript stubs (requires ts-proto or grpc-tools)
echo "--- TypeScript gRPC stubs ---"
if command -v protoc-gen-ts_proto >/dev/null 2>&1; then
    protoc \
        -I"$PROTO_DIR" \
        --ts_proto_out="$TS_OUT" \
        --ts_proto_opt=esModuleInterop=true \
        --ts_proto_opt=outputServices=grpc-js \
        "$PROTO_DIR"/*.proto
else
    echo "  SKIP: protoc-gen-ts_proto not found. Install: pnpm add -g ts-proto"
fi

echo "=== Proto generation complete ==="
