#!/bin/bash
# Run ONNX inference with GPU support
# This script sets up the environment for CUDA and cuDNN

set -e

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add conda cuDNN to library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Check if we need to use conda or system libstdc++
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    # Try system libstdc++ first (usually more up to date)
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
else
    # Fall back to conda
    export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
fi

# Run inference - now outputs to build/bin
cd "$PROJECT_ROOT"
./build/bin/onnx_inference "$@"


