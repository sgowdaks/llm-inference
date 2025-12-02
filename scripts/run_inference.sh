#!/bin/bash
# Run ONNX inference with CPU or GPU support
# Usage: ./run_inference.sh --device [cpu|gpu] [additional args...]

set -e

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse device argument (default: gpu)
DEVICE="gpu"
BINARY_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            BINARY_ARGS+=("--device" "$2")
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            BINARY_ARGS+=("$1")
            shift
            ;;
        *)
            BINARY_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set up environment based on device
if [[ "$DEVICE" == "gpu" ]]; then
    echo "Setting up GPU environment..."
    
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
else
    echo "Setting up CPU environment..."
    # CPU mode doesn't need special CUDA/cuDNN setup
fi

# Run inference
cd "$PROJECT_ROOT"
./build/bin/onnx_inference "${BINARY_ARGS[@]}"
