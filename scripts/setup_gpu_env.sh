#!/bin/bash
# Source this file to enable GPU inference: source setup_gpu_env.sh

# Add CUDA libraries
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add cuDNN from conda (only the CUDA libraries, not libstdc++)
if [ -n "$CONDA_PREFIX" ]; then
    # Create temporary directory for selective library linking
    if [ -z "$CUDNN_LIB_DIR" ]; then
        export CUDNN_LIB_DIR=$(mktemp -d /tmp/cudnn_libs_XXXXX)
        
        # Symlink only CUDA-related libraries to avoid conda's old libstdc++
        for lib in libcudnn libcublasLt libcublas libcudart libnvrtc libcufft libcurand libcusparse libcusolver; do
            ln -sf $CONDA_PREFIX/lib/${lib}*.so* $CUDNN_LIB_DIR/ 2>/dev/null
        done
        
        export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:$LD_LIBRARY_PATH
        echo "✅ GPU environment configured (cuDNN from: $CONDA_PREFIX/lib)"
        echo "🗑️  Temp lib directory: $CUDNN_LIB_DIR (will be cleaned on exit)"
        
        # Register cleanup on shell exit
        trap "rm -rf $CUDNN_LIB_DIR" EXIT
    fi
else
    echo "⚠️  CONDA_PREFIX not set - cuDNN may not be found"
fi

echo "🚀 Run inference with: ./build/bin/onnx_inference_ultra <prompt>"
