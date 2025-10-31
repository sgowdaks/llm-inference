#!/bin/bash
# Run ONNX inference with GPU support

# Add conda cuDNN to library path, but use system libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Override libstdc++ to use system version to avoid GLIBCXX issues
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Run inference
cd /home/shivani/work/llm-inference
./build/onnx_inference "$@"
