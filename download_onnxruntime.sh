#!/bin/bash
# Download ONNX Runtime with bundled cuDNN

echo "=== Downloading ONNX Runtime GPU (with cuDNN) ==="

cd /home/shivani

# Download ONNX Runtime 1.19.2 (latest) with CUDA support
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz

# Extract
tar -xzf onnxruntime-linux-x64-gpu-1.19.2.tgz

echo ""
echo "✓ Downloaded to: /home/shivani/onnxruntime-linux-x64-gpu-1.19.2"
echo ""
echo "Now rebuild with new ONNX Runtime:"
echo "  cd /home/shivani/work/llm-inference/build"
echo "  cmake .. -DONNXRUNTIME_ROOT_DIR=/home/shivani/onnxruntime-linux-x64-gpu-1.19.2"
echo "  make -j4"
