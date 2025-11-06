#!/bin/bash
# Test script for C++ ONNX inference

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Testing C++ ONNX Inference ==="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Test 1: Check if executable exists
echo "✓ Checking if onnx_inference executable exists..."
if [ -f "./build/bin/onnx_inference" ]; then
    echo "  ✓ Executable found"
else
    echo "  ✗ Executable not found. Run 'cd build && cmake .. && make' first"
    exit 1
fi

# Test 2: Check if model exists
echo "✓ Checking if ONNX model exists..."
if [ -f "./export3-8B/qwen.onnx" ]; then
    echo "  ✓ Model found"
    MODEL_SIZE=$(du -h export3-8B/qwen.onnx | cut -f1)
    echo "  Model size: $MODEL_SIZE"
else
    echo "  ✗ Model not found. Run 'python src/exporter.py --config configs/config.json --mode export' first"
    exit 1
fi

# Test 3: Check if tokenizer exists
echo "✓ Checking if tokenizer exists..."
if [ -f "./Qwen3-8B/tokenizer.json" ]; then
    echo "  ✓ Tokenizer found"
else
    echo "  ✗ Tokenizer not found"
    exit 1
fi

# Test 4: Run inference with short prompt and limited tokens
echo ""
echo "=== Running Inference Test ==="
echo "Prompt: 'Hi'"
echo "Note: First run takes ~45-60 seconds to load model"
echo ""

# Run with 3 minute timeout
timeout 180 ./build/bin/onnx_inference "Hi" 2>&1 | tee /tmp/onnx_test_output.log

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⚠ Inference timed out"
    echo "  Check /tmp/onnx_test_output.log for partial output"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference completed successfully!"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
    exit 1
fi

echo ""
echo "=== Test Results ==="
echo "Output saved to: /tmp/onnx_test_output.log"
echo ""
echo "To test manually:"
echo "  cd $PROJECT_ROOT"
echo "  ./scripts/run_gpu_inference.sh \"Your prompt here\""

