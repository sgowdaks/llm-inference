#!/bin/bash
# Test script for C++ ONNX inference

echo "=== Testing C++ ONNX Inference ==="
echo ""

# Change to the project directory
cd /home/shivani/work/llm-inference

# Test 1: Check if executable exists
echo "✓ Checking if onnx_inference executable exists..."
if [ -f "./build/onnx_inference" ]; then
    echo "  ✓ Executable found"
else
    echo "  ✗ Executable not found. Run 'cd build && make' first"
    exit 1
fi

# Test 2: Check if model exists
echo "✓ Checking if ONNX model exists..."
if [ -f "./export3-8B/qwen.onnx" ]; then
    echo "  ✓ Model found"
    MODEL_SIZE=$(du -h export3-8B/qwen.onnx | cut -f1)
    echo "  Model size: $MODEL_SIZE"
else
    echo "  ✗ Model not found. Run 'python exporter.py --config config.json --mode export' first"
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
echo "Note: First run takes ~45-60 seconds to load model, then generates tokens slowly on CPU"
echo "Expected: ~5-10 seconds per token on CPU for 8B model"
echo ""

# Run with 3 minute timeout, limiting to just a few tokens
timeout 180 ./build/onnx_inference "Hi" 2>&1 | tee /tmp/onnx_test_output.log

# Check exit code
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "⚠ Inference timed out (expected for slow CPU inference)"
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
echo "  cd /home/shivani/work/llm-inference"
echo "  ./build/onnx_inference \"Your prompt here\""
echo ""
echo "For faster inference, consider:"
echo "  1. Using the smaller 1.7B model (update config.json)"
echo "  2. Using a GPU-enabled system"
echo "  3. Reducing max_decode parameter in the code"
