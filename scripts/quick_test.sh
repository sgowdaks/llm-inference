#!/bin/bash
# Quick validation test - checks if model loads and inference starts

set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Quick Validation Test ==="
echo "Testing if the model loads and inference starts..."
echo ""

# Run inference but kill it after model loads (look for "Qwen Answering:" in output)
timeout 90 ./build/bin/onnx_inference "Hi" 2>&1 | tee /tmp/quick_test.log &
PID=$!

echo "Waiting for model to load (usually 45-60 seconds)..."
sleep 60

# Check if process is still running and if we see expected output
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Process is running"
    
    if grep -q "Using.*execution provider" /tmp/quick_test.log; then
        echo "✓ ONNX Runtime initialized"
    fi
    
    if grep -q "Qwen Answering:" /tmp/quick_test.log; then
        echo "✓ Model loaded successfully"
        echo "✓ Inference started"
        echo ""
        echo "SUCCESS! The C++ inference is working."
        echo "Killing test process..."
        kill $PID 2>/dev/null
        wait $PID 2>/dev/null
    else
        echo "⚠ Model still loading, waiting a bit more..."
        sleep 30
        if grep -q "Qwen Answering:" /tmp/quick_test.log; then
            echo "✓ Model loaded successfully"
            echo "✓ SUCCESS!"
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
        else
            echo "✗ Model loading taking longer than expected"
            echo "Check /tmp/quick_test.log for details"
            kill $PID 2>/dev/null
            wait $PID 2>/dev/null
        fi
    fi
else
    echo "✗ Process exited unexpectedly"
    echo "Check /tmp/quick_test.log for errors"
    cat /tmp/quick_test.log
    exit 1
fi

echo ""
echo "Full output in: /tmp/quick_test.log"

