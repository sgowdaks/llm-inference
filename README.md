# LLM Inference - Qwen Model ONNX Export & Inference

A high-performance C++ and Python implementation for exporting Qwen language models to ONNX format and running optimized inference with ONNX Runtime.

## Overview

This project provides tools to:
- Export Qwen models (Qwen3-1.7B, Qwen3-8B) from Hugging Face format to optimized ONNX format
- Run inference using ONNX Runtime in both Python and C++
- Achieve fast token generation with CPU/CUDA acceleration
- Test model outputs with structured evaluation datasets

## Features

- **ONNX Export**: Convert Qwen models to ONNX with optimized graph transformations
- **Dual Implementation**: Both Python and C++ inference engines
- **Performance Optimized**: Configured ONNX Runtime sessions with multiple optimization flags
- **KV-Cache Support**: Efficient autoregressive generation with past key-value caching
- **Tokenizer Integration**: Uses HuggingFace tokenizers (Python) and tokenizers-cpp (C++)
- **Quantization Ready**: Embedding layer quantization to uint8 for reduced memory footprint
- **Flexible Configuration**: JSON-based configuration for paths and model parameters

## Repository Structure

```
.
├── CMakeLists.txt           # Build configuration for C++ inference
├── config.json              # Runtime configuration (paths, model settings)
├── exporter.py              # Model export script (PyTorch → ONNX)
├── onnx_inference.py        # Python inference engine
├── onnx_inference.cpp       # C++ inference engine
├── launch.json              # VS Code debug configuration
├── todo.md                  # Development roadmap
├── build/                   # CMake build artifacts
├── export/                  # ONNX model weights and constants (1.7B)
├── export3-8B/              # ONNX model weights and constants (8B)
├── Qwen3-1.7B/              # Qwen 1.7B model files
├── Qwen3-8B/                # Qwen 8B model files
├── test/                    # Test datasets and evaluation scripts
├── tokenizers/              # Tokenizers C++ bindings (submodule)
└── README.md                # This file
```

## Requirements

### Python Dependencies
```bash
pip install torch transformers onnxruntime numpy
```

Optional for CUDA:
```bash
pip install onnxruntime-gpu
```

### C++ Dependencies
- CMake 3.16+
- C++17 compatible compiler (gcc, clang, MSVC)
- ONNX Runtime 1.19.0+ (GPU build recommended)
- CUDA Toolkit 11.x or 12.x (for GPU inference)
- cuDNN 9.x (for GPU inference)
- nlohmann/json library
- tokenizers-cpp (included as submodule)

## Quick Start

### 1. Configuration

Edit `config.json` to set your paths:

```json
{
    "paths": {
        "model_path": "/path/to/Qwen3-8B",
        "model_config": "/path/to/Qwen3-8B/config.json",
        "onnx_file": "/path/to/export/qwen.onnx",
        "test_file": "/path/to/test/test1.json"
    }
}
```

### 2. Export Model to ONNX

```bash
python exporter.py --config config.json --mode export
```

This will:
- Load the Qwen model from Hugging Face format
- Wrap it in an optimized PyTorch module
- Export to ONNX with dynamic axes for efficient batching
- Save model weights and constants to the export directory

### 3. Run Python Inference

```bash
# Single prompt
python onnx_inference.py --config config.json --prompt "What is the capital of France?"

# Test mode with JSON test file
python onnx_inference.py --config config.json --test-mode

# Short answer mode (stops at first sentence)
python onnx_inference.py --config config.json --short-answer --prompt "2 + 2 = ?"
```

### 4. Build and Run C++ Inference

#### Setup ONNX Runtime GPU

Download ONNX Runtime GPU build:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.19.0.tgz
```

Install cuDNN 9 (required for GPU inference):
```bash
# Using conda (recommended)
conda install -c conda-forge cudnn=9

# Or download from NVIDIA and set LD_LIBRARY_PATH manually
```

#### Build the C++ executable

```bash
# Configure build
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-gpu-1.19.0

# Build
make -j4

# Return to project root
cd ..
```

#### Run GPU Inference

Use the provided wrapper script to set up the environment:
```bash
# Create run_gpu_inference.sh if it doesn't exist
cat > run_gpu_inference.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
./build/onnx_inference "$@"
EOF
chmod +x run_gpu_inference.sh

# Run inference
./run_gpu_inference.sh "What is 2+2?"
```

Or for CPU-only inference (slower):
```bash
./build/onnx_inference "What is 2+2?"
```

## Architecture

### Export Pipeline (`exporter.py`)

The exporter wraps the original Hugging Face model with a custom `QWENWrapper` that:
1. **Quantizes embeddings** to uint8 (scale/zero-point per token)
2. **Precomputes rotary embeddings** (cos/sin tables)
3. **Integrates attention mask** generation
4. **Implements custom attention** with KV-cache management
5. **Applies RMSNorm with variance epsilon** for stability

Key optimizations:
- Q/K normalization scaling integrated into weights
- Efficient KV-cache concatenation and reuse
- Dynamic axes for variable sequence lengths

### Python Inference (`onnx_inference.py`)

Features:
- Clean, maintainable rewrite with type hints and logging
- HuggingFace tokenizer integration
- OrtValue-based inference for zero-copy tensor management
- Configurable ONNX Runtime session options
- Test mode for batch evaluation

Performance settings:
- Sequential execution mode
- All graph optimizations enabled
- QDQ cleanup for quantized models
- GELU approximation
- CPU memory arena enabled

### C++ Inference (`onnx_inference.cpp`)

Features:
- Native ONNX Runtime C++ API (v1.19.0+)
- GPU acceleration with CUDA execution provider
- tokenizers-cpp for fast tokenization
- Streaming token generation with KV-cache management
- Automatic CUDA detection and fallback to CPU
- Proper tensor lifecycle management for multi-iteration decode
- Naive UTF-8 decoding with character replacement

Optimizations:
- Zero-copy tensor creation where possible
- Efficient KV-cache reuse via std::move
- Persistent data buffers across decode iterations
- Minimal heap allocations in decode loop
- Device selection for multi-GPU systems

## Configuration Options

### ONNX Runtime Session Options

Both Python and C++ implementations configure:
- `log_severity_level`: 4 (minimal logging)
- `inter/intra_op_num_threads`: 0 (auto-detect)
- `execution_mode`: Sequential
- `graph_optimization_level`: All optimizations enabled
- `enable_cpu_mem_arena`: True
- `set_denormal_as_zero`: True
- `allow_spinning`: True for inter/intra ops
- `qdq_matmulnbits_accuracy_level`: 4
- `enable_gelu_approximation`: True

### Model Configuration (from `config.json`)

Required fields:
- `num_key_value_heads`: Number of KV heads (e.g., 4 for 8B model)
- `head_dim`: Dimension per attention head (e.g., 128)
- `num_hidden_layers`: Number of transformer layers (e.g., 32)

## Testing

Create a test file in JSON format (`test/test1.json`):

```json
[
    {
        "prompt": "What is 2+2?",
        "expected": "4"
    },
    {
        "prompt": "What is the capital of France?",
        "expected": "Paris"
    }
]
```

Run tests:
```bash
# Python
python onnx_inference.py --config config.json --test-mode

# HuggingFace baseline comparison
python exporter.py --config config.json --mode test
```

## Performance

Typical performance on example hardware:

**Qwen3-8B on GPU (NVIDIA RTX A6000, CUDA):**
- First token: ~200ms (model load time separate)
- Subsequent tokens: ~12 tokens/sec
- Using ONNX Runtime 1.19.0 GPU + cuDNN 9

**Qwen3-8B on GPU (NVIDIA RTX 3090):**
- First token: ~100ms
- Subsequent tokens: ~60-80 tokens/sec

**Qwen3-8B on CPU (AMD Ryzen 9 5950X):**
- First token: ~500ms
- Subsequent tokens: ~15-20 tokens/sec

**Qwen3-1.7B on CPU:**
- Subsequent tokens: ~30-40 tokens/sec

Performance varies based on sequence length, hardware, ONNX Runtime build, and CUDA/cuDNN versions.

## Development Roadmap

See `todo.md` for planned improvements:
- [x] Fix ONNX Runtime 1.19.0 API compatibility
- [x] Enable GPU inference with CUDA provider
- [x] Fix tensor lifecycle issues in decode loop
- [x] Add cuDNN 9 support
- [ ] Remove transformers dependency at inference time
- [ ] Create minimal runtime config during export
- [ ] Add HuggingFace download automation
- [ ] Expand test suite with evaluation metrics
- [ ] Add accuracy comparison with HF baseline
- [ ] Support multi-GPU inference
- [ ] Add batch inference support

## Troubleshooting

### C++ Build Issues

#### ONNX Runtime API Compatibility
The C++ code requires ONNX Runtime 1.19.0 or later. Key API changes:
- `AddSessionConfigEntry` → `AddConfigEntry`
- `SetLogVerbosityLevel` removed (use session options)
- `SetProviders` → `AppendExecutionProvider_CUDA` for GPU

If you encounter API errors, ensure you're using ONNX Runtime 1.19.0+.

#### Missing cuDNN for GPU Inference
**Error**: `Failed to load CUDA execution provider` or `cuDNN library not found`

**Solution**: Install cuDNN 9 and set up environment:
```bash
# Install via conda (recommended)
conda install -c conda-forge cudnn=9

# Use the wrapper script to run
./run_gpu_inference.sh "Your prompt here"
```

The wrapper script sets:
- `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (finds cuDNN)
- `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` (avoids version conflicts)

#### Opset Version Error
**Error**: `Could not find an implementation for Mul(14) node`

**Cause**: ONNX Runtime GPU builds may not include all CPU fallback kernels for newer opsets.

**Solution**: The exporter now uses opset 13 by default. If you have an old export, re-run:
```bash
python exporter.py --config config.json --mode export
```

#### Tensor Type Mismatches
**Error**: `Unexpected input data type. Actual: (tensor(int64)), expected: (tensor(float))`

**Cause**: Input tensors not properly reconstructed between decode iterations.

**Solution**: Already fixed in current code. Ensure you have the latest version where:
- `current_tokens`, `history_len_data`, `ids_len_data`, `attention_mask_data` persist across iterations
- KV cache tensors are moved (not copied) from outputs to inputs
- Memory info and shapes are properly maintained

### ONNX Runtime not found
Set the `ONNXRUNTIME_ROOT_DIR` CMake variable:
```bash
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-gpu-1.19.0
```

### Tokenizer errors
Ensure `tokenizer.json` exists in your model directory:
```bash
ls -la /path/to/Qwen3-8B/tokenizer.json
```

### CUDA not available
Check available providers in C++:
```bash
# The program will print: "Using CUDA execution provider on GPU 0"
# Or fall back to: "CUDA provider not available, using CPU"
```

For Python:
```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### GPU Memory Issues
If you run out of GPU memory:
- Use a smaller model (Qwen3-1.7B instead of 8B)
- Reduce batch size (currently 1)
- Select a different GPU: Modify `device_id` in `onnx_inference.cpp`
- Close other GPU applications

### Performance Issues
- Ensure you're using GPU inference (check for "Using CUDA" message)
- Build ONNX Runtime from source with optimizations
- Use cuDNN 9 for best performance
- Check GPU utilization with `nvidia-smi`

## Credits

This project is based on the [Native-LLM-for-Android](https://github.com/DakeQQ/Native-LLM-for-Android) example by DakeQQ, with significant refactoring for:
- Improved maintainability and code quality
- Better error handling and logging
- Type safety (Python type hints)
- Modular architecture
- Enhanced documentation

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

Author: Shivani Gowda
Repository: llm-inference
