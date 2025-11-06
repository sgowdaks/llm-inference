# Building Qwen ONNX Inference

This guide covers building and setting up the C++ ONNX inference engine for Qwen models.

## Prerequisites

### System Requirements
- Linux (tested on Ubuntu 20.04+)
- CMake 3.16 or higher
- C++17 compatible compiler (GCC 9+, Clang 10+)
- 8GB RAM minimum (16GB+ recommended for 8B model)
- GPU: NVIDIA GPU with CUDA Compute Capability 7.0+ (optional, for GPU acceleration)

### Software Dependencies

#### Required
- **ONNX Runtime 1.19.0+** (GPU build recommended)
- **nlohmann/json** library
- **tokenizers-cpp** (included as submodule)

#### For GPU Inference (Recommended)
- **CUDA Toolkit** 11.x or 12.x
- **cuDNN 9.x**

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sgowdaks/llm-inference.git
cd llm-inference
git submodule update --init --recursive
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    wget \
    nlohmann-json3-dev
```

#### Arch Linux
```bash
sudo pacman -S base-devel cmake nlohmann-json
```

### 3. Install ONNX Runtime

#### Option A: Download Pre-built GPU Binary (Recommended)
```bash
cd /path/to/your/libs
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.19.0.tgz
```

#### Option B: CPU-only Build
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-1.19.0.tgz
```

### 4. Install CUDA and cuDNN (For GPU Inference)

#### Using Conda (Recommended)
```bash
conda install -c conda-forge cudnn=9
```

#### Manual Installation
Download cuDNN 9 from [NVIDIA Developer](https://developer.nvidia.com/cudnn) and follow installation instructions.

### 5. Configure the Build

```bash
mkdir build
cd build

# For GPU build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-gpu-1.19.0

# For CPU-only build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-1.19.0
```

**Common CMake Options:**
- `-DCMAKE_BUILD_TYPE=Release` - Optimized release build (default)
- `-DCMAKE_BUILD_TYPE=Debug` - Debug build with symbols
- `-DCMAKE_INSTALL_PREFIX=/path` - Custom installation directory

### 6. Build

```bash
# Use all available cores
make -j$(nproc)

# Or specify number of cores
make -j4
```

The executable will be created at `build/bin/onnx_inference`.

### 7. Verify Build

```bash
# Check executable
ls -lh build/bin/onnx_inference

# Check dependencies
ldd build/bin/onnx_inference
```

## Python Environment Setup

### 1. Create Virtual Environment

```bash
# Using venv
python3 -m venv .venv
source .venv/bin/activate

# Or using conda
conda create -n qwen-inference python=3.10
conda activate qwen-inference
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

For GPU support in Python:
```bash
pip install onnxruntime-gpu
```

### 3. Install Package in Development Mode

```bash
pip install -e .
```

## Configuration

### 1. Setup Configuration File

```bash
cp configs/config.example.json configs/config.json
```

### 2. Edit Configuration

Update `configs/config.json` with your paths:

```json
{
    "paths": {
        "model_path": "/path/to/Qwen3-8B",
        "output_dir": "./export3-8B",
        "onnx_file": "./export3-8B/qwen.onnx"
    }
}
```

### 3. Download Model (if needed)

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B

# Or using git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B
```

## Export Model to ONNX

```bash
python src/exporter.py --config configs/config.json --mode export
```

This will:
1. Load the PyTorch model
2. Apply optimizations (quantization, KV-cache setup)
3. Export to ONNX format
4. Save model weights to the output directory

Expected output size: ~341KB for the graph + separate weight files.

## Running Inference

### C++ Inference

#### GPU (Recommended)
```bash
./scripts/run_gpu_inference.sh "What is the capital of France?"
```

#### CPU
```bash
./build/bin/onnx_inference "What is the capital of France?"
```

### Python Inference

```bash
python src/onnx_inference.py --config configs/config.json --prompt "What is machine learning?"
```

## Testing

Run the test suite to verify everything works:

```bash
# Quick validation test
./scripts/quick_test.sh

# Full inference test
./scripts/test_inference.sh
```

## Troubleshooting

### Build Errors

#### CMake cannot find nlohmann/json
```bash
# Install via package manager
sudo apt-get install nlohmann-json3-dev

# Or download header-only library
mkdir -p external/json
cd external/json
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
```

Then update CMakeLists.txt to include the path.

#### ONNX Runtime not found
Ensure `ONNXRUNTIME_ROOT_DIR` points to the extracted directory:
```bash
cmake .. -DONNXRUNTIME_ROOT_DIR=/absolute/path/to/onnxruntime-linux-x64-gpu-1.19.0
```

#### Tokenizers build fails
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update submodules
git submodule update --init --recursive
```

### Runtime Errors

#### cuDNN not found
```bash
# Set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Or use the wrapper script
./scripts/run_gpu_inference.sh "test"
```

#### GLIBCXX version mismatch
```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

#### Model file not found
Ensure you've exported the model and paths in config.json are correct:
```bash
python src/exporter.py --config configs/config.json --mode export
```

### Performance Issues

#### Slow CPU inference
- Use GPU inference (10-50x faster)
- Use smaller model (Qwen3-1.7B)
- Reduce max_tokens in config

#### GPU memory errors
- Use smaller batch size
- Select different GPU: modify device_id in code
- Use smaller model

## Advanced Configuration

### Multi-GPU Setup

Edit `src/onnx_inference.cpp`:
```cpp
cuda_options.device_id = 1;  // Use GPU 1 instead of GPU 0
```

### Custom Optimization Levels

Edit CMakeLists.txt:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
```

### Profiling

```bash
# Build with profiling
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Run with perf
perf record -g ./build/bin/onnx_inference "test"
perf report
```

## Next Steps

- Read [CONTRIBUTING.md](CONTRIBUTING.md) to contribute to the project
- Check [docs/todo.md](todo.md) for planned features
- See [examples/](../examples/) for usage examples

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review [docs/FIXES_APPLIED.md](FIXES_APPLIED.md) for known issues and fixes
