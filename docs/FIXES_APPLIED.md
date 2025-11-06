# Fixes Applied to C++ ONNX Inference Code

## Issues Found and Fixed

### 1. **API Compatibility Issues** ✅ FIXED
   - **Problem**: Code used deprecated/incorrect ONNX Runtime C++ API calls
   - **Fixes Applied**:
     - Changed `AddSessionConfigEntry` → `AddConfigEntry`
     - Removed `SetLogVerbosityLevel` (doesn't exist in this API version)
     - Changed `SetProviders` → `AppendExecutionProvider_CUDA`
     - Added try-catch for CUDA provider initialization

### 2. **Memory Safety Issue** ✅ FIXED
   - **Problem**: `GetInputNameAllocated()` returned `AllocatedStringPtr` that went out of scope
   - **Fix**: Store `AllocatedStringPtr` objects before extracting raw pointers
   ```cpp
   std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
   input_name_ptrs.push_back(session_->GetInputNameAllocated(i, allocator));
   input_names.push_back(input_name_ptrs.back().get());
   ```

### 3. **Ort::Value Copy Issue** ✅ FIXED
   - **Problem**: Attempted to copy `Ort::Value` which has deleted copy constructor
   - **Fix**: Use reference instead
   ```cpp
   auto& token_tensor = output_tensors[output_tensors.size() - 2];  // Reference, not copy
   ```

### 4. **CUDA Error Handling** ✅ FIXED
   - **Problem**: CUDA initialization failure caused program crash
   - **Fix**: Added graceful fallback to CPU
   ```cpp
   try {
       OrtCUDAProviderOptions cuda_options;
       session_opts.AppendExecutionProvider_CUDA(cuda_options);
       use_cuda_ = true;
   } catch (const std::exception& e) {
       std::cout << "CUDA provider failed, falling back to CPU" << std::endl;
       use_cuda_ = false;
   }
   ```

## Remaining Issue: ONNX Runtime Kernel Support

### Problem
The ONNX Runtime GPU build (version 1.19.0) at `/home/shivani/onnxruntime-linux-x64-gpu-1.19.0` 
is missing CPU kernel implementations for the `Mul` operator with opset 14+.

**Error**: `Could not find an implementation for Mul(14) node with name '/Mul_1'`

### Solutions (Choose One)

#### ✅ RECOMMENDED: Solution 1 - Re-export Model with Lower Opset
The exporter has been updated to use opset 13 instead of 17.

**Steps**:
1. Ensure Python dependencies are installed:
   ```bash
   pip install torch transformers onnxruntime numpy
   ```

2. Re-export the model:
   ```bash
   python exporter.py --config config.json --mode export
   ```

3. Rebuild and run C++ inference:
   ```bash
   cd build && make -j4
   cd .. && ./build/onnx_inference "What is 2+2?"
   ```

#### Solution 2 - Use CPU-only ONNX Runtime
Download ONNX Runtime CPU build from https://github.com/microsoft/onnxruntime/releases

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz
tar -xzf onnxruntime-linux-x64-1.19.0.tgz
```

Update CMake:
```bash
cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-linux-x64-1.19.0
make -j4
```

#### Solution 3 - Install System ONNX Runtime
```bash
# Ubuntu/Debian
sudo apt-get install libonnxruntime-dev

# Then update CMakeLists.txt to use system libraries
```

## Build Instructions

### Current Build Status
✅ Code compiles successfully
❌ Runtime fails due to missing ONNX Runtime kernels

### To Build:
```bash
cd /home/shivani/work/llm-inference/build
make -j4
```

### To Run (after fixing ONNX Runtime issue):
```bash
cd /home/shivani/work/llm-inference
./build/onnx_inference "Your prompt here"
```

## Code Changes Summary

**File**: `onnx_inference.cpp`

1. Updated `build_session()` method:
   - Fixed ONNX Runtime API calls
   - Added CUDA error handling
   - Simplified session options (commented out problematic config entries)

2. Updated `run_inference()` method:
   - Fixed memory safety with `AllocatedStringPtr`
   - Changed `auto token_tensor` to `auto& token_tensor`
   - Added debug output for inputs/outputs

3. General improvements:
   - Better error messages
   - Provider selection logging
   - Graceful CUDA fallback

## Next Steps

1. **Re-export the model** with opset 13 (recommended)
2. OR switch to CPU-only ONNX Runtime
3. Test inference with: `./build/onnx_inference "Test prompt"`
4. If successful, run with longer prompts and test generation quality

## Testing Checklist

- [ ] Model re-exported with opset 13
- [ ] C++ code rebuilt successfully
- [ ] Inference runs without crashes
- [ ] Token generation produces reasonable output
- [ ] Performance is acceptable (check tokens/second)
- [ ] Test with various prompt lengths
- [ ] Compare output with Python version for accuracy
