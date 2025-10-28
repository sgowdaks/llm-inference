cmake_minimum_required(VERSION 3.16)
project(onnx_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(nlohmann_json REQUIRED)

# ONNX Runtime (point this to your ONNX Runtime install)
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to ONNX Runtime installation")
if(NOT ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "Please specify ONNXRUNTIME_ROOT_DIR")
endif()

include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

# Add tokenizers C++ bindings as a subproject (builds Rust FFI + C++ wrapper)
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/bindings/cpp)
if(EXISTS ${TOKENIZERS_CPP_ROOT}/CMakeLists.txt)
    add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cpp_build)
    include_directories(${TOKENIZERS_CPP_ROOT}/include)
else()
    message(WARNING "Tokenizers C++ bindings not found at ${TOKENIZERS_CPP_ROOT}; you can still link an external tokenizers installation by adding include/link directories")
endif()

# Create executable
add_executable(onnx_inference onnx_inference.cpp)

# Link libraries
# tokenizers_cpp and tokenizers_c are the targets created by the tokenizers CMake
target_link_libraries(onnx_inference
    PRIVATE
    onnxruntime
    nlohmann_json::nlohmann_json
)

if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference PRIVATE tokenizers_cpp tokenizers_c)
endif()