cmake_minimum_required(VERSION 3.16)
project(qwen_onnx_inference VERSION 0.1.0 LANGUAGES CXX)

# Project information
set(PROJECT_DESCRIPTION "High-performance ONNX inference for Qwen language models")
set(PROJECT_AUTHOR "Shivani Gowda")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(nlohmann_json REQUIRED)

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to ONNX Runtime installation")
if(NOT ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "Please specify ONNXRUNTIME_ROOT_DIR with -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime")
endif()

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT_DIR}")
include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

# Tokenizers C++ bindings
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers/bindings/cpp)
if(EXISTS ${TOKENIZERS_CPP_ROOT}/CMakeLists.txt)
    message(STATUS "Building tokenizers-cpp from: ${TOKENIZERS_CPP_ROOT}")
    add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cpp_build)
    include_directories(${TOKENIZERS_CPP_ROOT}/include)
else()
    message(WARNING "Tokenizers C++ bindings not found at ${TOKENIZERS_CPP_ROOT}")
    message(WARNING "You can link an external tokenizers installation by adding include/link directories")
endif()

# Source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_inference.cpp
)

# Create executable
add_executable(onnx_inference ${SOURCES})

# Include directories
target_include_directories(onnx_inference
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(onnx_inference
    PRIVATE
    onnxruntime
    nlohmann_json::nlohmann_json
)

# Link tokenizers if available
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference PRIVATE tokenizers_cpp tokenizers_c)
endif()

# Create optimized ONNX inference executable
add_executable(onnx_inference_optimized ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_inference_optimized.cpp)
target_include_directories(onnx_inference_optimized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onnx_inference_optimized PRIVATE onnxruntime nlohmann_json::nlohmann_json)
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference_optimized PRIVATE tokenizers_cpp tokenizers_c)
endif()

# Create fast GPU ONNX inference executable
add_executable(onnx_inference_fast ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_inference_fast.cpp)
target_include_directories(onnx_inference_fast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onnx_inference_fast PRIVATE onnxruntime nlohmann_json::nlohmann_json)
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference_fast PRIVATE tokenizers_cpp tokenizers_c)
endif()

# Create ULTRA-fast GPU ONNX inference executable
add_executable(onnx_inference_ultra ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_inference_ultra.cpp)
target_include_directories(onnx_inference_ultra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onnx_inference_ultra PRIVATE onnxruntime nlohmann_json::nlohmann_json)
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference_ultra PRIVATE tokenizers_cpp tokenizers_c)
endif()

# Create PROFILED ONNX inference executable
add_executable(onnx_inference_profiled ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_inference_profiled.cpp)
target_include_directories(onnx_inference_profiled PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(onnx_inference_profiled PRIVATE onnxruntime nlohmann_json::nlohmann_json)
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference_profiled PRIVATE tokenizers_cpp tokenizers_c)
endif()
if(TARGET tokenizers_cpp)
    target_link_libraries(onnx_inference_optimized PRIVATE tokenizers_cpp tokenizers_c)
endif()

# Installation rules
install(TARGETS onnx_inference onnx_inference_optimized
    RUNTIME DESTINATION bin
)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")