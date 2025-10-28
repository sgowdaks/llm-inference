cmake_minimum_required(VERSION 3.16)
project(tokenizer_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(nlohmann_json REQUIRED)

# Add tokenizers C++ bindings (assumes tokenizers repo is under ./tokenizers)
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizers/bindings/cpp)
add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/tokenizers_cpp_build)
include_directories(${TOKENIZERS_CPP_ROOT}/include)

# Test executable (source is in parent directory)
add_executable(tokenizer_test ${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE tokenizers_cpp tokenizers_c nlohmann_json::nlohmann_json)
