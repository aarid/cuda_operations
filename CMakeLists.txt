cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_demo CUDA CXX)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(cuda_demo main.cu)

# Set CUDA specific properties
set_target_properties(cuda_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Enable debugging in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(cuda_demo PROPERTIES 
        CUDA_FLAGS "-G"
    )
endif()