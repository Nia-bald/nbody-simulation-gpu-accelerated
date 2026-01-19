cmake_minimum_required(VERSION 3.10)
project(NBody LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# We define the executable
add_executable(nbody_cpu nbody_cpu.cpp)
add_executable(nbody_gpu nbody_naive_gpu.cu)

target_link_libraries(nbody_cpu PRIVATE OpenGL::GL glfw)
target_link_libraries(nbody_gpu PRIVATE OpenGL::GL glfw)


# OPTIMIZATION FLAGS:
# -O3: Maximum optimization
# -march=native: Use AVX instructions available on your i7-7700HQ
target_compile_options(nbody_cpu PRIVATE -O3 -march=native)
target_compile_options(nbody_gpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)