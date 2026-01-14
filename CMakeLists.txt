cmake_minimum_required(VERSION 3.10)
project(NBody_CPU)

set(CMAKE_CXX_STANDARD 17)

# We define the executable
add_executable(nbody_cpu main.cpp)

# OPTIMIZATION FLAGS:
# -O3: Maximum optimization
# -march=native: Use AVX instructions available on your i7-7700HQ
target_compile_options(nbody_cpu PRIVATE -O3 -march=native)