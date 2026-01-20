cmake_minimum_required(VERSION 3.10)
project(NBody LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)


include_directories(include)

set(SOURCES 
    src/main.cpp 
    src/CPUSimulation.cpp 
    src/GPUSimulation.cu
)

add_executable(nbody_modular ${SOURCES})

target_link_libraries(nbody_modular PRIVATE OpenGL::GL glfw)
target_compile_options(nbody_modular PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)