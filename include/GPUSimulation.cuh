#pragma once
#include <vector>
#include "ISimulation.h"
#include <cuda_runtime.h>
// #include <cuda_gl_interop.h>

enum class Strategy {NAIVE, SHARED};


class GPUSimulation : public ISimulation {
    
    Particle* d_particles;
    float3* d_posVisual; // pointer to openGL buffer for visualization
    float3* d_colorVisual; // pointer to openGL buffer for visualization

    cudaGraphicsResource* cuda_color_resource = nullptr; // Handle to the OpenGL VBO
    cudaGraphicsResource* cuda_vbo_resource = nullptr; // Handle to the OpenGL VBO
    int total_particles;
    int nBlocks;
    Strategy strategy;
    size_t bytes;
    bool useInterop;
public:
    GPUSimulation(std::vector<Particle>& h_particles, Strategy strategy, unsigned int vbo_pos = 0, unsigned int vbo_col = 0);
    ~GPUSimulation();
    void step() override;
    void write_to_host(std::vector<Particle>& h_particles) override;
    std::string get_name() override;
    void step_naive();
    void step_shared();
};