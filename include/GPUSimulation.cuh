#pragma once
#include <vector>
#include "ISimulation.h"
#include <cuda_runtime.h>

enum class Strategy {NAIVE, SHARED};


class GPUSimulation : public ISimulation {
    
    Particle* d_particles;
    int total_particles;
    int nBlocks;
    Strategy strategy;
    size_t bytes;
public:
    GPUSimulation(std::vector<Particle>& h_particles, Strategy strategy);
    ~GPUSimulation();
    void step() override;
    void write_to_host(std::vector<Particle>& h_particles) override;
    std::string get_name() override;
    void step_naive();
    void step_shared();
};