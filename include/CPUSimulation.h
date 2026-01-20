#pragma once
#include <vector>
#include "ISimulation.h"
#include <cmath>

class CPUSimulation : public ISimulation {
    
    std::vector<Particle>& particles;
    int total_particles;
public:
    CPUSimulation(std::vector<Particle>& data);
    void step() override;
    void write_to_host(std::vector<Particle>& h_particles) override;
    std::string get_name() override;
};