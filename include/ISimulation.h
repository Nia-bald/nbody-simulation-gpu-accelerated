#pragma once
#include <string>
#include <iostream>


// defining all constants here
const float G = 1.0f;
const float SOFTENNING = 0.1f;
const float DT = 0.01f;
const int BLOCK_SIZE = 256; // number of threads in a block, wise to 

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

class ISimulation {
public:
    virtual ~ISimulation() = default;
    virtual void step() = 0;
    virtual std::string get_name() = 0;
    virtual void write_to_host(std::vector<Particle>& h_particles) = 0;
};
