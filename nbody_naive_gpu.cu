#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h> // Graphics Header

struct Particle {

    float x, y, z;
    float vx, vy, vz;
    float mass;
};


const float G = 1.0f;
const float SOFTENNING = 0.1f;
const float DT = 0.01f;
const int BLOCK_SIZE = 256; // number of threads in a block, wise to 

__global__ void computeVelocity(Particle* p, int n) {

    // all threads are labeled from 0 to k corresponding to each particly
    // note that thread is an abstract concept so same cuda core might execute multiple threads
    // albeit not at the same time
    // blockIdx which block number
    // blockDim how many threads in a black
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > n) return;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;


    for (int j = 0; j < n; j++){

        if (i == j) continue;
        float dx = p[i].x - p[j].x;
        float dy = p[i].y - p[j].y;
        float dz = p[i].z - p[j].z;

        float r_squared = dx*dx + dy*dy + dz*dz + SOFTENNING*SOFTENNING;
        float r_inv = rsqrtf(r_squared);
        float r_inv_cubed = r_inv*r_inv*r_inv;

        float multiplicative_factor = G*p[i].mass*p[j].mass*r_inv_cubed;

        fx += multiplicative_factor*dx;
        fy += multiplicative_factor*dy;
        fz += multiplicative_factor*dz;
    }

    p[i].vx += (fx/p[i].mass)*DT;
    p[i].vy += (fy/p[i].mass)*DT;
    p[i].vz += (fz/p[i].mass)*DT;
}

__global__ void updatePositions(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > n) return;

    p[i].x += p[i].vx*DT;
    p[i].y += p[i].vy*DT;
    p[i].z += p[i].vz*DT;
}


int main() {
    const int N = 10000;
    const int STEPS = 10;

    size_t bytes = N*sizeof(Particle);

    std::vector<Particle> h_particles(N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> posDist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> massDist(1.0f, 5.0f);

    for (int i = 0; i < N; i++){
        h_particles[i].x = posDist(gen);
        h_particles[i].y = posDist(gen);
        h_particles[i].z = posDist(gen);
        h_particles[i].mass = posDist(gen);
    }

    Particle* d_particles;

    cudaMalloc(&d_particles, bytes);
    cudaMemcpy(d_particles, h_particles.data(), bytes, cudaMemcpyHostToDevice);

    // grid size aka number of blocks required;
    int nBlocks = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

    std::cout << "Launching" << nBlocks << " blocks with " << BLOCK_SIZE << "threads each." << std::endl;

    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < STEPS; i++){
        computeVelocity<<<nBlocks, BLOCK_SIZE>>>(d_particles, N);
        updatePositions<<<nBlocks, BLOCK_SIZE>>>(d_particles, N);
        cudaDeviceSynchronize();
        std::cout << "." << std::flush;
    }

    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    cudaFree(d_particles);

    std::cout << "Total Time:     " << total_ms << " ms" << std::endl;
    std::cout << "Avg Time/Frame: " << (float)total_ms / STEPS << " ms" << std::endl;
    
    double ops_per_sec = ((double)N * N * STEPS) / (total_ms / 1000.0);
    std::cout << "Throughput:     " << ops_per_sec / 1e6 << " M-Interactions/sec" << std::endl;

    return 0;


}