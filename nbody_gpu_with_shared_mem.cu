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
    // this new computevelocity function is essentially and practically exactly the same code
    // it is just that instead of loading everysingle variable from VRAM everysingle time
    // we load the data in blocks to shared memory for reapeated use

    // all threads are labeled from 0 to k corresponding to each particly
    // note that thread is an abstract concept so same cuda core might execute multiple threads
    // albeit not at the same time
    // blockIdx which block number
    // blockDim how many threads in a black
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float my_x, my_y, my_z;
    if (i < n){
        my_x = p[i].x;
        my_y = p[i].y;
        my_z = p[i].z;
    };
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    __shared__ float3 sh_pos[BLOCK_SIZE];
    __shared__ float sh_mass[BLOCK_SIZE];

    for (int tile = 0; tile < gridDim.x; tile++){
        // ----- section 1 -------
        // when thinking of this section do not think of it
        // as just these instruction think all the thread will be executing these instructions
        // here we load all the data tile by tile
        // here threadIdx.x gives you the local thread number
        // in the first for loop all the data for the first tile is loaded
        // in this case the way we have writtin this section, number of particle loaded in a single == number of threads in single block
        int idx = tile*blockDim.x + threadIdx.x;

        if (idx < n){
            sh_pos[threadIdx.x] = make_float3(p[idx].x, p[idx].y, p[idx].z);
            sh_mass[threadIdx.x] = p[idx].mass;
        }
        else{
            sh_pos[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
            sh_mass[threadIdx.x] = 0.0f;
        }
        // ----- block 1 -----

        __syncthreads();
        // the above line stops the gpu, so after this line it safe to say that all the data fro tile 1 has been loaded
        // and the data fro tile one includes everything from 0 to 255

        // once the data for first tile is loaded we need to do computation for this tile
        if (i < n) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; k++){
                float dx = sh_pos[k].x - my_x;
                float dy = sh_pos[k].y - my_y;
                float dz = sh_pos[k].z - my_z;

                float r_squared = dx*dx + dy*dy + dz*dz + SOFTENNING*SOFTENNING;
                float r_inv = rsqrtf(r_squared);
                float r_inv_cubed = r_inv*r_inv*r_inv;

                float multiplicative_factor = G*sh_mass[k]*sh_mass[k]*r_inv_cubed;

                fx += multiplicative_factor*dx;
                fy += multiplicative_factor*dy;
                fz += multiplicative_factor*dz;
            }
        }

        // finish for everyone to finish calculating before we overwrite shared Mem
        __syncthreads();
    }

    if (i < n){
        p[i].vx += (fx/p[i].mass)*DT;
        p[i].vy += (fy/p[i].mass)*DT;
        p[i].vz += (fz/p[i].mass)*DT;
    }
}

__global__ void updatePositions(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;

    p[i].x += p[i].vx*DT;
    p[i].y += p[i].vy*DT;
    p[i].z += p[i].vz*DT;
}


void drawParticles(const std::vector<Particle>& particles) {
    glClear(GL_COLOR_BUFFER_BIT); // Clear previous frame
    glLoadIdentity();             // Reset coordinate system (Scale = 1.0)
    glScalef(0.01f, 0.01f, 0.01f); // Zoom out so coordinates -100 to 100 fit screen

    glBegin(GL_POINTS);
    for (const auto& p : particles) {
        // Color based on Z-depth (Blue = far, White = close)
        float intensity = (p.z + 100.0f) / 200.0f; 
        if(intensity < 0.1f) intensity = 0.1f;
        if(intensity > 1.0f) intensity = 1.0f;
        
        glColor3f(intensity, intensity, 1.0f); 
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();

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
        h_particles[i].mass = massDist(gen);
        h_particles[i].vx = 0.0f;
        h_particles[i].vy = 0.0f;
        h_particles[i].vz = 0.0f;
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