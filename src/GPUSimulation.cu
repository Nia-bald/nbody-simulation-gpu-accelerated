#include "GPUSimulation.cuh"

GPUSimulation::GPUSimulation(std::vector<Particle>& h_particles, Strategy strgy){
    strategy = strgy;
    total_particles = h_particles.size();
    nBlocks = (total_particles + BLOCK_SIZE - 1)/BLOCK_SIZE;
    bytes = total_particles*sizeof(Particle);
    cudaMalloc(&d_particles, bytes);
    cudaMemcpy(d_particles, h_particles.data(), bytes, cudaMemcpyHostToDevice);
    std::cout << "Launching" << nBlocks << " blocks with " << BLOCK_SIZE << "threads each." << std::endl;
    cudaDeviceSynchronize();
}


std::string GPUSimulation::get_name() {
    if (strategy == Strategy::NAIVE) {
        return "GPU NAIVE";
    }
    if (strategy == Strategy::SHARED) {
        return "GPU SHARED";
    }
    return "GPU UNKNOWN";
}

__global__ void computeVelocityNaive(Particle* p, int n) {

    // all threads are labeled from 0 to k corresponding to each particly
    // note that thread is an abstract concept so same cuda core might execute multiple threads
    // albeit not at the same time
    // blockIdx which block number
    // blockDim how many threads in a black
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
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

        fx += -multiplicative_factor*dx;
        fy += -multiplicative_factor*dy;
        fz += -multiplicative_factor*dz;
    }

    p[i].vx += (fx/p[i].mass)*DT;
    p[i].vy += (fy/p[i].mass)*DT;
    p[i].vz += (fz/p[i].mass)*DT;
}

__global__ void computeVelocityShared(Particle* p, int n) {
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

void GPUSimulation::step_naive() {
    computeVelocityNaive<<<nBlocks, BLOCK_SIZE>>>(d_particles, total_particles);
    updatePositions<<<nBlocks, BLOCK_SIZE>>>(d_particles, total_particles);
}

void GPUSimulation::step_shared() {
    computeVelocityShared<<<nBlocks, BLOCK_SIZE>>>(d_particles, total_particles);
    updatePositions<<<nBlocks, BLOCK_SIZE>>>(d_particles, total_particles);
}


void GPUSimulation::step(){
    if (strategy == Strategy :: NAIVE){
        step_naive();
    }
    if (strategy == Strategy :: SHARED){
        step_shared();
    }
    cudaDeviceSynchronize();
}

void GPUSimulation::write_to_host(std::vector<Particle>& h_particles){
    cudaMemcpy(h_particles.data(), d_particles, bytes, cudaMemcpyDeviceToHost);
}


GPUSimulation::~GPUSimulation(){
    cudaFree(d_particles);
}