#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <GLFW/glfw3.h> // Graphics Header

// --- STRUCTURES ---
struct Particle {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;         // Mass (No longer assumed to be 1.0)
};

// --- CONSTANTS ---
// We use 1.0 for G to keep simulation stable at this small scale, 
// but it is now a variable you can tune.
const float G = 6.0f;           
const float SOFTENING = 0.1f;   // Prevents division by zero
const float DT = 0.01f;         // Time step

// --- PHYSICS ENGINE (CPU) ---
void updatePhysics(std::vector<Particle>& particles) {
    int N = particles.size();

    // Loop over all particles to calculate forces
    for (int i = 0; i < N; ++i) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        for (int j = 0; j < N; ++j) {
            if (i == j) continue; // Newton's 3rd Law: You don't attract yourself

            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;

            // r^2 = dx^2 + dy^2 + dz^2 + epsilon^2
            float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
            float invDist = 1.0f / std::sqrt(distSq);
            float invDist3 = invDist * invDist * invDist;

            // --- STRICT PHYSICS FORMULA ---
            // F_scalar = (G * m1 * m2) / r^2
            // F_vector = F_scalar * (vec_r / r)
            // F_vector = (G * m1 * m2 * vec_r) / r^3
            
            float forceMagnitude = G * particles[i].mass * particles[j].mass * invDist3;

            fx += dx * forceMagnitude;
            fy += dy * forceMagnitude;
            fz += dz * forceMagnitude;
        }

        // --- NEWTON'S SECOND LAW (a = F / m) ---
        // Acceleration = Force / Mass_of_i
        float ax = fx / particles[i].mass;
        float ay = fy / particles[i].mass;
        float az = fz / particles[i].mass;

        // Update Velocity (v = u + at)
        particles[i].vx += ax * DT;
        particles[i].vy += ay * DT;
        particles[i].vz += az * DT;
    }

    // Update Positions (s = s + vt)
    for (int i = 0; i < N; ++i) {
        particles[i].x += particles[i].vx * DT;
        particles[i].y += particles[i].vy * DT;
        particles[i].z += particles[i].vz * DT;
    }
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
    // PARAMETERS
    const int N = 20000;    // 10,000 Particles
    const int STEPS = 10;   // Run 10 benchmark frames

    std::vector<Particle> particles(N);

    // Random Initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> posDist(-100.0f, 100.0f); // Position
    std::uniform_real_distribution<float> massDist(1.0f, 5.0f);     // Random Mass (1kg to 5kg)
    
    for (int i = 0; i < N; ++i) {
        particles[i].x = posDist(gen);
        particles[i].y = posDist(gen);
        particles[i].z = posDist(gen);
        particles[i].vx = 0;
        particles[i].vy = 0;
        particles[i].vz = 0;
        particles[i].mass = massDist(gen); // Assign random mass
    }
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA N-Body Simulation", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    std::cout << "--- N-BODY SIMULATION (CPU REFERENCE) ---" << std::endl;
    std::cout << "Particles: " << N << std::endl;
    std::cout << "Mass/G:    Explicitly calculated" << std::endl;
    std::cout << "Simulating " << STEPS << " frames..." << std::endl;

    // --- BENCHMARK ---
    auto start = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)){
        updatePhysics(particles);
        drawParticles(particles);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Total Time:     " << total_ms << " ms" << std::endl;
    std::cout << "Avg Time/Frame: " << (float)total_ms / STEPS << " ms" << std::endl;
    
    // Calculate FLOPS estimate
    // Interactions per frame = N^2
    // Operations per interaction approx 20 FLOPS (mul, add, div, sqrt)
    double interactions = (double)N * (double)N;
    double ops_per_sec = (interactions * STEPS) / (total_ms / 1000.0);
    
    std::cout << "Throughput:     " << ops_per_sec / 1e6 << " Million Interactions/sec" << std::endl;

    return 0;
}