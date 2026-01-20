#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include "CPUSimulation.h"
#include "GPUSimulation.cuh"
#include <GLFW/glfw3.h> // Graphics Header


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
    int N = 10000;
    int STEPS = 20;
    std::string mode = "shared";
    std::string run_mode = "visual"; //visual or benchmark
    std::vector<Particle> particles(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> posDist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> massDist(1.0f, 5.0f);
    for (int i = 0; i < N; i++){
        particles[i].x = posDist(gen);
        particles[i].y = posDist(gen);
        particles[i].z = posDist(gen);
        particles[i].mass = massDist(gen);
        particles[i].vx = 0.0f;
        particles[i].vy = 0.0f;
        particles[i].vz = 0.0f;
    }

    std::unique_ptr<ISimulation> sim;

    if (mode == "cpu") {
        sim = std::make_unique<CPUSimulation>(particles);
    } else if (mode == "naive") {
        sim = std::make_unique<GPUSimulation>(particles, Strategy::NAIVE);
    } else if (mode == "shared") {
        sim = std::make_unique<GPUSimulation>(particles, Strategy::SHARED);
    } else {
        std::cerr << "Usage: ./nbody [cpu|naive|shared] [N] [STEPS]\n";
        return 1;
    }    


    if (run_mode == "benchmark") {
        std::cout << "--- BENCHMARK: " << sim->get_name() << " ---\n";
        std::cout << "Particles: " << N << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<STEPS; i++) {
            sim->step();
            if(i % 5 == 0) std::cout << "." << std::flush;
        }
        std::cout << "\n";
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Total Time: " << ms << " ms\n";
        std::cout << "Throughput: " << (double)N*N*STEPS / (ms/1000.0) / 1e9 << " G-Interactions/sec\n";
    
    }
    if (run_mode == "visual") {
        if (!glfwInit()) return -1;

        GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA N-Body Simulation", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);
        while (!glfwWindowShouldClose(window)){
            sim->step();
            sim->write_to_host(particles);
            drawParticles(particles);
            glfwSwapBuffers(window);
            glfwPollEvents();
    
        }
    }
    return 0;
}