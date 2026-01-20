#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <cstdlib>
#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include "CPUSimulation.h"
#include "GPUSimulation.cuh"


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

int main(int argc, char* argv[]) {
    // 1. Set Default Values
    int N = 10000;
    int STEPS = 20;
    std::string streategy = "shared";
    std::string run_mode = "visual_vbo";  

    // 2. Override with arguments if they exist
    // argv[0] is the program name, so we start checking from argv[1]
    if (argc > 1) {
        N = std::stoi(argv[1]); 
    }
    if (argc > 2) {
        STEPS = std::stoi(argv[2]);
    }
    if (argc > 3) {
        streategy = argv[3];
    }
    if (argc > 4) {
        run_mode = argv[4];
    }
    std::vector<Particle> particles(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> posDist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> massDist(1.0f, 5.0f);
    for (int i = 0; i < N; i++){
        particles[i].x = posDist(gen);
        particles[i].y = posDist(gen);
        particles[i].z = posDist(gen);
        particles[i].mass = massDist(gen);
        float r = std::sqrt((particles[i].x*particles[i].x) + (particles[i].y*particles[i].y) + (particles[i].z*particles[i].z) + SOFTENNING*SOFTENNING);
        float magnitude = 10.0f;
        particles[i].vx = -magnitude*particles[i].y/r; // Simple rotation
        particles[i].vy = magnitude*particles[i].x/r;
        particles[i].vz = 0.0f;
    }

    std::unique_ptr<ISimulation> sim;

    if (streategy == "cpu") {
        sim = std::make_unique<CPUSimulation>(particles);
    } 
    else if (streategy == "naive") {
        sim = std::make_unique<GPUSimulation>(particles, Strategy::NAIVE);
    } else if (streategy == "shared") {
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
    if (run_mode == "visual_vbo") {
        if (streategy == "cpu") {
            std::cout << "vbo not supported with CPU" << std::endl;
            return 0;
        }
        if (!glfwInit()) return -1;

        GLFWwindow* window = glfwCreateWindow(800, 800, "CUDA N-Body Simulation", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);
        glewInit(); // Init GLEW after context
        
        unsigned int vbo_pos;
        glGenBuffers(1, &vbo_pos);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
        glBufferData(GL_ARRAY_BUFFER, N * sizeof(float) * 3, 0, GL_DYNAMIC_DRAW);

        // 2. Create Color VBO (NEW)
        unsigned int vbo_col;
        glGenBuffers(1, &vbo_col);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
        glBufferData(GL_ARRAY_BUFFER, N * sizeof(float) * 3, 0, GL_DYNAMIC_DRAW);

        if (streategy == "naive") {
            sim = std::make_unique<GPUSimulation>(particles, Strategy::NAIVE, vbo_pos, vbo_col);
        } else if (streategy == "shared") {
            sim = std::make_unique<GPUSimulation>(particles, Strategy::SHARED, vbo_pos, vbo_col);
        } else {
            std::cerr << "Usage: ./nbody [cpu|naive|shared] [N] [STEPS]\n";
            return 1;
        }

        float rotate_x = 0.0f;
        float rotate_y = 0.0f;
        while (!glfwWindowShouldClose(window)){
            sim->step();
            // B. Clear Screen
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) rotate_y += 1.0f;
            if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) rotate_y -= 1.0f;
            // Rotate X-axis (Pitch)
            if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) rotate_x -= 1.0f;
            if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) rotate_x += 1.0f;
        
            // --- 3. RENDERING ---
            glClear(GL_COLOR_BUFFER_BIT);
            glLoadIdentity();
        
            // ORDER MATTERS HERE:
            // A. Apply Rotation first (so you rotate the world)
            glRotatef(rotate_x, 1.0f, 0.0f, 0.0f); // Rotate around X-axis
            glRotatef(rotate_y, 0.0f, 1.0f, 0.0f); // Rotate around Y-axis
            
            // B. Then Scale (Keep your existing zoom)
            glScalef(0.01f, 0.01f, 0.01f); 
        
            // Enable Position Array
            glEnableClientState(GL_VERTEX_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
            glVertexPointer(3, GL_FLOAT, 0, 0);
        
            // Enable Color Array
            glEnableClientState(GL_COLOR_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
            glColorPointer(3, GL_FLOAT, 0, 0);
        
            // Draw
            glDrawArrays(GL_POINTS, 0, N);
        
            // Cleanup
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        glfwTerminate();
    }

    return 0;
}