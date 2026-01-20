#include "CPUSimulation.h"

CPUSimulation::CPUSimulation(std::vector<Particle>& data)
    : particles(data), total_particles(data.size()) {}


std::string CPUSimulation::get_name() {
    return "CPU (Scalar)";
}

void CPUSimulation::write_to_host(std::vector<Particle>& h_particles){
    // does not need to do anything because in constructor particles are passed by reference
}


void CPUSimulation::step(){

    // Loop over all particles to calculate forces
    for (int i = 0; i < total_particles; ++i) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        for (int j = 0; j < total_particles; ++j) {
            if (i == j) continue; // Newton's 3rd Law: You don't attract yourself

            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;

            // r^2 = dx^2 + dy^2 + dz^2 + epsilon^2
            float distSq = dx*dx + dy*dy + dz*dz + SOFTENNING*SOFTENNING;
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
    for (int i = 0; i < total_particles; ++i) {
        particles[i].x += particles[i].vx * DT;
        particles[i].y += particles[i].vy * DT;
        particles[i].z += particles[i].vz * DT;
    }

}