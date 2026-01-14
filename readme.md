### **Project Design Document: CUDA N-Body Gravity Engine**

**Project Name:** `CudaGravity` (or a catchy name like *Orbital*)
**Target Architecture:** NVIDIA Pascal (GTX 1050 Ti) + Linux (Ubuntu)
**Goal:** Demonstrate "HPC-grade" optimization by simulating 50,000+ interactive particles at 60 FPS, providing a benchmark comparison between Scalar CPU and Parallel GPU execution.

---

### **1. Scope & Objectives**

* **What we are building:** A standalone C++ application that simulates Newtonian gravity for N particles.
* **What we are NOT building:** A game. We don't care about "gameplay," "textures," or "sound." We only care about **Physics Accuracy** and **Raw FLOPS (Floating Point Operations Per Second).**

---

### **2. Requirements**

#### **A. Functional Requirements (What it does)**

1. **Simulation:** Calculate gravitational force between *every* pair of particles ( complexity).
* Formula: 
* Integration: Semi-implicit Euler (Velocity updates Position).


2. **Dual Backend:** The engine must be switchable at runtime (or compile time) between:
* **CPU Mode:** Single-threaded (Reference implementation).
* **GPU Mode:** CUDA Kernel (High performance).


3. **Visualization:** Render particles as points in 3D space.
* Must use **OpenGL Interoperability** (Map CUDA memory directly to OpenGL textures without copying back to CPU RAM).


4. **Interactivity:**
* User can rotate/zoom the camera.
* User can toggle simulation Pause/Play.



#### **B. Non-Functional Requirements (How it performs)**

1. **Performance:**
* **CPU:** Support ~5,000 particles at >10 FPS.
* **GPU:** Support ~50,000 particles at >30 FPS.


2. **Accuracy:**
* CPU and GPU positions must match (within floating-point error limits) after 100 frames.


3. **Code Quality:**
* No "Magic Numbers" (Use Constants).
* RAII for Memory Management (Custom Wrappers for `cudaMalloc`).
* Proper Error Handling (Macros to catch `cudaError_t`).



---

### **3. The Architecture Stack**

We will use a **Hybrid Host-Device** architecture.

| Layer | Technology | Purpose |
| --- | --- | --- |
| **Presentation** | OpenGL / GLUT / ImGui | Visuals, Window management, Inputs. |
| **Bridge** | CUDA-GL Interop | Allows the GPU to draw data it just calculated (Zero Copy). |
| **Logic (Host)** | C++17 | Orchestrates the loop, handles input, manages memory. |
| **Compute (Device)** | CUDA C++ | The raw number crunching ( Kernel). |

**Data Flow Diagram:**

1. **Init:** Allocate Memory (Unified Memory `cudaMallocManaged`).
2. **Step:** CPU calls `kernel<<<blocks, threads>>>`.
3. **Compute:** GPU Cores update positions X, Y, Z.
4. **Render:** OpenGL reads X, Y, Z directly from GPU memory (No `memcpy` back to CPU).
5. **Repeat.**

---

### **4. Smart Project Plan (The Sprint)**

We will execute this in **4 Phases** to minimize risk.

#### **Phase 1: The Validator (CPU Baseline)**

* **Goal:** Correct Physics, Bad Performance.
* **Deliverable:** A console app that prints particle positions.
* **Why:** If the particles fly off to infinity, we know it's a math bug, not a GPU bug.
* **Timeline:** Day 1 (Today).

#### **Phase 2: The Port (Naive CUDA)**

* **Goal:** Move the loop to a `.cu` file. Use `cudaMemcpy` to move data back and forth.
* **Deliverable:** Console app that runs 10x faster but transfers memory slowly.
* **Why:** Get the toolchain working (NVCC linking).
* **Timeline:** Day 2.

#### **Phase 3: The Optimizer (Unified & Visuals)**

* **Goal:** Visualize the data. Remove the PCIe bottleneck.
* **Deliverable:** A window showing thousands of stars moving.
* **Tech:** OpenGL + `cudaMallocManaged` (Unified Memory).
* **Timeline:** Day 3-4.

#### **Phase 4: The Senior Engineer (Shared Memory)**

* **Goal:** The "NVIDIA optimization."
* **Tech:** Use `__shared__` memory to cache particle data on the Streaming Multiprocessor (L1 Cache control).
* **Deliverable:** 2x-3x speedup over Phase 3.
* **Timeline:** Day 5.
