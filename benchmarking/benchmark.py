import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os

# --- CONFIGURATION ---
# Make sure this points to your compiled C++ executable
EXECUTABLE_PATH = "build/nbody_modular" 

# Benchmark Parameters
# IMPORTANT: Use at least 10000 to avoid 0ms execution times!
PARTICLE_COUNT = 1000 
STEPS = 12
RUN_MODE = "benchmark"
STRATEGIES = ["cpu", "naive", "shared"]

def run_benchmark(strategy):
    cmd = [
        EXECUTABLE_PATH,
        str(PARTICLE_COUNT),
        str(STEPS),
        strategy,
        RUN_MODE
    ]
    
    print(f"Running strategy: {strategy.upper()} (N={PARTICLE_COUNT})...")
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        
        output = result.stdout
        
        # --- PARSING THE OUTPUT ---
        # 1. Parse Time
        time_match = re.search(r"Total Time:\s+(\d+)\s+ms", output)
        
        # 2. Parse Throughput (Updated to handle 'inf')
        # We look for digits, dots, OR the word 'inf'
        tput_match = re.search(r"Throughput:\s+([\d\.]+|inf)\s+G-Interactions/sec", output, re.IGNORECASE)

        if time_match and tput_match:
            time_ms = int(time_match.group(1))
            tput_str = tput_match.group(1).lower()

            # Handle the "infinity" edge case
            if "inf" in tput_str:
                print(f"  -> Warning: Run was too fast (0 ms). Throughput is infinite.")
                throughput = 0.0 # Set to 0 so we can still plot without crashing
            else:
                throughput = float(tput_str)

            return {
                "strategy": strategy,
                "time_ms": time_ms,
                "throughput": throughput
            }
        else:
            print(f"Warning: Could not parse output for {strategy}")
            # Print output for debugging if parsing fails
            print(f"Raw Output:\n{output}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {strategy}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Executable '{EXECUTABLE_PATH}' not found.")
        sys.exit(1)

def plot_results(results):
    strategies = [r["strategy"].upper() for r in results]
    times = [r["time_ms"] for r in results]
    throughputs = [r["throughput"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'N-Body Performance (N={PARTICLE_COUNT}, Steps={STEPS})', fontsize=16)

    # Plot 1: Time
    bars1 = ax1.bar(strategies, times, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_title('Total Time (Lower is Better)')
    ax1.set_ylabel('ms')
    ax1.bar_label(bars1, fmt='%d ms', padding=3)

    # Plot 2: Throughput
    bars2 = ax2.bar(strategies, throughputs, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax2.set_title('Throughput (Higher is Better)')
    ax2.set_ylabel('G-Interactions/sec')
    
    # Label bars, handling the 0/inf case
    labels = [f"{v:.3f}" if v > 0 else "inf" for v in throughputs]
    ax2.bar_label(bars2, labels=labels, padding=3)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print(f"\nChart saved as 'benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    if not os.path.isfile(EXECUTABLE_PATH):
        print(f"Error: '{EXECUTABLE_PATH}' not found. Did you compile your C++ code?")
    else:
        results = []
        print("--- STARTING AUTOMATED BENCHMARK ---\n")
        for strat in STRATEGIES:
            data = run_benchmark(strat)
            if data:
                results.append(data)
        
        if len(results) > 0:
            plot_results(results)