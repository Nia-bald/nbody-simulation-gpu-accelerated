import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os

# --- CONFIGURATION ---
EXECUTABLE_PATH = "build/nbody_modular"

# Sweep Parameters
START_N = 1000
END_N = 20000
STEP_N = 1000
PARTICLE_RANGE = list(range(START_N, END_N + 1, STEP_N))

STEPS = 12
RUN_MODE = "benchmark"
STRATEGIES = ["cpu", "naive", "shared"]

# Colors for plotting
COLORS = {
    "cpu": "red",
    "naive": "blue",
    "shared": "green"
}

def run_benchmark(strategy, n):
    """
    Runs the simulation for a specific strategy and particle count N.
    """
    cmd = [
        EXECUTABLE_PATH,
        str(n),
        str(STEPS),
        strategy,
        RUN_MODE
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        
        output = result.stdout
        
        # Regex Parsing
        time_match = re.search(r"Total Time:\s+(\d+)\s+ms", output)
        tput_match = re.search(r"Throughput:\s+([\d\.]+|inf)\s+G-Interactions/sec", output, re.IGNORECASE)

        if time_match and tput_match:
            time_ms = int(time_match.group(1))
            tput_str = tput_match.group(1).lower()

            if "inf" in tput_str:
                throughput = 0.0
            else:
                throughput = float(tput_str)

            return time_ms, throughput
        else:
            return None, None

    except subprocess.CalledProcessError as e:
        print(f"Error running {strategy} with N={n}: {e}")
        return None, None
    except FileNotFoundError:
        print(f"Error: Executable '{EXECUTABLE_PATH}' not found.")
        sys.exit(1)

def plot_sweep_results(data):
    """
    Plots line graphs for Time vs N and Throughput vs N.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'N-Body Scaling Analysis (Steps={STEPS})', fontsize=16)

    # --- Plot 1: Execution Time vs N ---
    for strategy in STRATEGIES:
        if strategy in data and data[strategy]['n_values']:
            ax1.plot(
                data[strategy]['n_values'], 
                data[strategy]['times'], 
                marker='o', 
                label=strategy.upper(), 
                color=COLORS.get(strategy, 'black')
            )

    ax1.set_title('Execution Time (Lower is Better)')
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Time (ms)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- Plot 2: Throughput vs N ---
    for strategy in STRATEGIES:
        if strategy in data and data[strategy]['n_values']:
            ax2.plot(
                data[strategy]['n_values'], 
                data[strategy]['throughputs'], 
                marker='o', 
                label=strategy.upper(), 
                color=COLORS.get(strategy, 'black')
            )

    ax2.set_title('Throughput (Higher is Better)')
    ax2.set_xlabel('Number of Particles (N)')
    ax2.set_ylabel('G-Interactions/sec')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    output_file = 'benchmark_sweep_results.png'
    plt.savefig(output_file)
    print(f"\n✅ Benchmark Complete! Chart saved as '{output_file}'")
    plt.show()

def main():
    if not os.path.isfile(EXECUTABLE_PATH):
        print(f"Error: '{EXECUTABLE_PATH}' not found. Did you compile your C++ code?")
        return

    # Initialize data structure
    # Format: { 'cpu': {'n_values': [], 'times': [], 'throughputs': []}, ... }
    results_data = {strat: {'n_values': [], 'times': [], 'throughputs': []} for strat in STRATEGIES}

    print(f"--- STARTING BENCHMARK SWEEP (N={START_N} to {END_N}) ---\n")
    
    total_runs = len(PARTICLE_RANGE) * len(STRATEGIES)
    current_run = 0

    for n in PARTICLE_RANGE:
        print(f"Testing N = {n}...")
        for strat in STRATEGIES:
            current_run += 1
            # Progress indicator
            # print(f"  [{current_run}/{total_runs}] Running {strat.upper()}...", end="", flush=True)
            
            time_ms, throughput = run_benchmark(strat, n)
            
            if time_ms is not None:
                results_data[strat]['n_values'].append(n)
                results_data[strat]['times'].append(time_ms)
                results_data[strat]['throughputs'].append(throughput)
                # print(f" Done ({time_ms}ms)")
            else:
                print(f" Failed")

    plot_sweep_results(results_data)

if __name__ == "__main__":
    main()