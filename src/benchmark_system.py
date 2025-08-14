#!/usr/bin/env python3
"""
benchmark_extreme.py - Extended benchmark to test 50-200+ parallel environments
For high-end systems with many CPU cores that show low utilization at 32 envs.
"""

import os
import time
import psutil
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import subprocess
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def get_system_info():
    """Get detailed system information."""
    print("=" * 60)
    print("EXTENDED SYSTEM BENCHMARK")
    print("=" * 60)

    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    print(f"CPU Cores (Physical): {cpu_count}")
    print(f"CPU Cores (Logical): {cpu_count_logical}")
    print(f"CPU Frequency: {cpu_freq.current:.2f} MHz (Max: {cpu_freq.max:.2f} MHz)")

    try:
        cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True).decode().strip()
        print(f"CPU Model: {cpu_info.split(':')[1].strip()}")
    except:
        pass

    mem = psutil.virtual_memory()
    print(f"RAM Total: {mem.total / (1024 ** 3):.2f} GB")
    print(f"RAM Available: {mem.available / (1024 ** 3):.2f} GB")

    try:
        nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                                             shell=True).decode().strip()
        print(f"GPU: {nvidia_smi}")
    except:
        print("GPU: Not detected")

    print("=" * 60)
    return cpu_count, cpu_count_logical, mem.available / (1024 ** 3)


def simulate_env_step(env_id, steps=100):
    """Simulate environment stepping with more realistic workload."""
    np.random.seed(env_id)
    total_time = 0

    # Pre-generate weight matrices (32*32=1024 input dims)
    w1 = np.random.randn(1024, 256).astype(np.float32)
    w2 = np.random.randn(256, 4).astype(np.float32)

    for _ in range(steps):
        start = time.perf_counter()

        # Simulate observation processing (32x32 map + metadata)
        obs_map = np.random.randn(32, 32).astype(np.float32)
        obs_positions = np.random.randn(1, 2).astype(np.float32)

        # Simulate neural network forward pass (simplified)
        # obs_map.flatten() is shape (1024,), w1 is (1024, 256)
        hidden = np.tanh(np.dot(obs_map.flatten(), w1))  # Results in (256,)
        output = np.dot(hidden, w2)  # Results in (4,)
        action = np.argmax(output)

        # Simulate environment step
        new_obs = obs_map * 0.99 + np.random.randn(32, 32).astype(np.float32) * 0.01
        reward = np.sum(new_obs) * 0.001

        # Small delay for environment mechanics
        time.sleep(0.0001)

        total_time += time.perf_counter() - start

    return total_time, steps


def benchmark_extended_parallel():
    """Benchmark with extended range of parallel environments."""
    print("\nTesting extended range of parallel environments...")
    print("This will take a few minutes...\n")

    cpu_cores = mp.cpu_count()

    # Extended test range - go up to 200 or 3x logical cores
    test_counts = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256]
    max_test = min(256, cpu_cores * 8)  # Test up to 8x logical cores or 256
    test_counts = [n for n in test_counts if n <= max_test]

    results = {}

    for n_envs in test_counts:
        print(f"Testing {n_envs:3d} parallel environments...", end='', flush=True)

        # Monitor initial state
        psutil.cpu_percent(interval=None)  # Reset
        initial_mem = psutil.virtual_memory().percent

        # Run test
        start = time.time()
        with ProcessPoolExecutor(max_workers=min(n_envs, cpu_cores * 2)) as executor:
            futures = [executor.submit(simulate_env_step, i, 50) for i in range(n_envs)]
            total_steps = sum(f.result()[1] for f in futures)
        elapsed = time.time() - start

        # Get performance metrics
        steps_per_sec = total_steps / elapsed

        # Monitor resource usage (sample multiple times)
        cpu_samples = []
        mem_samples = []
        for _ in range(3):
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            mem_samples.append(psutil.virtual_memory().percent)

        cpu_usage = np.mean(cpu_samples)
        mem_usage = np.mean(mem_samples)

        # Calculate efficiency
        if 1 in results:
            speedup = steps_per_sec / results[1]['steps_per_sec']
            efficiency = (speedup / n_envs) * 100
        else:
            speedup = 1.0
            efficiency = 100.0

        results[n_envs] = {
            'steps_per_sec': steps_per_sec,
            'time': elapsed,
            'cpu_usage': cpu_usage,
            'mem_usage': mem_usage,
            'speedup': speedup,
            'efficiency': efficiency
        }

        print(f" | {steps_per_sec:6.0f} steps/s | "
              f"CPU: {cpu_usage:5.1f}% | "
              f"RAM: {mem_usage:5.1f}% | "
              f"Speedup: {speedup:5.1f}x | "
              f"Efficiency: {efficiency:5.1f}%")

        # Stop if we're hitting resource limits
        if cpu_usage > 90 or mem_usage > 85:
            print(f"\n⚠️  Stopping test - approaching resource limits")
            break

    return results


def analyze_results(results: Dict, cpu_cores: int):
    """Analyze and visualize the benchmark results."""
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    # Find optimal configurations
    env_counts = sorted(results.keys())

    # Best raw performance
    best_perf = max(results.keys(), key=lambda k: results[k]['steps_per_sec'])
    print(f"\n🚀 Maximum Performance: {best_perf} envs")
    print(f"   Steps/sec: {results[best_perf]['steps_per_sec']:,.0f}")
    print(f"   Speedup: {results[best_perf]['speedup']:.1f}x")
    print(f"   CPU: {results[best_perf]['cpu_usage']:.1f}%")
    print(f"   RAM: {results[best_perf]['mem_usage']:.1f}%")

    # Best efficiency (performance per environment)
    best_efficiency = max(results.keys(),
                          key=lambda k: results[k]['efficiency'] if results[k]['efficiency'] > 50 else 0)
    if best_efficiency != best_perf:
        print(f"\n✅ Best Efficiency: {best_efficiency} envs")
        print(f"   Steps/sec: {results[best_efficiency]['steps_per_sec']:,.0f}")
        print(f"   Efficiency: {results[best_efficiency]['efficiency']:.1f}%")
        print(f"   CPU: {results[best_efficiency]['cpu_usage']:.1f}%")

    # Sweet spot (good performance, <70% CPU, >40% efficiency)
    sweet_spots = [k for k in results.keys()
                   if results[k]['cpu_usage'] < 70
                   and results[k]['efficiency'] > 40
                   and results[k]['mem_usage'] < 70]

    if sweet_spots:
        sweet_spot = max(sweet_spots, key=lambda k: results[k]['steps_per_sec'])
        print(f"\n🎯 Recommended (Balanced): {sweet_spot} envs")
        print(f"   Steps/sec: {results[sweet_spot]['steps_per_sec']:,.0f}")
        print(f"   Speedup: {results[sweet_spot]['speedup']:.1f}x")
        print(f"   Efficiency: {results[sweet_spot]['efficiency']:.1f}%")
        print(f"   CPU: {results[sweet_spot]['cpu_usage']:.1f}%")
        print(f"   RAM: {results[sweet_spot]['mem_usage']:.1f}%")
    else:
        sweet_spot = best_perf

    # Create visualization
    try:
        create_performance_plots(results)
    except Exception as e:
        print(f"\nCouldn't create plots: {e}")

    return sweet_spot


def create_performance_plots(results: Dict):
    """Create performance visualization plots."""
    env_counts = sorted(results.keys())
    steps_per_sec = [results[k]['steps_per_sec'] for k in env_counts]
    cpu_usage = [results[k]['cpu_usage'] for k in env_counts]
    efficiency = [results[k]['efficiency'] for k in env_counts]
    speedup = [results[k]['speedup'] for k in env_counts]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parallel Environment Performance Analysis', fontsize=16)

    # Steps per second
    axes[0, 0].plot(env_counts, steps_per_sec, 'b-o')
    axes[0, 0].set_xlabel('Number of Environments')
    axes[0, 0].set_ylabel('Steps/Second')
    axes[0, 0].set_title('Throughput')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log', base=2)

    # CPU Usage
    axes[0, 1].plot(env_counts, cpu_usage, 'r-o')
    axes[0, 1].axhline(y=80, color='orange', linestyle='--', label='80% threshold')
    axes[0, 1].set_xlabel('Number of Environments')
    axes[0, 1].set_ylabel('CPU Usage (%)')
    axes[0, 1].set_title('CPU Utilization')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log', base=2)
    axes[0, 1].legend()

    # Speedup
    axes[1, 0].plot(env_counts, speedup, 'g-o', label='Actual')
    axes[1, 0].plot(env_counts, env_counts, 'k--', alpha=0.5, label='Ideal')
    axes[1, 0].set_xlabel('Number of Environments')
    axes[1, 0].set_ylabel('Speedup')
    axes[1, 0].set_title('Speedup vs Single Environment')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].legend()

    # Efficiency
    axes[1, 1].plot(env_counts, efficiency, 'm-o')
    axes[1, 1].axhline(y=50, color='orange', linestyle='--', label='50% threshold')
    axes[1, 1].set_xlabel('Number of Environments')
    axes[1, 1].set_ylabel('Efficiency (%)')
    axes[1, 1].set_title('Parallel Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150)
    print("\n📊 Performance plots saved to 'benchmark_results.png'")


def generate_optimal_config(optimal_envs: int, cpu_cores: int, ram_gb: float):
    """Generate optimal configuration for the selected environment count."""

    # Scale hyperparameters based on environment count
    if optimal_envs <= 32:
        buffer_size = min(1000000, int(ram_gb * 0.25 * 1024 * 1024 * 1024 / (32 * 32 * 4 * 2)))
        batch_size = min(256, 32 * max(1, optimal_envs // 4))
        learning_rate = 5e-4
        gradient_steps = min(optimal_envs // 4, 4)
    elif optimal_envs <= 64:
        buffer_size = min(500000, int(ram_gb * 0.20 * 1024 * 1024 * 1024 / (32 * 32 * 4 * 2)))
        batch_size = 256
        learning_rate = 3e-4
        gradient_steps = min(optimal_envs // 8, 8)
    else:
        # For very high environment counts
        buffer_size = min(250000, int(ram_gb * 0.15 * 1024 * 1024 * 1024 / (32 * 32 * 4 * 2)))
        batch_size = 512
        learning_rate = 1e-4
        gradient_steps = min(optimal_envs // 16, 16)

    # Round buffer size
    buffer_size = (buffer_size // 10000) * 10000

    config = {
        'n_envs': optimal_envs,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'learning_starts': min(10000, 1000 * max(1, optimal_envs // 8)),
        'learning_rate': learning_rate,
        'train_freq': 1,  # Always train with many envs
        'gradient_steps': gradient_steps,
        'target_update_interval': max(100, 10000 // optimal_envs),
    }

    print("\n" + "=" * 60)
    print("OPTIMAL CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value:,}")

    # Training time estimates
    steps_per_stage = 1000000
    stages = 9
    fps_estimate = optimal_envs * 80  # Conservative estimate

    print(f"\nEstimated Performance:")
    print(f"  Training FPS: ~{fps_estimate:,}")
    print(f"  Time per 1M steps: ~{steps_per_stage / fps_estimate / 60:.1f} minutes")
    print(f"  Total training time: ~{stages * steps_per_stage / fps_estimate / 3600:.1f} hours")

    return config


def main():
    """Run extended benchmark."""

    # Get system info
    cpu_cores, cpu_logical, ram_gb = get_system_info()

    # Run extended benchmark
    results = benchmark_extended_parallel()

    # Analyze results
    optimal_envs = analyze_results(results, cpu_cores)

    # Generate configuration
    config = generate_optimal_config(optimal_envs, cpu_cores, ram_gb)

    # Save configuration
    config_str = f"""# Optimal training configuration from extended benchmark
# Generated by benchmark_extreme.py

OPTIMAL_CONFIG = {{
    'n_envs': {config['n_envs']},
    'buffer_size': {config['buffer_size']},
    'batch_size': {config['batch_size']},
    'learning_starts': {config['learning_starts']},
    'learning_rate': {config['learning_rate']},
    'train_freq': {config['train_freq']},
    'gradient_steps': {config['gradient_steps']},
    'target_update_interval': {config['target_update_interval']},
}}

# System info
SYSTEM_INFO = {{
    'cpu_cores_physical': {cpu_cores},
    'cpu_cores_logical': {cpu_logical},
    'ram_gb': {ram_gb:.1f},
}}

# Benchmark results summary
BENCHMARK_RESULTS = {{
    'max_tested_envs': {max(results.keys())},
    'optimal_envs': {optimal_envs},
    'max_steps_per_sec': {max(r['steps_per_sec'] for r in results.values()):.0f},
    'optimal_steps_per_sec': {results[optimal_envs]['steps_per_sec']:.0f},
}}
"""

    with open('optimal_config.py', 'w') as f:
        f.write(config_str)

    print("\n✅ Configuration saved to 'optimal_config.py'")
    print("   Run 'python train_dqn_parallel.py' to start training!")

    return config


if __name__ == "__main__":
    main()