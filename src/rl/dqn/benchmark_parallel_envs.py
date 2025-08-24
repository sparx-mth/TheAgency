"""
benchmark_parallel_envs_efficientnet.py - Find optimal number of parallel environments for DQN training
Tests different numbers of environments with EfficientNet B0 feature extractor.
"""

import os
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings
import torch
import gc

warnings.filterwarnings('ignore')

# Add your project to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from sensors.camera_sensor import CameraSensor

# Import the EfficientNet feature extractor
from rl.feature_extractors.efficientnet_feature_extractor import (
    SLAMEfficientNetExtractor,
    SLAMLightweightEfficientNetExtractor
)

# CONFIGURATION
MAP_PATH = "/home/user/nadav/TheAgency/resources/planner/maps/house_map_11.txt"
MEMORY_LIMIT_PERCENT = 70  # Stop if memory usage exceeds this
STEPS_PER_TEST = 1000  # Steps to run for each configuration
TEST_CONFIGS = [1, 2, 4, 8, 12, 16, 20, 24]  # Number of environments to test
USE_SUBPROCESS = False  # Set to True to test SubprocVecEnv (usually slower for small envs)
USE_LIGHTWEIGHT = False  # Set to False to use full EfficientNet-B0


def create_env(env_id: int = 0):
    """Create a single environment for testing."""
    def _init():
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            max_steps=2000,
            map_path=MAP_PATH,
            render_mode=None,
            sensor_config={0: sensor},
            discovery_reward=1.0,
            collision_penalty=-0.5,
            step_penalty=0.0,
            completion_bonus=50.0,
        )
        env = MultiDiscreteToDiscreteWrapper(env)
        env.reset(seed=42 + env_id)
        return env
    return _init


def get_memory_usage():
    """Get current memory usage percentage."""
    return psutil.virtual_memory().percent


def get_gpu_memory_usage():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
    return 0


def benchmark_config(n_envs: int, use_subprocess: bool = False) -> Dict:
    """
    Benchmark a specific number of parallel environments with EfficientNet.

    Returns:
        Dict with fps, memory_usage, total_time, and success status
    """
    print(f"\n🔍 Testing {n_envs} environments with EfficientNet-B0...")

    # Check memory before starting
    initial_memory = get_memory_usage()
    if initial_memory > MEMORY_LIMIT_PERCENT:
        print(f"  ⚠️  Memory already at {initial_memory:.1f}% - skipping")
        return {'success': False, 'reason': 'memory_limit_initial'}

    try:
        # Create vectorized environment
        env_fns = [create_env(i) for i in range(n_envs)]

        if use_subprocess and n_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

        # Create DQN model with EfficientNet
        print(f"  Creating model with {'Lightweight' if USE_LIGHTWEIGHT else 'Full'} EfficientNet...")

        if USE_LIGHTWEIGHT:
            feature_extractor_class = SLAMLightweightEfficientNetExtractor
            feature_extractor_kwargs = dict(features_dim=256)
        else:
            feature_extractor_class = SLAMEfficientNetExtractor
            feature_extractor_kwargs = dict(
                features_dim=256,
                efficientnet_variant='b0',
                pretrained=True,
                freeze_backbone=False  # Allow fine-tuning
            )

        model = DQN(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=dict(
                features_extractor_class=feature_extractor_class,
                features_extractor_kwargs=feature_extractor_kwargs,
                net_arch=[512, 512],
            ),
            learning_rate=5e-5,  # Lower LR for EfficientNet
            buffer_size=100_000,  # Smaller buffer for testing
            learning_starts=100,
            batch_size=32,
            device='auto',
            verbose=0
        )

        device_str = str(model.device)
        print(f"  Device: {device_str}")

        # Warm-up
        print(f"  Warming up...")
        obs = vec_env.reset()
        for _ in range(10):
            actions = np.random.randint(0, vec_env.action_space.n, size=n_envs)
            vec_env.step(actions)

        # Benchmark
        print(f"  Running benchmark...")
        start_time = time.time()
        memory_samples = []
        gpu_memory_samples = []

        # Run training steps
        for step in range(STEPS_PER_TEST):
            # Check memory periodically
            if step % 100 == 0:
                current_memory = get_memory_usage()
                memory_samples.append(current_memory)

                if torch.cuda.is_available():
                    gpu_mem = get_gpu_memory_usage()
                    gpu_memory_samples.append(gpu_mem)

                if current_memory > MEMORY_LIMIT_PERCENT:
                    print(f"  ⚠️  Memory limit exceeded: {current_memory:.1f}%")
                    vec_env.close()
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return {'success': False, 'reason': 'memory_limit_exceeded'}

            # Random actions for consistent testing
            actions = np.random.randint(0, vec_env.action_space.n, size=n_envs)
            vec_env.step(actions)

        # Calculate metrics
        total_time = time.time() - start_time
        fps = (STEPS_PER_TEST * n_envs) / total_time
        avg_memory = np.mean(memory_samples)
        max_memory = np.max(memory_samples)

        avg_gpu_memory = np.mean(gpu_memory_samples) if gpu_memory_samples else 0
        max_gpu_memory = np.max(gpu_memory_samples) if gpu_memory_samples else 0

        # Cleanup
        vec_env.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  ✓ FPS: {fps:.0f} | RAM: {avg_memory:.1f}% (max: {max_memory:.1f}%)", end="")
        if torch.cuda.is_available():
            print(f" | GPU: {avg_gpu_memory:.1f}% (max: {max_gpu_memory:.1f}%)")
        else:
            print()

        return {
            'success': True,
            'n_envs': n_envs,
            'fps': fps,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'avg_gpu_memory': avg_gpu_memory,
            'max_gpu_memory': max_gpu_memory,
            'total_time': total_time,
            'steps_per_env': STEPS_PER_TEST
        }

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        # Cleanup on error
        try:
            vec_env.close()
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {'success': False, 'reason': str(e)}


def plot_results(results: List[Dict]):
    """Create visualization of benchmark results."""
    # Filter successful runs
    successful = [r for r in results if r['success']]
    if not successful:
        print("No successful benchmarks to plot!")
        return

    n_envs = [r['n_envs'] for r in successful]
    fps = [r['fps'] for r in successful]
    avg_memory = [r['avg_memory'] for r in successful]
    max_memory = [r['max_memory'] for r in successful]
    avg_gpu_memory = [r.get('avg_gpu_memory', 0) for r in successful]

    # Calculate efficiency (FPS per environment)
    efficiency = [r['fps'] / r['n_envs'] for r in successful]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EfficientNet-B0 Parallel Environment Performance', fontsize=16, fontweight='bold')

    # 1. FPS vs Number of Environments
    ax1 = axes[0, 0]
    ax1.plot(n_envs, fps, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Parallel Environments')
    ax1.set_ylabel('Total FPS (steps/second)', color='b')
    ax1.set_title('Processing Speed')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')

    # 2. Memory Usage
    ax2 = axes[0, 1]
    ax2.plot(n_envs, avg_memory, 'g-s', label='Avg RAM', linewidth=2, markersize=8)
    ax2.plot(n_envs, max_memory, 'r-^', label='Max RAM', linewidth=2, markersize=8)
    if any(avg_gpu_memory):
        ax2.plot(n_envs, avg_gpu_memory, 'orange', marker='D', label='Avg GPU', linewidth=2, markersize=8)
    ax2.axhline(y=MEMORY_LIMIT_PERCENT, color='r', linestyle='--', alpha=0.5, label='RAM Limit')
    ax2.set_xlabel('Number of Parallel Environments')
    ax2.set_ylabel('Memory Usage (%)')
    ax2.set_title('Memory Consumption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Efficiency (FPS per environment)
    ax3 = axes[1, 0]
    ax3.plot(n_envs, efficiency, 'purple', marker='D', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Parallel Environments')
    ax3.set_ylabel('FPS per Environment')
    ax3.set_title('Parallel Efficiency')
    ax3.grid(True, alpha=0.3)

    # 4. Combined metric (FPS vs Memory)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(avg_memory, fps, c=n_envs, s=100, cmap='viridis', edgecolors='black', linewidth=1)

    # Annotate points with n_envs
    for i, txt in enumerate(n_envs):
        ax4.annotate(str(txt), (avg_memory[i], fps[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax4.set_xlabel('Average RAM Usage (%)')
    ax4.set_ylabel('Total FPS')
    ax4.set_title('Speed vs Memory Trade-off')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('# Environments')

    plt.tight_layout()

    # Save figure
    filename = 'efficientnet_benchmark.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to '{filename}'")
    plt.show()


def find_optimal_config(results: List[Dict]) -> Dict:
    """Find the optimal configuration based on FPS and memory constraints."""
    successful = [r for r in results if r['success']]
    if not successful:
        return None

    # Find config with highest FPS that stays under 60% memory
    safe_configs = [r for r in successful if r['max_memory'] < 60]  # 10% safety margin

    if safe_configs:
        optimal = max(safe_configs, key=lambda x: x['fps'])
    else:
        # If none are safe, get the one with lowest memory
        optimal = min(successful, key=lambda x: x['max_memory'])

    return optimal


def main():
    """Run the benchmark suite."""
    print("="*60)
    print("🚀 EFFICIENTNET-B0 PARALLEL ENVIRONMENT BENCHMARK")
    print("="*60)
    print(f"Feature Extractor: {'Lightweight' if USE_LIGHTWEIGHT else 'Full'} EfficientNet-B0")
    print(f"Map: {MAP_PATH}")
    print(f"Memory limit: {MEMORY_LIMIT_PERCENT}%")
    print(f"Steps per test: {STEPS_PER_TEST}")
    print(f"Configurations to test: {TEST_CONFIGS}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available (using CPU)")

    print("="*60)

    # Check if map exists
    if not os.path.exists(MAP_PATH):
        print(f"❌ Map file not found: {MAP_PATH}")
        return

    # Run benchmarks
    results = []
    for n_envs in TEST_CONFIGS:
        result = benchmark_config(n_envs, use_subprocess=USE_SUBPROCESS)
        results.append(result)

        # Stop if memory limit was exceeded
        if not result['success'] and result.get('reason') == 'memory_limit_exceeded':
            print(f"\n⚠️  Stopping benchmark - memory limit reached at {n_envs} environments")
            break

    # Plot results
    plot_results(results)

    # Find optimal configuration
    optimal = find_optimal_config(results)

    # Print summary
    print("\n" + "="*60)
    print("📈 BENCHMARK RESULTS SUMMARY")
    print("="*60)

    successful = [r for r in results if r['success']]
    if successful:
        print("\n📊 Performance Summary:")
        header = f"{'Config':<10} {'FPS':<10} {'FPS/Env':<10} {'Avg RAM %':<12} {'Max RAM %':<12}"
        if torch.cuda.is_available():
            header += f"{'Avg GPU %':<12}"
        print(header)
        print("-"*70)

        for r in successful:
            fps_per_env = r['fps'] / r['n_envs']
            row = f"{r['n_envs']:<10} {r['fps']:<10.0f} {fps_per_env:<10.1f} "
            row += f"{r['avg_memory']:<12.1f} {r['max_memory']:<12.1f}"
            if torch.cuda.is_available():
                row += f"{r.get('avg_gpu_memory', 0):<12.1f}"
            print(row)

    if optimal:
        print("\n" + "="*60)
        print("🎯 OPTIMAL CONFIGURATION FOR EFFICIENTNET-B0")
        print("="*60)
        print(f"\n✅ RECOMMENDED: Use {optimal['n_envs']} parallel environments")
        print(f"   - FPS: {optimal['fps']:.0f} steps/second")
        print(f"   - RAM: {optimal['avg_memory']:.1f}% average, {optimal['max_memory']:.1f}% peak")
        if torch.cuda.is_available() and optimal.get('avg_gpu_memory'):
            print(f"   - GPU: {optimal['avg_gpu_memory']:.1f}% average")
        print(f"   - Efficiency: {optimal['fps']/optimal['n_envs']:.1f} FPS per environment")

        # Estimate training time
        steps_per_stage = 10_000_000
        hours = steps_per_stage / optimal['fps'] / 3600
        print(f"\n📈 Training time estimate per 10M steps: {hours:.1f} hours")

        # Warning about EfficientNet performance
        print("\n⚠️  Note: EfficientNet-B0 is more computationally intensive than simple CNNs.")
        print("    Consider using the lightweight version or fewer environments if memory is an issue.")
    else:
        print("\n❌ Could not determine optimal configuration")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()