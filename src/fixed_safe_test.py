#!/usr/bin/env python3
"""
fixed_safe_test.py - Fixed version with proper SubprocVecEnv usage
"""

import os
import sys
import time
import psutil
import numpy as np
import gc
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.curriculum_wrapper import CurriculumWrapper
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_11.txt"

# SAFETY LIMITS
MAX_MEMORY_PERCENT = 70
SAFETY_MARGIN_GB = 4


def check_system_resources():
    """Check if system has enough resources to continue."""
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)

    available_gb = mem.available / (1024 ** 3)
    used_percent = mem.percent

    print(f"System: CPU {cpu:.1f}%, RAM {used_percent:.1f}% used, {available_gb:.1f}GB free")

    if used_percent > MAX_MEMORY_PERCENT:
        print(f" WARNING: Memory usage too high ({used_percent:.1f}%)")
        return False

    if available_gb < SAFETY_MARGIN_GB:
        print(f" WARNING: Not enough free RAM ({available_gb:.1f}GB < {SAFETY_MARGIN_GB}GB)")
        return False

    return True


def make_env(hidden_size: int, rank: int, seed: int = 0):
    """Create a single environment instance for SubprocVecEnv."""

    def _init():
        # Set random seed for this process
        np.random.seed(seed + rank)

        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)

        env = MultiAgentSLAMEnv(
            width=actual_width,
            height=actual_height,
            num_agents=1,
            max_steps=2000,
            map_path=MAP_PATH,
            render_mode=None,
            sensor_config={0: sensor},
            discovery_reward=1.0,
            collision_penalty=-0.5,
            step_penalty=-0.01,
            completion_bonus=50.0,
        )

        if actual_width == 32 and actual_height == 32:
            env = CurriculumWrapper(env, hidden_size=hidden_size)

        env = MultiDiscreteToDiscreteWrapper(env)
        return env

    return _init


def test_baseline():
    """Test single environment baseline."""
    print("\n" + "=" * 60)
    print("Testing single environment baseline...")
    print("=" * 60)

    env_fn = make_env(8, 0)
    env = env_fn()

    # Test with random actions
    print("Running 1000 steps...")
    start = time.time()

    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.time() - start
    fps = 1000 / elapsed

    env.close()

    print(f"Single environment: {fps:.0f} FPS")
    return fps


def safe_test_parallel(n_envs: int, test_steps: int = 10000):
    """Test parallel environments with proper SubprocVecEnv."""
    print(f"\n{'=' * 60}")
    print(f"Testing {n_envs} parallel environments")
    print(f"{'=' * 60}")

    # Check resources
    if not check_system_resources():
        return None

    try:
        # Create environments
        print(f"Creating {n_envs} parallel environments...")

        # ALWAYS use SubprocVecEnv for true parallelization (except n=1)
        if n_envs == 1:
            env_fn = make_env(8, 0, seed=42)
            vec_env = DummyVecEnv([env_fn])
            print("Using DummyVecEnv (n=1)")
        else:
            # Create environment functions for SubprocVecEnv
            env_fns = [make_env(8, i, seed=42) for i in range(n_envs)]

            # Use SubprocVecEnv WITHOUT n_workers parameter (it doesn't exist)
            # SubprocVecEnv will create n_envs processes automatically
            vec_env = SubprocVecEnv(env_fns, start_method='spawn')
            print(f"Using SubprocVecEnv with {n_envs} processes")

        vec_env = VecMonitor(vec_env)

        # Check memory after environment creation
        if not check_system_resources():
            vec_env.close()
            return None

        # Create model with scaled parameters
        print("Creating DQN model...")

        # Scale hyperparameters
        if n_envs <= 8:
            batch_size = 128
            gradient_steps = 2
            buffer_size = 50000
        elif n_envs <= 16:
            batch_size = 256
            gradient_steps = 4
            buffer_size = 100000
        else:
            batch_size = 512
            gradient_steps = 8
            buffer_size = 200000

        model = DQN(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=dict(
                features_extractor_class=SLAMCNNExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[512, 512],
            ),
            learning_rate=5e-4,
            buffer_size=buffer_size,
            learning_starts=min(2000, test_steps // 5),
            batch_size=batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=1,
            gradient_steps=gradient_steps,
            target_update_interval=500,
            verbose=0,
            device='cuda' if n_envs > 8 else 'auto',  # Force CUDA for larger runs
        )

        print(f"Model created on {model.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Gradient steps: {gradient_steps}")

        # Training
        print(f"\nTraining for {test_steps:,} steps...")

        # Monitor performance
        start = time.time()
        cpu_samples = []
        mem_samples = []

        # Train in chunks with monitoring
        chunk_size = 2000
        steps_done = 0

        while steps_done < test_steps:
            # Check resources
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            cpu_samples.append(cpu)
            mem_samples.append(mem)

            # Train chunk
            model.learn(
                total_timesteps=min(chunk_size, test_steps - steps_done),
                reset_num_timesteps=False,
                progress_bar=False
            )

            steps_done += chunk_size

            # Progress
            elapsed = time.time() - start
            fps = steps_done / elapsed if elapsed > 0 else 0
            print(f"  Steps: {steps_done:,}/{test_steps:,} | FPS: {fps:.0f} | CPU: {cpu:.1f}% | RAM: {mem:.1f}%",
                  end='\r')

            # Safety check
            if mem > MAX_MEMORY_PERCENT:
                print(f"\n  Stopping - memory limit reached")
                break

        print()  # New line

        # Calculate results
        total_time = time.time() - start
        actual_fps = steps_done / total_time
        avg_cpu = np.mean(cpu_samples)
        avg_mem = np.mean(mem_samples)
        max_cpu = np.max(cpu_samples)
        max_mem = np.max(mem_samples)

        # Cleanup
        print("Cleaning up...")
        vec_env.close()
        del model
        gc.collect()

        # Results
        print(f"\n{'=' * 60}")
        print(f"RESULTS for {n_envs} environments:")
        print(f"{'=' * 60}")
        print(f"FPS: {actual_fps:.0f}")
        print(f"CPU: {avg_cpu:.1f}% avg, {max_cpu:.1f}% max")
        print(f"Memory: {avg_mem:.1f}% avg, {max_mem:.1f}% max")

        return {
            'n_envs': n_envs,
            'fps': actual_fps,
            'cpu_avg': avg_cpu,
            'cpu_max': max_cpu,
            'mem_avg': avg_mem,
            'mem_max': max_mem
        }

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        try:
            vec_env.close()
        except:
            pass
        gc.collect()
        return None


def main():
    """Main testing function."""

    if not os.path.exists(MAP_PATH):
        print(f"ERROR: Map not found: {MAP_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("PARALLEL ENVIRONMENT TESTING (FIXED)")
    print("=" * 60)

    # Get baseline
    baseline_fps = test_baseline()

    # Test sequence - conservative due to previous crash
    test_counts = [1, 2, 4, 8, 16]

    print(f"\nTest sequence: {test_counts}")
    print("Using proper SubprocVecEnv for parallelization")
    print("=" * 60)

    results = []

    for n_envs in test_counts:
        # Extra safety check
        mem = psutil.virtual_memory()
        if mem.percent > 50:
            print(f"\n  Memory usage already at {mem.percent:.1f}%, stopping tests")
            break

        result = safe_test_parallel(n_envs, test_steps=10000)

        if result is None:
            print(f"Failed to test {n_envs} environments")
            if n_envs > 8:
                print("Stopping here for safety")
                break
        else:
            results.append(result)

            # Stop if resources are getting tight
            if result['mem_max'] > 60 or result['cpu_max'] > 80:
                print(f"\n  Resource usage high, stopping tests")
                break

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\nBaseline (single env): {baseline_fps:.0f} FPS\n")

        print(f"{'Envs':<6} {'FPS':<8} {'Speedup':<10} {'CPU Avg':<10} {'Mem Avg':<10}")
        print("-" * 50)

        for r in results:
            speedup = r['fps'] / baseline_fps
            print(f"{r['n_envs']:<6} {r['fps']:<8.0f} {speedup:<10.2f}x {r['cpu_avg']:<10.1f}% {r['mem_avg']:<10.1f}%")

        # Find best
        best = max(results, key=lambda x: x['fps'])

        print(f"\n Best configuration: {best['n_envs']} environments")
        print(f"   FPS: {best['fps']:.0f}")
        print(f"   Speedup: {best['fps'] / baseline_fps:.1f}x")
        print(f"   1M steps: {1000000 / best['fps'] / 60:.1f} minutes")
        print(f"   9M steps: {9000000 / best['fps'] / 60:.1f} minutes")

        # Save config
        config = f"""# Optimal configuration (fixed)
OPTIMAL_CONFIG = {{
    'n_envs': {best['n_envs']},
    'buffer_size': {100000 if best['n_envs'] <= 16 else 200000},
    'batch_size': {256 if best['n_envs'] <= 16 else 512},
    'learning_starts': {2000 * max(1, best['n_envs'] // 4)},
    'learning_rate': 5e-4,
    'train_freq': 1,
    'gradient_steps': {min(8, best['n_envs'] // 2)},
    'target_update_interval': {max(100, 2000 // best['n_envs'])},
}}

SYSTEM_INFO = {{
    'cpu_cores_physical': 24,
    'cpu_cores_logical': 32,
    'ram_gb': 30.94,
}}
"""

        with open('optimal_config.py', 'w') as f:
            f.write(config)

        print(f"\n Configuration saved to optimal_config.py")


if __name__ == "__main__":
    main()