"""
examples/test_complete_system.py - Updated for new unified state/action space

This file demonstrates the complete SLAM system with all its features:
- Single and multi-agent scenarios
- Different sensor configurations
- Communication abstractions
- Agent strategies
- Gymnasium compatibility
- Stable Baselines3 integration
"""

import sys
import numpy as np
from typing import Dict, Any

# Environment imports - NO LONGER NEED SingleAgentWrapper
from environments.slam_env import MultiAgentSLAMEnv

# Sensor imports
from sensors.camera_sensor import CameraSensor
from sensors.lidar_sensor import LidarSensor

# Communication imports
from communication.local_comm import LocalCommunication

# Agent imports
from agents.random_agent import RandomAgent
from agents.frontier_agent import FrontierAgent


def test_single_agent_frontier():
    """Test single agent with frontier-based exploration."""
    print("=" * 60)
    print("Testing Single Agent with Frontier Strategy")
    print("=" * 60)

    # Create environment with num_agents=1
    env = MultiAgentSLAMEnv(
        width=25,
        height=25,
        num_agents=1,  # Single agent
        max_steps=500,
        sensor_config={0: CameraSensor(max_range=8, fov_deg=60)},
        randomize=True,
        render_mode="human",
        discovery_reward=0.1,
        collision_penalty=-1.0,
    )

    # Create frontier agent
    agent = FrontierAgent(num_agents=1)

    # Run episode
    obs, info = env.reset()
    print(f"Starting position: {obs['positions'][0]}")
    print(f"Sensor type: {info['sensor_types'][0]}")

    total_reward = 0
    done = False
    truncated = False

    while not done and not truncated:
        # Get action from agent (now as array)
        action = agent.get_actions(obs, info)

        # Step environment (single reward)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()

        # Print progress every 50 steps
        if info['step'] % 50 == 0:
            print(f"Step {info['step']}: Progress {info['progress']*100:.1f}%, "
                  f"Reward: {total_reward:.2f}, Collisions: {info['collision_counts'][0]}")

    print(f"\nEpisode Complete!")
    print(f"Total steps: {info['step']}")
    print(f"Final progress: {info['progress']*100:.1f}%")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total collisions: {info['collision_counts'][0]}")

    env.close()


def test_multi_agent_mixed_sensors():
    """Test multi-agent with different sensor types."""
    print("=" * 60)
    print("Testing Multi-Agent with Mixed Sensors")
    print("=" * 60)

    # Configure different sensors for each drone
    sensor_config = {
        0: CameraSensor(max_range=10, fov_deg=45),      # Narrow camera
        1: LidarSensor(max_range=12, num_rays=180),     # Half-resolution LIDAR
        2: CameraSensor(max_range=8, fov_deg=90),       # Wide-angle camera
    }

    # Create environment with mixed sensors
    env = MultiAgentSLAMEnv(
        width=35,
        height=35,
        num_agents=3,
        max_steps=800,
        sensor_config=sensor_config,
        communication=LocalCommunication(),
        randomize=True,
        render_mode="human",
    )

    # Create frontier agent for all drones
    agent = FrontierAgent(num_agents=3)

    # Run episode
    obs, info = env.reset()
    print("Drone sensor types:", info['sensor_types'])

    total_reward = 0.0
    done = False
    truncated = False

    while not done and not truncated:
        # Get actions (array)
        actions = agent.get_actions(obs, info)

        # Step environment (single reward)
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        # Render
        env.render()

        # Print progress
        if info['step'] % 100 == 0:
            print(f"Step {info['step']}: Progress {info['progress']*100:.1f}%, Total Reward: {total_reward:.2f}")
            for i in range(3):
                print(f"  Drone {i} ({info['sensor_types'][i]}): "
                      f"Collisions: {info['collision_counts'][i]}")

    print(f"\nEpisode Complete!")
    print(f"Total steps: {info['step']}")
    print(f"Final progress: {info['progress']*100:.1f}%")
    print(f"Total reward: {total_reward:.2f}")
    for i in range(3):
        print(f"Drone {i}: Collisions: {info['collision_counts'][i]}")

    env.close()


def test_heterogeneous_sensors():
    """Demonstrate heterogeneous sensor capabilities."""
    print("=" * 60)
    print("Testing Heterogeneous Sensor Configuration")
    print("=" * 60)

    # Create different sensor configurations
    configs = [
        ("Camera (Narrow)", CameraSensor(max_range=12, fov_deg=30)),
        ("Camera (Wide)", CameraSensor(max_range=8, fov_deg=120)),
        ("LIDAR (Full)", LidarSensor(max_range=15, num_rays=360)),
        ("LIDAR (Sparse)", LidarSensor(max_range=20, num_rays=36)),
    ]

    for name, sensor in configs:
        print(f"\nTesting {name}:")
        print(f"  Type: {sensor.get_sensor_type()}")
        print(f"  Max Range: {sensor.get_max_range()}")
        print(f"  Parameters: {sensor.get_sensor_params()}")

        # Create single-agent env with this sensor
        env = MultiAgentSLAMEnv(
            width=20,
            height=20,
            num_agents=1,
            max_steps=200,
            sensor_config={0: sensor},
            randomize=False,
            render_mode=None,
        )

        # Run a few steps
        obs, info = env.reset()
        agent = RandomAgent(num_agents=1, forward_bias=0.7)

        discoveries = 0
        for _ in range(50):
            action = agent.get_actions(obs, info)
            obs, reward, done, truncated, info = env.step(action)
            discoveries = info['discovered_cells']
            if done or truncated:
                break

        print(f"  Discovered {discoveries} cells in 50 steps")
        env.close()


def test_gymnasium_compatibility():
    """Test Gymnasium compatibility."""
    print("=" * 60)
    print("Testing Gymnasium Compatibility")
    print("=" * 60)

    # Test environment with single agent
    env = MultiAgentSLAMEnv(
        width=16,
        height=16,
        num_agents=1,
        max_steps=200,
        randomize=False,
    )

    # Check spaces
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation keys: {list(obs.keys())}")
    print(f"Initial info keys: {list(info.keys())}")

    # Test step with random actions
    print("\nTesting random actions:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.3f}, "
              f"terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("✓ Environment is Gymnasium compatible!")


def test_stable_baselines3_training():
    """Test Stable Baselines3 integration."""
    print("=" * 60)
    print("Testing Stable Baselines3 Integration")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        # Create environment with single agent
        env = MultiAgentSLAMEnv(
            width=16,
            height=16,
            num_agents=1,
            max_steps=200,
            randomize=True,
        )

        # Check environment
        print("Running SB3 environment checker...")
        check_env(env)
        print("✓ Environment passes SB3 compatibility check!")

        # Create and train PPO model
        print("\nCreating PPO model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=3,
            verbose=1,
        )
        print("✓ PPO model created successfully!")

        # Train for a few steps
        print("\nTraining for 1000 steps...")
        model.learn(total_timesteps=1000)

        # Test the trained model
        print("\nTesting trained model...")
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Test episode reward: {total_reward:.2f}")
        print("✓ SB3 training and inference working!")

        env.close()

    except ImportError:
        print("Stable Baselines3 not installed.")
        print("Install with: pip install stable-baselines3")


def benchmark_agents():
    """Benchmark different agent strategies."""
    print("=" * 60)
    print("Benchmarking Agent Strategies")
    print("=" * 60)

    agents_to_test = [
        ("Random (low bias)", RandomAgent(1, forward_bias=0.4)),
        ("Random (high bias)", RandomAgent(1, forward_bias=0.8)),
        ("Frontier", FrontierAgent(1)),
    ]

    num_episodes = 3
    env_config = {
        'width': 20,
        'height': 20,
        'num_agents': 1,
        'max_steps': 300,
        'randomize': True,
        'render_mode': None,
    }

    results = {}

    for agent_name, agent in agents_to_test:
        print(f"\nTesting {agent_name} agent...")

        episode_metrics = []

        for episode in range(num_episodes):
            env = MultiAgentSLAMEnv(**env_config)
            obs, info = env.reset()
            agent.reset()

            total_reward = 0
            done = False
            truncated = False

            while not done and not truncated:
                action = agent.get_actions(obs, info)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

            episode_metrics.append({
                'reward': total_reward,
                'progress': info['progress'],
                'steps': info['step'],
                'collisions': info['collision_counts'][0],
            })

            env.close()
            print(f"  Episode {episode+1}: Reward={total_reward:.1f}, "
                  f"Progress={info['progress']*100:.1f}%")

        # Calculate averages
        results[agent_name] = {
            'avg_reward': np.mean([m['reward'] for m in episode_metrics]),
            'avg_progress': np.mean([m['progress'] for m in episode_metrics]),
            'avg_steps': np.mean([m['steps'] for m in episode_metrics]),
            'avg_collisions': np.mean([m['collisions'] for m in episode_metrics]),
        }

    # Print results
    print("\n" + "=" * 60)
    print("Benchmark Results (averaged over {} episodes)".format(num_episodes))
    print("=" * 60)

    for agent_name, metrics in results.items():
        print(f"\n{agent_name} Agent:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Progress: {metrics['avg_progress']*100:.1f}%")
        print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
        print(f"  Avg Collisions: {metrics['avg_collisions']:.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            test_single_agent_frontier()
        elif sys.argv[1] == "multi":
            test_multi_agent_mixed_sensors()
        elif sys.argv[1] == "hetero":
            test_heterogeneous_sensors()
        elif sys.argv[1] == "gym":
            test_gymnasium_compatibility()
        elif sys.argv[1] == "sb3":
            test_stable_baselines3_training()
        elif sys.argv[1] == "benchmark":
            benchmark_agents()
        else:
            print("Unknown option. Use: single, multi, hetero, gym, sb3, or benchmark")
    else:
        # Run core tests
        print("Running core system tests...\n")
        test_gymnasium_compatibility()
        print("\n")
        test_heterogeneous_sensors()
        print("\n")
        benchmark_agents()