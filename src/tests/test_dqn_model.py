"""
test_dqn_model.py - Simple test script to evaluate trained DQN model
Loads weights and runs 10 episodes with visualization and metrics.
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN

from environments.slam_env import MultiAgentSLAMEnv
from environments.curriculum_wrapper import CurriculumWrapper
from environments.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from sensors.camera_sensor import CameraSensor

# Configuration
MODEL_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/improved_stage_2_hidden_10.zip"
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_12.txt"
NUM_EPISODES = 10
HIDDEN_SIZE = 10  # Match the model's training hidden size
RENDER_DELAY = 0.02  # Seconds between frames


def test_model():
    """Test the trained DQN model on multiple episodes."""

    print("=" * 60)
    print(" DQN MODEL TESTING")
    print(f" Model: {MODEL_PATH}")
    print(f" Map: {MAP_PATH}")
    print(f" Episodes: {NUM_EPISODES}")
    print(f" Hidden size: {HIDDEN_SIZE}x{HIDDEN_SIZE}")
    print("=" * 60 + "\n")

    # Load the trained model
    print("Loading model...")
    model = DQN.load(MODEL_PATH)
    print("Model loaded successfully!\n")

    # Create environment with same configuration as training
    sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
    base_env = MultiAgentSLAMEnv(
        width=32,
        height=32,
        num_agents=1,
        max_steps=2000,
        map_path=MAP_PATH,
        render_mode='human',
        sensor_config={0: sensor},
        discovery_reward=1.0,
        collision_penalty=-0.5,
        step_penalty=0.0,
        completion_bonus=50.0,
    )

    # Apply wrappers
    env = CurriculumWrapper(base_env, hidden_size=HIDDEN_SIZE)
    env = MultiDiscreteToDiscreteWrapper(env)

    # Track overall statistics
    all_steps = []
    all_rewards = []
    all_discoveries = []
    all_collisions = []
    all_completions = []

    # Run test episodes
    for episode in range(NUM_EPISODES):
        print(f"\n{'=' * 40}")
        print(f" EPISODE {episode + 1}/{NUM_EPISODES}")
        print("-" * 40)

        # Reset environment
        obs, info = env.reset()
        done = False

        # Episode metrics
        total_steps = 0
        total_reward = 0
        total_collisions = 0

        # Run episode
        time.sleep(RENDER_DELAY*10)
        while not done:
            # Get action from model (deterministic for testing)
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update metrics
            total_steps += 1
            total_reward += reward

            # Check for new collisions
            collision_counts = info.get('collision_counts', [0])
            current_collisions = sum(collision_counts)
            if current_collisions > total_collisions:
                total_collisions = current_collisions

            # Render
            env.render()
            time.sleep(RENDER_DELAY)

            # Stop if taking too long (safety)
            if total_steps >= 100:
                print("  [Episode exceeded 500 steps, stopping...]")
                break

        # Get final metrics
        discovered = info.get('discovered_cells', 0)
        total_reachable = info.get('total_reachable', 0)
        progress = (discovered / total_reachable * 100) if total_reachable > 0 else 0
        completed = discovered >= total_reachable

        # Print episode summary
        print(f"\n  Episode {episode + 1} Summary:")
        print(f"  ├─ Steps taken: {total_steps}")
        print(f"  ├─ Total reward: {total_reward:.1f}")
        print(f"  ├─ Cells discovered: {discovered}/{total_reachable} ({progress:.1f}%)")
        print(f"  ├─ Collisions: {total_collisions}")
        print(f"  └─ Completed: {'✓ YES' if completed else '✗ NO'}")

        # Store metrics
        all_steps.append(total_steps)
        all_rewards.append(total_reward)
        all_discoveries.append(progress)
        all_collisions.append(total_collisions)
        all_completions.append(1 if completed else 0)

        # Brief pause to see final state
        time.sleep(0.5)

    # Close environment
    env.close()

    # Print overall statistics
    print("\n" + "=" * 60)
    print(" OVERALL STATISTICS")
    print("=" * 60)
    print(f" Total episodes: {NUM_EPISODES}")
    print(
        f" Successful completions: {sum(all_completions)}/{NUM_EPISODES} ({sum(all_completions) / NUM_EPISODES * 100:.0f}%)")
    print("-" * 40)
    print(f" Average steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f" Average reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f" Average discovery: {np.mean(all_discoveries):.1f}% ± {np.std(all_discoveries):.1f}%")
    print(f" Average collisions: {np.mean(all_collisions):.1f} ± {np.std(all_collisions):.1f}")
    print("-" * 40)
    print(f" Best episode:")
    best_idx = np.argmax(all_rewards)
    print(f"   Episode {best_idx + 1}: {all_rewards[best_idx]:.1f} reward, {all_steps[best_idx]} steps")
    print(f" Worst episode:")
    worst_idx = np.argmin(all_rewards)
    print(f"   Episode {worst_idx + 1}: {all_rewards[worst_idx]:.1f} reward, {all_steps[worst_idx]} steps")
    print("=" * 60)


if __name__ == "__main__":
    # Verify files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        exit(1)

    if not os.path.exists(MAP_PATH):
        print(f"ERROR: Map not found at {MAP_PATH}")
        exit(1)

    # Run test
    test_model()