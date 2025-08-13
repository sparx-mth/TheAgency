"""
test_simple.py - Simple testing script for trained PPO model on house_map_10.txt
Loads the trained model and visualizes performance.
"""

import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor


def create_env(render=True):
    """Create the environment for testing."""

    # Same sensor configuration as training
    sensor = CameraSensor(
        max_range=5,
        fov_deg=90,
        num_rays=20
    )

    env = MultiAgentSLAMEnv(
        width=10,
        height=10,
        num_agents=1,
        max_steps=500,
        map_path="/home/user/nadav/TheAgency/resources/planner/maps/house_map_10.txt",
        randomize=False,
        render_mode='human' if render else None,
        sensor_config={0: sensor},
        discovery_reward=1.0,
        collision_penalty=-0.1,
        step_penalty=0.0,
        completion_bonus=50.0,
    )

    return env


def test_model(model_path="./models/simple/final_model", n_episodes=5):
    """Test the trained model and display results."""

    print("=" * 60)
    print("TESTING PPO MODEL ON HOUSE MAP 10")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\nError: Model not found at {model_path}.zip")
        print("Please train a model first using train_simple.py")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded successfully")

    # Create environment
    print("\nCreating test environment with rendering...")
    env = create_env(render=True)

    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    # Load normalization if it exists
    norm_path = "../rl/models/simple/vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"Loading normalization from {norm_path}...")
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Test episodes
    print(f"\nRunning {n_episodes} test episodes...")
    print("-" * 60)

    episode_rewards = []
    episode_lengths = []
    episode_progress = []
    episode_discoveries = []

    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("-" * 30)

        obs = vec_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = vec_env.step(action)

            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1

            # Print progress every 50 steps
            if steps % 50 == 0:
                current_info = info[0] if isinstance(info, list) else info
                print(f"  Step {steps}: Progress={current_info.get('progress', 0) * 100:.1f}%, "
                      f"Discovered={current_info.get('discovered_cells', 0)} cells")

        # Get final info
        final_info = info[0] if isinstance(info, list) else info
        final_progress = final_info.get('progress', 0) * 100
        final_discovered = final_info.get('discovered_cells', 0)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_progress.append(final_progress)
        episode_discoveries.append(final_discovered)

        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps Taken: {steps}")
        print(f"  Final Progress: {final_progress:.1f}%")
        print(f"  Cells Discovered: {final_discovered}")

        # Check if completed
        if final_progress >= 99:
            print("  ✓ Map fully explored!")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Average Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps:      {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Progress:   {np.mean(episode_progress):.1f}% ± {np.std(episode_progress):.1f}%")
    print(f"Average Discovery:  {np.mean(episode_discoveries):.1f} ± {np.std(episode_discoveries):.1f} cells")

    success_rate = sum(p >= 99 for p in episode_progress) / n_episodes * 100
    print(f"Success Rate:       {success_rate:.0f}% ({sum(p >= 99 for p in episode_progress)}/{n_episodes} episodes)")

    # Performance assessment
    print("\n" + "-" * 60)
    if np.mean(episode_progress) >= 95:
        print("✅ EXCELLENT: Agent successfully explores the map!")
    elif np.mean(episode_progress) >= 80:
        print("⚠️ GOOD: Agent explores most of the map but misses some areas")
    elif np.mean(episode_progress) >= 60:
        print("⚠️ FAIR: Agent needs more training")
    else:
        print("❌ POOR: Agent struggling - consider adjusting hyperparameters")

    vec_env.close()


def test_checkpoint(checkpoint_name):
    """Test a specific checkpoint."""
    checkpoint_path = f"./models/simple/checkpoints/{checkpoint_name}"
    print(f"\nTesting checkpoint: {checkpoint_name}")
    test_model(checkpoint_path, n_episodes=3)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test specific model or checkpoint
        model_path = sys.argv[1]
        n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_model(model_path, n_episodes)
    else:
        # Test final model by default
        test_model("./models/simple/final_model", n_episodes=5)

        # Optionally test checkpoints
        print("\n" + "=" * 60)
        response = input("\nTest checkpoints? (y/n): ")
        if response.lower() == 'y':
            checkpoint_dir = "../rl/models/simple/checkpoints"
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoints:
                    print(f"\nFound {len(checkpoints)} checkpoints")
                    # Test last checkpoint
                    checkpoints.sort()
                    test_checkpoint(checkpoints[-1].replace('.zip', ''))
                else:
                    print("No checkpoints found")