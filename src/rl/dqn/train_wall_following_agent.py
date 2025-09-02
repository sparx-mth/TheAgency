"""
train_wall_following_agent.py - DQN training for Wall Following task
Optimized for maximum training speed
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.tasks.wall_following_wrapper import WallFollowingWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

# OPTIMIZATION SETTINGS
N_ENVS = 4  # Number of parallel environments
TOTAL_TIMESTEPS = 10_000_000  # Total training steps


class WallFollowingCallback(BaseCallback):
    """Callback for monitoring wall following training progress."""

    def __init__(self, n_envs=1):
        super().__init__()
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.task_successes = []
        self.wall_coverages = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"Training started with {self.n_envs} environments")
        print("-" * 60)

    def _on_step(self) -> bool:
        self.total_steps += self.n_envs

        # Check for episode completions
        for i in range(self.n_envs):
            if self.locals.get('dones')[i]:
                self.episode_count += 1
                info = self.locals['infos'][i]

                # Collect metrics
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    self.task_successes.append(info.get('task_success', False))

                    # Get wall coverage if available
                    if 'wall_coverage' in info:
                        self.wall_coverages.append(info['wall_coverage'])

                # Print progress every 50 episodes
                if self.episode_count % 50 == 0:
                    recent_rewards = self.episode_rewards[-50:]
                    recent_lengths = self.episode_lengths[-50:]
                    recent_successes = self.task_successes[-50:]
                    recent_coverages = self.wall_coverages[-50:] if self.wall_coverages else []

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_coverage = np.mean(recent_coverages) * 100 if recent_coverages else 0

                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"Coverage: {avg_coverage:5.1f}% | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

        return True


def create_env(env_id: int = 0):
    """Create a single wall following environment."""

    def _init():
        # Load map dimensions
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor with reduced complexity for speed
        sensor = CameraSensor(
            max_range=2,  # Reduced range for faster computation
            fov_deg=90,
            num_rays=12  # Reduced rays for speed
        )

        # Environment configuration
        env_config = {
            'width': actual_width,
            'height': actual_height,
            'num_agents': 1,
            'max_steps': 1000,
            'map_path': MAP_PATH,
            'render_mode': None,
            'sensor_config': {0: sensor},
            # Base SLAM rewards
            'discovery_reward': 0.5,
            'collision_penalty': -5.0,
            'step_penalty': -0.01,
            'completion_bonus': 0.0,
        }

        # Create Wall Following wrapper
        env = WallFollowingWrapper(env_config=env_config)

        # Add monitor
        env = Monitor(env)

        # Set seed
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN for wall following task."""

    # Verify map
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("=" * 60)
    print("DQN WALL FOLLOWING TRAINING")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Total steps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60 + "\n")

    # Estimate training time
    estimated_fps = N_ENVS * 300  # Conservative estimate with SubprocVecEnv
    estimated_hours = TOTAL_TIMESTEPS / estimated_fps / 3600
    print(f"Estimated training time: {estimated_hours:.1f} hours")
    print(f"({TOTAL_TIMESTEPS / 1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    # Create environments using SubprocVecEnv for true parallelization
    print(f"Creating {N_ENVS} parallel environments...")
    env_fns = [create_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)  # Use SubprocVecEnv for parallelization
    vec_env = VecMonitor(vec_env)

    # Create model with appropriate hyperparameters
    print("Creating DQN model...")
    model = DQN(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=SLAMCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[512, 512],  # Medium-sized network
        ),
        # DQN hyperparameters
        learning_rate=5e-4,  # Slightly higher for faster learning
        buffer_size=500_000,  # Smaller buffer for faster sampling
        learning_starts=1000,
        batch_size=64,  # Larger batch for better GPU utilization
        tau=1.0,
        gamma=0.99,
        train_freq=8,  # Train less frequently for speed
        gradient_steps=1,
        target_update_interval=5000,  # Update target network less frequently
        # Exploration
        exploration_fraction=0.5,  # Explore for 50% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        # Other
        max_grad_norm=10,
        seed=42,
        device='auto',
    )

    print(f"Model created on {model.device}\n")

    # Create callbacks
    callbacks = []

    # Progress callback
    progress_callback = WallFollowingCallback(n_envs=N_ENVS)
    callbacks.append(progress_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // N_ENVS,  # Save every 250k steps
        save_path="./models/wall_following_checkpoints/",
        name_prefix="wall_following"
    )
    callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks)

    # Train
    print("Starting training...")
    print(f"Target: {TOTAL_TIMESTEPS:,} steps\n")

    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False
    )

    # Training complete
    training_time = time.time() - start_time
    avg_fps = TOTAL_TIMESTEPS / training_time

    # Save final model
    model.save("./models/dqn_wall_following_final")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Total time: {training_time / 60:.1f} minutes")
    print(f"Average FPS: {avg_fps:.0f}")
    print(f"Model saved to ./models/dqn_wall_following_final.zip")
    print("=" * 60)

    vec_env.close()


if __name__ == "__main__":
    # Verify map exists
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/wall_following_checkpoints", exist_ok=True)

    # Training mode
    train()
