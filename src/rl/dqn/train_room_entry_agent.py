"""
train_room_entry_optimized.py - Ultra-optimized DQN training with pre-computed doorways
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.tasks.room_entry_wrapper import RoomEntryWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# Import the pre-computation utility
from environments.tasks.doorway_utils import precompute_doorways  # Adjust import path as needed

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

# OPTIMIZATION SETTINGS
N_ENVS = 8
STEPS_PER_STAGE = 100_000_000

# Pre-compute doorways ONCE at module level for maximum efficiency
print(f"Pre-computing doorways from {MAP_PATH}...")
PRECOMPUTED_DOORWAYS = precompute_doorways(MAP_PATH)
print(f"Found {len(PRECOMPUTED_DOORWAYS)} doorways: {list(PRECOMPUTED_DOORWAYS.keys())[:5]}...")


class FastRoomEntryCallback(BaseCallback):
    """Minimal callback for maximum speed."""

    def __init__(self, n_envs=1):
        super().__init__()
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.task_successes = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"Training started with {self.n_envs} environments")
        print("-" * 60)

    def _on_step(self) -> bool:
        self.total_steps += self.n_envs

        for i in range(self.n_envs):
            if self.locals.get('dones')[i]:
                self.episode_count += 1
                info = self.locals['infos'][i]

                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    self.task_successes.append(info.get('task_success', False))

                if self.episode_count % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    recent_successes = self.task_successes[-100:]

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0

                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

        return True


def create_env(env_id: int = 0):
    """Create a single room entry environment with pre-computed doorways."""

    def _init():
        # Load map dimensions
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor
        sensor = CameraSensor(max_range=2, fov_deg=90, num_rays=12)

        # Environment configuration
        env_config = {
            'width': actual_width,
            'height': actual_height,
            'num_agents': 1,
            'max_steps': 2000,
            'map_path': MAP_PATH,
            'render_mode': None,
            'sensor_config': {0: sensor},
            'discovery_reward': 1.0,
            'collision_penalty': -0.5,
            'step_penalty': -0.01,
            'completion_bonus': 50.0,
        }

        # Create Room Entry wrapper with PRE-COMPUTED doorways
        env = RoomEntryWrapper(
            env_config=env_config,
            precomputed_doorways=PRECOMPUTED_DOORWAYS,
            success_reward=20.0,  # Changed from entry_reward
            progress_reward=1.0,  # Changed from approach_reward
            # Removed wrong_direction_penalty (doesn't exist)
            collision_penalty=-1.0,
            step_penalty=-0.01,
            max_task_steps=500,
            auto_explore=True,
            max_exploration_steps=1000,
            min_doorways_to_discover=1,
            exploration_strategy="frontier",
        )

        # Add monitor
        env = Monitor(env)

        # Set seed
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN for room entry task - OPTIMIZED VERSION."""

    # Verify map
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("="*60)
    print("DQN ROOM ENTRY TRAINING - ULTRA-OPTIMIZED")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Pre-computed doorways: {len(PRECOMPUTED_DOORWAYS)}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Total steps: {STEPS_PER_STAGE:,}")
    print("="*60 + "\n")

    estimated_fps = N_ENVS * 500  # Should be even faster with optimizations
    estimated_hours = STEPS_PER_STAGE / estimated_fps / 3600
    print(f"Estimated training time: {estimated_hours:.1f} hours")
    print(f"  ({STEPS_PER_STAGE/1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    # Create environments
    print(f"Creating {N_ENVS} environments...")
    env_fns = [create_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Create model
    print("Creating DQN model...")
    model = DQN(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=SLAMCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[512, 512],
        ),
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.7,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        seed=42,
        device='auto',
    )

    print(f"Model created on {model.device}\n")

    # Create callbacks
    callbacks = []

    progress_callback = FastRoomEntryCallback(n_envs=N_ENVS)
    callbacks.append(progress_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
        save_path="./models/room_entry_checkpoints/",
        name_prefix="room_entry_optimized"
    )
    callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks)

    # Train
    print("Starting training...")
    print(f"Target: {STEPS_PER_STAGE:,} steps\n")

    start_time = time.time()

    model.learn(
        total_timesteps=STEPS_PER_STAGE,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False
    )

    # Training complete
    training_time = time.time() - start_time
    avg_fps = STEPS_PER_STAGE / training_time

    # Save final model
    model.save("./models/dqn_room_entry_optimized_final")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Average FPS: {avg_fps:.0f}")
    print(f"Model saved to ./models/dqn_room_entry_optimized_final.zip")
    print("="*60)

    vec_env.close()


if __name__ == "__main__":
    # Verify map exists
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/room_entry_checkpoints", exist_ok=True)

    # Start training
    train()