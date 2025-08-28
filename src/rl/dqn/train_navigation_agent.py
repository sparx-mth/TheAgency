"""
train_navigation_optimized.py - Optimized DQN training for Navigation task
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.tasks.navigation_wrapper import NavigationWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

# TRAINING SETTINGS
N_ENVS = 8
TOTAL_TIMESTEPS = 10_000_000


class NavigationCallback(BaseCallback):
    """Callback for monitoring navigation training."""

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
        self.distances_to_goal = []
        self.initial_distances = []

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"Navigation training started with {self.n_envs} environments")
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

                    # Track navigation-specific metrics
                    if 'distance_to_goal' in info:
                        self.distances_to_goal.append(info['distance_to_goal'])

                # Print every 100 episodes
                if self.episode_count % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    recent_successes = self.task_successes[-100:]
                    recent_distances = self.distances_to_goal[-100:] if self.distances_to_goal else []

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_final_dist = np.mean([d for d in recent_distances if d > 0]) if recent_distances else 0

                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"Final Dist: {avg_final_dist:4.1f} | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

        return True


def create_env(goal_selection: str, env_id: int = 0):
    """Create single navigation environment."""

    def _init():
        # Load map dimensions
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor
        sensor = CameraSensor(
            max_range=8,
            fov_deg=60,
            num_rays=24
        )

        # Environment configuration
        env_config = {
            'width': actual_width,
            'height': actual_height,
            'num_agents': 1,
            'max_steps': 2000,
            'map_path': MAP_PATH,
            'render_mode': None,
            'sensor_config': {0: sensor},
            # Base rewards
            'discovery_reward': 0.5,
            'collision_penalty': -1.0,
            'step_penalty': 0.0,
            'completion_bonus': 0.0,
        }

        # Create navigation wrapper
        env = NavigationWrapper(
            env_config=env_config,
            # Exploration
            exploration_steps=20,
            # Task parameters
            max_steps_to_goal=200,
            goal_selection=goal_selection,
            # Rewards
            goal_reached_reward=200.0,
            closer_reward_scale=1.0,
            farther_penalty_scale=0.5,
            time_penalty=0.01,
            collision_penalty=-1.0,
        )

        # Add monitor
        env = Monitor(env)

        # Set seed
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN for navigation task with curriculum."""

    # Verify map
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("=" * 60)
    print("DQN NAVIGATION TRAINING - OPTIMIZED")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")

    # Training stages with curriculum
    CURRICULUM = [
        ("Stage 1: Random Goals", "random", 3_000_000),
        ("Stage 2: Challenging Goals", "challenging", 3_000_000),
        ("Stage 3: Farthest Goals", "farthest", 4_000_000),
    ]

    print(f"Training stages: {len(CURRICULUM)}")
    for idx, (name, goal_type, steps) in enumerate(CURRICULUM):
        print(f"  {idx + 1}. {name} ({steps:,} steps)")
    print("=" * 60 + "\n")

    # Estimate time
    estimated_fps = N_ENVS * 450
    estimated_hours = TOTAL_TIMESTEPS / estimated_fps / 3600
    print(f"Estimated total time: {estimated_hours:.1f} hours")
    print(f"({TOTAL_TIMESTEPS / 1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    model = None

    for stage_idx, (stage_name, goal_selection, stage_steps) in enumerate(CURRICULUM):
        stage_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_idx + 1}/{len(CURRICULUM)}: {stage_name}")
        print(f"Goal selection: {goal_selection}")
        print("-" * 40)

        # Create parallel environments
        print(f"Creating {N_ENVS} environments...")
        env_fns = [create_env(goal_selection, i) for i in range(N_ENVS)]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

        if model is None:
            # Create new model
            print("Creating DQN model...")
            model = DQN(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=dict(
                    features_extractor_class=SLAMCNNExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[512, 512],
                ),
                # Hyperparameters
                learning_rate=1e-4,
                buffer_size=1_000_000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10000,
                # Exploration
                exploration_fraction=0.6,
                exploration_initial_eps=1.0 if stage_idx == 0 else 0.5,  # Less exploration in later stages
                exploration_final_eps=0.05,
                # Other
                max_grad_norm=10,
                seed=42,
                device='auto',
            )
            print(f"Model created on {model.device}")
        else:
            # Update environment
            model.set_env(vec_env)
            # Adjust exploration for later stages
            if stage_idx == 1:
                model.exploration_initial_eps = 0.3
            elif stage_idx == 2:
                model.exploration_initial_eps = 0.2
            print("Model environment updated")

        # Create callbacks
        callbacks = []

        # Progress callback
        progress_callback = NavigationCallback(n_envs=N_ENVS)
        callbacks.append(progress_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500_000 // N_ENVS,
            save_path="./models/navigation_checkpoints/",
            name_prefix=f"stage_{stage_idx + 1}_{goal_selection}"
        )
        callbacks.append(checkpoint_callback)

        callback = CallbackList(callbacks)

        # Train
        print(f"\nTraining stage {stage_idx + 1}...")
        print(f"Target: {stage_steps:,} steps")

        model.learn(
            total_timesteps=stage_steps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False
        )

        # Stage complete
        stage_time = time.time() - stage_start
        stage_fps = stage_steps / stage_time

        # Save stage model
        model_path = f"./models/navigation_stage_{stage_idx + 1}_{goal_selection}"
        model.save(model_path)

        print(f"\nStage {stage_idx + 1} complete!")
        print(f"  Time: {stage_time / 60:.1f} minutes")
        print(f"  Average FPS: {stage_fps:.0f}")
        print(f"  Model saved: {model_path}.zip")

        # Estimate remaining
        stages_left = len(CURRICULUM) - stage_idx - 1
        if stages_left > 0:
            remaining_steps = sum(s[2] for s in CURRICULUM[stage_idx + 1:])
            est_remaining = remaining_steps / stage_fps / 60
            print(f"  Est. remaining: {est_remaining:.0f} minutes")

        vec_env.close()

    # Save final model
    model.save("./models/dqn_navigation_optimized_final")

    print("\n" + "=" * 60)
    print("NAVIGATION TRAINING COMPLETE!")
    print(f"Final model: ./models/dqn_navigation_optimized_final.zip")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Verify map
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/navigation_checkpoints", exist_ok=True)

    train()