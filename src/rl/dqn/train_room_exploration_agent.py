"""
train_room_exploration_optimized.py - Optimized DQN training with pre-computed rooms
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.tasks.room_exploration_wrapper import RoomExplorationWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# Import pre-computation utility
from environments.tasks.room_utils import precompute_room_data  # Adjust import path

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

# OPTIMIZATION SETTINGS
N_ENVS = 8  # Increased for better parallelization
STEPS_PER_STAGE = 100_000_000

# Pre-compute room data ONCE at module level
print(f"Pre-computing room data from {MAP_PATH}...")
PRECOMPUTED_ROOMS = precompute_room_data(MAP_PATH)
print(f"Found {len(PRECOMPUTED_ROOMS['doorways'])} doorways and {len(PRECOMPUTED_ROOMS['rooms'])} rooms")


class RoomExplorationCallback(BaseCallback):
    """Optimized callback for monitoring training."""

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
        self.door_failures = []
        self.room_coverages = []

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

                    # Task metrics
                    self.task_successes.append(info.get('task_success', False))
                    self.door_failures.append(info.get('passed_through_door', False))
                    self.room_coverages.append(info.get('room_coverage', 0.0))

                # Print every 100 episodes
                if self.episode_count % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    recent_successes = self.task_successes[-100:]
                    recent_door_fails = self.door_failures[-100:]
                    recent_coverages = self.room_coverages[-100:]

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    door_fail_rate = np.mean(recent_door_fails) * 100 if recent_door_fails else 0
                    avg_coverage = np.mean(recent_coverages) * 100 if recent_coverages else 0

                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"Door Fail: {door_fail_rate:5.1f}% | "
                          f"Coverage: {avg_coverage:5.1f}% | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

        return True


def create_env(coverage_threshold: float, env_id: int = 0):
    """Create single room exploration environment with pre-computed rooms."""

    def _init():
        # Load map dimensions
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor (optimized for speed)
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
            # Base rewards
            'discovery_reward': 1.0,
            'collision_penalty': -0.5,
            'step_penalty': -0.01,
            'completion_bonus': 50.0,
        }

        # Create optimized wrapper with pre-computed rooms
        env = RoomExplorationWrapper(
            env_config=env_config,
            # Pass pre-computed room data
            precomputed_rooms=PRECOMPUTED_ROOMS,
            # Auto-exploration
            auto_explore=True,
            max_exploration_steps=500,
            min_room_discovery=0.3,  # Discover 30% before starting
            exploration_strategy="frontier",
            # Task rewards
            exploration_reward=0.1,
            door_penalty=-10.0,
            completion_reward=10.0,
            step_penalty=-0.001,
            coverage_threshold=coverage_threshold,
            max_task_steps=500,
        )

        # Add monitor
        env = Monitor(env)

        # Set seed
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN for room exploration with curriculum learning."""

    # Verify map
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("=" * 60)
    print("DQN ROOM EXPLORATION TRAINING - OPTIMIZED")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Pre-computed: {len(PRECOMPUTED_ROOMS['doorways'])} doorways, {len(PRECOMPUTED_ROOMS['rooms'])} rooms")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Steps per stage: {STEPS_PER_STAGE:,}")

    # Training stages with curriculum
    CURRICULUM = [
        ("Stage 1: 90% Coverage", 0.9),
        ("Stage 2: 95% Coverage", 0.95),
        ("Stage 3: 100% Coverage", 1.0),
    ]

    print(f"Training stages: {len(CURRICULUM)}")
    for idx, (stage_name, threshold) in enumerate(CURRICULUM):
        print(f"  {idx + 1}. {stage_name}")
    print("=" * 60 + "\n")

    # Estimate time
    estimated_fps = N_ENVS * 450  # Higher with optimizations
    total_steps = len(CURRICULUM) * STEPS_PER_STAGE
    estimated_hours = total_steps / estimated_fps / 3600
    print(f"Estimated total time: {estimated_hours:.1f} hours")
    print(f"({len(CURRICULUM)} stages × {STEPS_PER_STAGE / 1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    model = None

    for stage_idx, (stage_name, coverage_threshold) in enumerate(CURRICULUM):
        stage_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_idx + 1}/{len(CURRICULUM)}: {stage_name}")
        print("-" * 40)

        # Create parallel environments
        print(f"Creating {N_ENVS} environments (coverage={coverage_threshold:.0%})...")
        env_fns = [create_env(coverage_threshold, i) for i in range(N_ENVS)]
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
                exploration_fraction=0.7,
                exploration_initial_eps=1.0,
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
            print("Model environment updated")

        # Create callbacks
        callbacks = []

        # Progress callback
        progress_callback = RoomExplorationCallback(n_envs=N_ENVS)
        callbacks.append(progress_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500_000 // N_ENVS,
            save_path="./models/room_exploration_checkpoints/",
            name_prefix=f"stage_{stage_idx + 1}_coverage_{int(coverage_threshold * 100)}"
        )
        callbacks.append(checkpoint_callback)

        callback = CallbackList(callbacks)

        # Train
        print(f"\nTraining stage {stage_idx + 1}...")
        print(f"Target: {STEPS_PER_STAGE:,} steps")

        model.learn(
            total_timesteps=STEPS_PER_STAGE,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False
        )

        # Stage complete
        stage_time = time.time() - stage_start
        stage_fps = STEPS_PER_STAGE / stage_time

        # Save stage model
        model_path = f"./models/room_exploration_stage_{stage_idx + 1}_coverage_{int(coverage_threshold * 100)}"
        model.save(model_path)

        print(f"\nStage {stage_idx + 1} complete!")
        print(f"  Time: {stage_time / 60:.1f} minutes")
        print(f"  Average FPS: {stage_fps:.0f}")
        print(f"  Model saved: {model_path}.zip")

        # Estimate remaining
        stages_left = len(CURRICULUM) - stage_idx - 1
        if stages_left > 0:
            est_remaining = (stages_left * STEPS_PER_STAGE) / stage_fps / 60
            print(f"  Est. remaining: {est_remaining:.0f} minutes")

        vec_env.close()

    # Save final model
    model.save("./models/dqn_room_exploration_optimized_final")

    print("\n" + "=" * 60)
    print("ROOM EXPLORATION TRAINING COMPLETE!")
    print(f"Final model: ./models/dqn_room_exploration_optimized_final.zip")
    print("=" * 60)


if __name__ == "__main__":
    # Verify map
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/room_exploration_checkpoints", exist_ok=True)

    # Start training
    train()