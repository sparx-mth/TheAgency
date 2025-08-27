"""
train_dqn_room_exploration.py - DQN training for Room Exploration task
Based on the improved DQN training structure with task-specific adjustments.
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.tasks.room_exploration_wrapper import RoomExplorationWrapper
from sensors.camera_sensor import CameraSensor
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# FIXED MAP PATH - same as SLAM training
MAP_PATH = "/home/user/nadav/TheAgency/resources/planner/maps/house_map_11.txt"

# OPTIMIZATION SETTINGS
N_ENVS = 4  # Use 4 environments for ~2x speedup
STEPS_PER_STAGE = 5_000_000  # 5 million steps per stage


class RoomExplorationProgressCallback(BaseCallback):
    """Optimized callback for room exploration training."""

    def __init__(self, render_freq=2000, n_envs=1):
        super().__init__()
        self.render_freq = render_freq
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.task_successes = []
        self.door_failures = []
        self.timeout_failures = []
        self.doorways_found = []

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.start_time = time.time()
        print(f"🏠 Room Exploration Training started with {self.n_envs} environments")
        print("-" * 60)

    def _on_step(self) -> bool:
        self.total_steps += self.n_envs

        # Check for episode completions in any environment
        for i in range(self.n_envs):
            if self.locals.get('dones')[i]:
                self.episode_count += 1
                info = self.locals['infos'][i]

                # Collect metrics from VecMonitor
                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)

                # Get task-specific metrics
                task_success = info.get('task_success', False)
                task_status = info.get('task_status', 0)

                self.task_successes.append(task_success)

                # Determine failure type if failed
                if not task_success:
                    if task_status == 2:  # FAILURE status
                        # Check if it was a door failure (would have door penalty in rewards)
                        # We'll assume door failure if task failed early (< 100 steps)
                        if length < 100:
                            self.door_failures.append(True)
                            self.timeout_failures.append(False)
                        else:
                            self.door_failures.append(False)
                            self.timeout_failures.append(True)
                    else:
                        self.door_failures.append(False)
                        self.timeout_failures.append(False)
                else:
                    self.door_failures.append(False)
                    self.timeout_failures.append(False)

                # Print progress every 100 episodes
                if self.episode_count % 100 == 0:
                    # Calculate averages
                    recent_rewards = self.episode_rewards[-100:] if len(
                        self.episode_rewards) >= 100 else self.episode_rewards
                    recent_lengths = self.episode_lengths[-100:] if len(
                        self.episode_lengths) >= 100 else self.episode_lengths
                    recent_successes = self.task_successes[-100:] if len(
                        self.task_successes) >= 100 else self.task_successes
                    recent_door_fails = self.door_failures[-100:] if len(
                        self.door_failures) >= 100 else self.door_failures
                    recent_timeout_fails = self.timeout_failures[-100:] if len(
                        self.timeout_failures) >= 100 else self.timeout_failures

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    door_fail_rate = np.mean(recent_door_fails) * 100 if recent_door_fails else 0
                    timeout_rate = np.mean(recent_timeout_fails) * 100 if recent_timeout_fails else 0

                    # Calculate FPS
                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0

                    # Get exploration rate
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"Door Fail: {door_fail_rate:5.1f}% | "
                          f"Timeout: {timeout_rate:5.1f}% | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

                # Render episode for visualization
                if self.episode_count % self.render_freq == 0:
                    self.render_episode()

        return True

    def render_episode(self):
        """Render one episode to see agent performance."""
        print(f"\n>>> Rendering episode {self.episode_count}...")

        # Create test environment with rendering
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)

        env_config = {
            'width': 32,
            'height': 32,
            'num_agents': 1,
            'max_steps': 2000,
            'map_path': MAP_PATH,
            'render_mode': 'human',
            'sensor_config': {0: sensor},
            'discovery_reward': 1.0,
            'collision_penalty': -0.5,
            'step_penalty': 0.0,
            'completion_bonus': 50.0,
        }

        test_env = RoomExplorationWrapper(
            env_config=env_config,
            exploration_reward=0.1,
            door_penalty=-10.0,
            completion_reward=10.0,
            step_penalty=-0.001,
            coverage_threshold=1.0,
            max_task_steps=500,
        )
        # No MultiDiscreteToDiscreteWrapper needed - RoomExplorationWrapper already handles this

        # Run episode
        obs, _ = test_env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 500:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            test_env.render()
            time.sleep(0.02)

        # Show final state
        task_success = info.get('task_success', False)
        test_env.render()
        time.sleep(0.5)
        test_env.close()

        success_str = "SUCCESS" if task_success else "FAILED"
        print(f">>> Rendering complete: {steps} steps, reward: {total_reward:.1f}, result: {success_str}")


def create_env(coverage_threshold: float, env_id: int = 0):
    """Create a single room exploration environment for vectorization."""

    def _init():
        # Load map
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)

        # Environment configuration matching SLAM training
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
            'step_penalty': 0.0,
            'completion_bonus': 50.0,
        }

        # Create Room Exploration wrapper
        env = RoomExplorationWrapper(
            env_config=env_config,
            # Task-specific rewards
            exploration_reward=0.1,
            door_penalty=-10.0,
            completion_reward=10.0,
            step_penalty=-0.001,
            coverage_threshold=coverage_threshold,
            max_task_steps=500,
        )

        # No MultiDiscreteToDiscreteWrapper needed - wrapper already handles this

        # Add monitor for episode statistics
        env = Monitor(env)

        # Set seed for reproducibility
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN for room exploration task with curriculum learning."""

    # Load map once to verify
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("=" * 60)
    print("🏠 DQN ROOM EXPLORATION TRAINING")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Steps per stage: {STEPS_PER_STAGE:,}")

    # Training stages with coverage threshold curriculum
    CURRICULUM = [
        ("Stage 1: 90% Coverage Required", 0.9),  # Start with easier goal
        ("Stage 2: 95% Coverage Required", 0.95),  # Intermediate
        ("Stage 3: 100% Coverage Required", 1.0),  # Full coverage
    ]

    print(f"Training stages: {len(CURRICULUM)}")
    for idx, (stage_name, threshold) in enumerate(CURRICULUM):
        print(f"  {idx + 1}. {stage_name}")

    print("=" * 60 + "\n")

    # Estimate training time
    estimated_fps = N_ENVS * 400  # Conservative estimate
    total_steps = len(CURRICULUM) * STEPS_PER_STAGE
    estimated_hours = total_steps / estimated_fps / 3600
    print(f"📊 Estimated total training time: {estimated_hours:.1f} hours")
    print(f"   ({len(CURRICULUM)} stages × {STEPS_PER_STAGE / 1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    model = None

    for stage_idx, (stage_name, coverage_threshold) in enumerate(CURRICULUM):
        stage_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"🏠 STAGE {stage_idx + 1}/{len(CURRICULUM)}: {stage_name}")
        print("-" * 40)

        # Create vectorized environment
        print(f"Creating {N_ENVS} environments with coverage_threshold={coverage_threshold:.0%}...")

        env_fns = [create_env(coverage_threshold, i) for i in range(N_ENVS)]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)  # Add monitoring wrapper

        if model is None:
            # Create new model with same hyperparameters as SLAM training
            print("Creating new DQN model...")

            model = DQN(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=dict(
                    features_extractor_class=SLAMCNNExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[512, 512, 512, 512],
                ),
                # DQN PARAMETERS (same as SLAM training)
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

            print(f"✓ Model created on {model.device}")
        else:
            # Reuse existing model with new environment
            model.set_env(vec_env)
            print("✓ Model environment updated")

        # Create callbacks
        callbacks = []

        # Progress callback
        progress_callback = RoomExplorationProgressCallback(
            render_freq=2000,
            n_envs=N_ENVS
        )
        callbacks.append(progress_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500000 // N_ENVS,  # Save every 500k steps
            save_path="./models/room_exploration_checkpoints/",
            name_prefix=f"stage_{stage_idx + 1}_coverage_{int(coverage_threshold * 100)}"
        )
        callbacks.append(checkpoint_callback)

        # Combine callbacks
        callback = CallbackList(callbacks)

        # Train
        print(f"\nStarting training for stage {stage_idx + 1}...")
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

        print(f"\n✓ Stage {stage_idx + 1} complete!")
        print(f"   Time: {stage_time / 60:.1f} minutes")
        print(f"   Average FPS: {stage_fps:.0f}")
        print(f"   Model saved to: {model_path}.zip")

        # Estimate remaining time
        stages_left = len(CURRICULUM) - stage_idx - 1
        if stages_left > 0:
            est_remaining = (stages_left * STEPS_PER_STAGE) / stage_fps / 60
            print(f"   Estimated time remaining: {est_remaining:.0f} minutes")

        # Clean up
        vec_env.close()

    # Save final model
    model.save("./models/dqn_room_exploration_final")

    print("\n" + "=" * 60)
    print("🏠 ROOM EXPLORATION TRAINING COMPLETE!")
    print(f"Final model saved to ./models/dqn_room_exploration_final.zip")
    print("=" * 60)


if __name__ == "__main__":
    # Verify map exists
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"✓ Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/room_exploration_checkpoints", exist_ok=True)

    # Start training
    train()