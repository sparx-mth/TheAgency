"""
train_dqn_efficientnet.py - DQN training with EfficientNet feature extractor
Enhanced version using more powerful feature extraction with memory fixes.
"""

import os
import time
import numpy as np
import gc
import torch
from collections import deque
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.slam_env import MultiAgentSLAMEnv
from environments.curriculum_wrapper import CurriculumWrapper
from environments.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from sensors.camera_sensor import CameraSensor
import psutil

# Import the new EfficientNet feature extractor
from rl.feature_extractors.efficientnet_feature_extractor import (
    SLAMEfficientNetExtractor,
    SLAMLightweightEfficientNetExtractor
)

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_11.txt"

# OPTIMIZATION SETTINGS
N_ENVS = 8 # Use 4 environments for ~2x speedup without overhead
STEPS_PER_STAGE = 10_000_000  # 10 million steps per stage

# Choose which EfficientNet variant to use
USE_LIGHTWEIGHT = True  # Set to False to use full pretrained EfficientNet


class OptimizedProgressCallback(BaseCallback):
    """Optimized callback for multi-environment training with enhanced success tracking."""

    def __init__(self, render_freq=2000, n_envs=1):
        super().__init__()
        self.render_freq = render_freq
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None
        self.last_gc_step = 0  # Track when we last did garbage collection

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_discoveries = []

        # Enhanced success tracking
        self.success_history = deque(maxlen=100)
        self.recent_completions = 0

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.start_time = time.time()
        print(f"🚀 Training started with {self.n_envs} environments")

        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print("-" * 60)

    def _on_step(self) -> bool:
        self.total_steps += self.n_envs

        # Periodic memory cleanup (every 10000 steps)
        if self.total_steps - self.last_gc_step > 10000:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_gc_step = self.total_steps

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

                # Get exploration metrics
                discovered = info.get('discovered_cells', 0)
                total = info.get('total_reachable', 0)
                progress = info.get('progress', 0) * 100
                collisions = sum(info.get('collision_counts', [0]))
                hidden_size = info.get('hidden_size', 'N/A')

                self.recent_discoveries.append(discovered)

                # Determine if episode was successful
                success = discovered >= total * 0.99 if total > 0 else False
                self.success_history.append(success)

                if success:
                    self.recent_completions += 1

                # Print progress every 100 episodes
                if self.episode_count % 100 == 0:
                    # Calculate averages
                    recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                    recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths
                    recent_disc = self.recent_discoveries[-100:] if len(self.recent_discoveries) >= 100 else self.recent_discoveries

                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    avg_disc = np.mean(recent_disc) if recent_disc else 0

                    # Calculate success metrics
                    success_count = sum(self.success_history)
                    success_rate = (success_count / len(self.success_history)) * 100 if self.success_history else 0

                    # Calculate FPS
                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0

                    # Get exploration rate
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    # Get RAM usage
                    process = psutil.Process(os.getpid())
                    ram_gb = process.memory_info().rss / 1e9  # Resident Set Size (in GB)

                    gpu_mem = ""
                    if torch.cuda.is_available():
                        gpu_gb = torch.cuda.memory_allocated() / 1e9
                        gpu_mem = f" | GPU: {gpu_gb:.2f}GB"

                    ram_mem = f" | RAM: {ram_gb:.2f}GB"

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Disc: {avg_disc:3.0f}/{total:3d} | "
                          f"Success: {success_count:2d}/100 ({success_rate:5.1f}%) | "
                          f"Collision: {collisions:3d} | "
                          f"Hidden: {hidden_size:>3} | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f} |"
                          f"{ram_mem:>3} |"
                          f"{gpu_mem}")


                    # Reset completion counter
                    self.recent_completions = 0

                # Render episode for visualization
                if self.episode_count % self.render_freq == 0:
                    self.render_episode(hidden_size)

        return True

    def render_episode(self, hidden_size):
        """Render one episode to see agent performance."""
        print(f"\n>>> Rendering episode {self.episode_count}...")

        # Create test environment with rendering
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        test_env = MultiAgentSLAMEnv(
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

        if isinstance(hidden_size, int):
            test_env = CurriculumWrapper(test_env, hidden_size=hidden_size)
        test_env = MultiDiscreteToDiscreteWrapper(test_env)

        # Run episode
        obs, _ = test_env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 300:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            test_env.render()
            time.sleep(0.02)

        # Show final state
        discovered = info.get('discovered_cells', 0)
        total = info.get('total_reachable', 0)
        success = discovered >= total * 0.99 if total > 0 else False
        test_env.render()
        time.sleep(0.5)
        test_env.close()
        print(f">>> Rendering complete: {steps} steps, reward: {total_reward:.1f}, "
              f"discovered: {discovered}/{total}, success: {success}")


def create_env(hidden_size: int, env_id: int = 0):
    """Create a single environment for vectorization with improved rewards."""

    def _init():
        # Load map
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)

        # Create base environment
        base_env = MultiAgentSLAMEnv(
            width=actual_width,
            height=actual_height,
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

        # Apply wrappers
        if actual_width == 32 and actual_height == 32:
            env = CurriculumWrapper(base_env, hidden_size=hidden_size)
            env = MultiDiscreteToDiscreteWrapper(env)
        else:
            env = MultiDiscreteToDiscreteWrapper(base_env)

        # Add monitor for episode statistics
        env = Monitor(env)

        # Set seed for reproducibility
        env.reset(seed=42 + env_id)

        return env

    return _init


def train():
    """Train DQN with EfficientNet feature extractor and curriculum learning."""

    # Load map once to verify
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("="*60)
    print(" EFFICIENTNET DQN CURRICULUM TRAINING")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Steps per stage: {STEPS_PER_STAGE:,}")

    if USE_LIGHTWEIGHT:
        print("Feature Extractor: Lightweight EfficientNet (custom)")
    else:
        print("Feature Extractor: Full EfficientNet-B0 (pretrained)")

    # Curriculum stages
    if map_width == 32 and map_height == 32:
        CURRICULUM = [8, 10, 12, 14, 16, 20, 24, 28, 32]
        print(f"Curriculum stages: {CURRICULUM}")
    else:
        CURRICULUM = [None]
        print("No curriculum (map is not 32x32)")

    print("="*60 + "\n")

    # Estimate training time
    estimated_fps = N_ENVS * 200  # Conservative estimate (EfficientNet is heavier)
    total_steps = len(CURRICULUM) * STEPS_PER_STAGE
    estimated_hours = total_steps / estimated_fps / 3600
    print(f"Estimated total training time: {estimated_hours:.1f} hours")
    print(f"   ({len(CURRICULUM)} stages × {STEPS_PER_STAGE/1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    # Enable cudnn optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = None

    for stage_idx, hidden_size in enumerate(CURRICULUM):
        stage_start = time.time()

        if hidden_size is not None:
            print(f"\n{'='*60}")
            print(f" STAGE {stage_idx+1}/{len(CURRICULUM)}: Hidden {hidden_size}x{hidden_size}")
        else:
            print(f"\n{'='*60}")
            print(f" TRAINING without curriculum")
        print("-"*40)

        # Create vectorized environment
        print(f"Creating {N_ENVS} environments...")

        env_fns = [create_env(hidden_size if hidden_size else 0, i) for i in range(N_ENVS)]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

        if model is None:
            # Create new model with EfficientNet feature extractor
            print("Creating new DQN model with EfficientNet feature extractor...")

            # Choose feature extractor
            if USE_LIGHTWEIGHT:
                feature_extractor_class = SLAMLightweightEfficientNetExtractor
                feature_extractor_kwargs = dict(features_dim=256)
                print("Using lightweight EfficientNet variant")
            else:
                feature_extractor_class = SLAMEfficientNetExtractor
                feature_extractor_kwargs = dict(
                    features_dim=256,
                    efficientnet_variant='b0',
                    pretrained=True,
                    freeze_backbone=True  # Allow fine-tuning
                )
                print("Using full EfficientNet-B0 with pretrained weights")

            model = DQN(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=dict(
                    features_extractor_class=feature_extractor_class,
                    features_extractor_kwargs=feature_extractor_kwargs,
                    net_arch=[256, 256],  # Policy/value networks
                ),
                # DQN hyperparameters - ADJUSTED FOR MEMORY
                learning_rate=1e-4,  # Slightly lower LR for stability
                buffer_size=500_000,  # Reduced buffer size to save memory
                learning_starts=1000,
                batch_size=32,  # Smaller batch size - critical for memory
                tau=1.0,
                gamma=0.99,
                train_freq=8,  # Train less frequently to reduce memory pressure
                gradient_steps=1,
                target_update_interval=10000,
                # Exploration
                exploration_fraction=0.7,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                # Other
                max_grad_norm=10,
                seed=42,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )

            print(f"✅ Model created on {model.device}")
        else:
            # Reuse existing model with new environment
            model.set_env(vec_env)
            print("Model environment updated")

        # Create callbacks
        callbacks = []

        # Progress callback
        progress_callback = OptimizedProgressCallback(
            render_freq=2000,
            n_envs=N_ENVS
        )
        callbacks.append(progress_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500000 // N_ENVS,
            save_path="./models/efficientnet_checkpoints/",
            name_prefix=f"stage_{stage_idx+1}_hidden_{hidden_size}"
        )
        callbacks.append(checkpoint_callback)

        # Combine callbacks
        callback = CallbackList(callbacks)

        # Train
        print(f"\nStarting training for stage {stage_idx+1}...")
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
        if hidden_size is not None:
            model_path = f"./models/efficientnet_stage_{stage_idx+1}_hidden_{hidden_size}"
        else:
            model_path = f"./models/efficientnet_checkpoint_{stage_idx+1}"

        model.save(model_path)

        print(f"\n✅ Stage {stage_idx+1} complete!")
        print(f"   Time: {stage_time/60:.1f} minutes")
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
    model.save("./models/dqn_efficientnet_final")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final model saved to ./models/dqn_efficientnet_final.zip")
    print("="*60)


if __name__ == "__main__":
    # Verify map exists
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"✅ Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/efficientnet_checkpoints", exist_ok=True)

    # Start training
    train()