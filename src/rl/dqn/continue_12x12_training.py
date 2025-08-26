"""
continue_12x12_training.py - Continue training on 12x12 until consistent completion
Loads the saved 12x12 model and continues training with appropriate parameters.
"""

import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.curriculum_wrapper import CurriculumWrapper
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from sensors.camera_sensor import CameraSensor

# Paths
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_11.txt"
MODEL_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/12x12_continuation/continued_12x12_1500000_steps.zip"

# Training parameters
N_ENVS = 4
ADDITIONAL_STEPS = 10_000_000  # Another 10M steps for 12x12
HIDDEN_SIZE = 12  # Continue with 12x12


class ContinuationCallback(BaseCallback):
    """Modified callback that tracks completion rate more carefully."""

    def __init__(self, render_freq=2000, n_envs=1):
        super().__init__()
        self.render_freq = render_freq
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_discoveries = []
        self.recent_completions = 0
        self.consecutive_completions = 0
        self.best_discovery = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"📊 Continuation training started with {self.n_envs} environments")
        print("-" * 60)

    def _on_step(self) -> bool:
        self.total_steps += self.n_envs

        for i in range(self.n_envs):
            if self.locals.get('dones')[i]:
                self.episode_count += 1
                info = self.locals['infos'][i]

                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)

                discovered = info.get('discovered_cells', 0)
                total = info.get('total_reachable', 0)
                collisions = sum(info.get('collision_counts', [0]))

                self.recent_discoveries.append(discovered)

                # Track best
                if discovered > self.best_discovery:
                    self.best_discovery = discovered

                # Check completion (>=99% discovered)
                if discovered >= total * 0.99:
                    self.recent_completions += 1
                    self.consecutive_completions += 1
                else:
                    self.consecutive_completions = 0

                # Print every 100 episodes
                if self.episode_count % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    recent_disc = self.recent_discoveries[-100:]

                    avg_reward = np.mean(recent_rewards)
                    avg_length = np.mean(recent_lengths)
                    avg_disc = np.mean(recent_disc)
                    max_disc = np.max(recent_disc) if recent_disc else 0
                    completion_rate = self.recent_completions

                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:8,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Disc: {avg_disc:5.1f}/{total:3d} | "
                          f"Max: {max_disc:3d} | "
                          f"Best: {self.best_discovery:3d} | "
                          f"Complete: {completion_rate:3d}% | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")

                    # Success message if consistently completing
                    if completion_rate >= 90:
                        print(f"  🎯 Excellent! Completing {completion_rate}% of episodes!")
                    elif completion_rate >= 50:
                        print(f"  ✓ Good progress! Completing {completion_rate}% of episodes")

                    self.recent_completions = 0

                # Render periodically
                if self.episode_count % self.render_freq == 0:
                    self.render_episode()

        # Early stopping if mastery achieved (95%+ completion for 500 episodes)
        if self.consecutive_completions >= 500:
            print("\n🏆 MASTERY ACHIEVED! Agent completing 95%+ consistently!")
            return False  # Stop training

        return True

    def render_episode(self):
        print(f"\n>>> Rendering episode {self.episode_count}...")

        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        test_env = MultiAgentSLAMEnv(
            width=32, height=32, num_agents=1, max_steps=2000,
            map_path=MAP_PATH, render_mode='human',
            sensor_config={0: sensor},
            discovery_reward=1.0, collision_penalty=-0.5,
            step_penalty=0.0, completion_bonus=72.0,
        )
        test_env = CurriculumWrapper(test_env, hidden_size=HIDDEN_SIZE)
        test_env = MultiDiscreteToDiscreteWrapper(test_env)

        obs, _ = test_env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 800:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            test_env.render()
            time.sleep(0.02)

        discovered = info.get('discovered_cells', 0)
        total = info.get('total_reachable', 0)
        test_env.render()
        time.sleep(0.5)
        test_env.close()

        status = "✓ COMPLETE" if discovered >= total * 0.99 else "✗ Incomplete"
        print(f">>> {status}: {steps} steps, reward: {total_reward:.1f}, discovered: {discovered}/{total}")


def create_env(env_id: int = 0):
    """Create environment for 12x12 training."""

    def _init():
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        base_env = MultiAgentSLAMEnv(
            width=actual_width, height=actual_height, num_agents=1, max_steps=2000,
            map_path=MAP_PATH, render_mode=None,
            sensor_config={0: sensor},
            discovery_reward=1.0,
            collision_penalty=-0.5,
            step_penalty=0.0,
            completion_bonus=72.0,  # Keep original bonus
        )

        env = CurriculumWrapper(base_env, hidden_size=HIDDEN_SIZE)
        env = MultiDiscreteToDiscreteWrapper(env)
        env = Monitor(env)
        env.reset(seed=42 + env_id)
        return env

    return _init


def continue_training():
    """Continue training 12x12 until mastery."""

    print("=" * 60)
    print("📚 CONTINUING 12x12 TRAINING FROM BEST CHECKPOINT")
    print(f"Loading model: {MODEL_PATH}")
    print("This checkpoint is from ~episode 2100 with 60% completion rate")
    print(f"Hidden size: {HIDDEN_SIZE}x{HIDDEN_SIZE}")
    print(f"Additional steps: {ADDITIONAL_STEPS:,}")
    print("=" * 60 + "\n")

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Create environments
    print(f"Creating {N_ENVS} environments...")
    env_fns = [create_env(i) for i in range(N_ENVS)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Load model with continuation parameters
    print("Loading model with CORRECTED parameters for stability...")
    model = DQN.load(
        MODEL_PATH,
        env=vec_env,
        # FIXED PARAMETERS: Low exploration, moderate learning
        # At 1.5M steps you had 60% completion, so we need stable exploitation
        exploration_fraction=1.0,  # Keep constant exploration throughout
        exploration_initial_eps=0.03,  # Only 3% random actions (very low)
        exploration_final_eps=0.03,  # Stay constant - no decay
        learning_rate=3e-5,  # Moderate learning rate (not too slow)
        target_update_interval=10000,  # Regular updates (not too infrequent)
    )

    print(f"✓ Model loaded on {model.device}")
    print(f"  CONSTANT exploration: ε = 0.03 (3% random)")
    print(f"  Learning rate: 3e-5 (moderate for stability)")
    print(f"  Target updates: every 10k steps (balanced)")
    print("  Goal: Achieve 90%+ completion rate without forgetting\n")

    # Callbacks
    callbacks = []

    progress_callback = ContinuationCallback(render_freq=2000, n_envs=N_ENVS)
    callbacks.append(progress_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // N_ENVS,
        save_path="./models/12x12_continuation/",
        name_prefix="continued_12x12"
    )
    callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks)

    # Train
    print("Starting continuation training...")
    print("Goal: Achieve 90%+ completion rate\n")

    start_time = time.time()

    model.learn(
        total_timesteps=ADDITIONAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,  # Reset for this continuation phase
        progress_bar=False
    )

    # Save final model
    final_path = "./models/12x12_mastered"
    model.save(final_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ CONTINUATION TRAINING COMPLETE!")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"Final model saved to: {final_path}.zip")
    print("=" * 60)

    # Clean up
    vec_env.close()

    return final_path


if __name__ == "__main__":
    # Verify files
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"Map not found: {MAP_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"✓ Map verified: {MAP_PATH}")
    print(f"✓ Model verified: {MODEL_PATH}\n")

    # Create directories
    os.makedirs("./models/12x12_continuation", exist_ok=True)

    # Continue training
    final_model = continue_training()

    print(f"\n🎯 Use this model for 14x14: {final_model}.zip")