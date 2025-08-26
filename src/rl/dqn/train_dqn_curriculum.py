"""
train_dqn_curriculum_transfer.py - Minimal changes to load 10x10 and continue from 12x12
Only the essential modifications from your original file.
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
from rl.feature_extractors.cnn_feature_extractor import SLAMCNNExtractor

# FIXED MAP PATH
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_11.txt"

# ADD THIS: Path to pre-trained 10x10 model
PRETRAINED_MODEL = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/adaptive_stage_2_hidden_10.zip"

# OPTIMIZATION SETTINGS
N_ENVS = 4
STEPS_PER_STAGE = 10_000_000

# [KEEP ALL YOUR CALLBACK CLASSES EXACTLY THE SAME - OptimizedProgressCallback unchanged]
class OptimizedProgressCallback(BaseCallback):
    """Optimized callback for multi-environment training."""

    def __init__(self, render_freq=2000, n_envs=1):
        super().__init__()
        self.render_freq = render_freq
        self.n_envs = n_envs
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_discoveries = []
        self.recent_completions = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f" Training started with {self.n_envs} environments")
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
                progress = info.get('progress', 0) * 100
                collisions = sum(info.get('collision_counts', [0]))
                hidden_size = info.get('hidden_size', 'N/A')
                self.recent_discoveries.append(discovered)
                if discovered >= total * 1.0:
                    self.recent_completions += 1
                if self.episode_count % 100 == 0:
                    recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                    recent_lengths = self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths
                    recent_disc = self.recent_discoveries[-100:] if len(self.recent_discoveries) >= 100 else self.recent_discoveries
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    avg_length = np.mean(recent_lengths) if recent_lengths else 0
                    avg_disc = np.mean(recent_disc) if recent_disc else 0
                    completion_rate = self.recent_completions
                    elapsed = time.time() - self.start_time
                    fps = self.total_steps / elapsed if elapsed > 0 else 0
                    epsilon = self.model.exploration_schedule(self.model._current_progress_remaining)
                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Disc: {avg_disc:3.0f}/{total:3d} | "
                          f"Complete: {completion_rate:3d}% | "
                          f"Collision: {collisions:3d} | "
                          f"Hidden: {hidden_size:>3} | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f}")
                    self.recent_completions = 0
                if self.episode_count % self.render_freq == 0:
                    self.render_episode(hidden_size)
        return True

    def render_episode(self, hidden_size):
        print(f"\n>>> Rendering episode {self.episode_count}...")
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        test_env = MultiAgentSLAMEnv(
            width=32, height=32, num_agents=1, max_steps=2000,
            map_path=MAP_PATH, render_mode='human',
            sensor_config={0: sensor},
            discovery_reward=1.0, collision_penalty=-0.5,
            step_penalty=0.0, completion_bonus=50.0,
        )
        if isinstance(hidden_size, int):
            test_env = CurriculumWrapper(test_env, hidden_size=hidden_size)
        test_env = MultiDiscreteToDiscreteWrapper(test_env)
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
        discovered = info.get('discovered_cells', 0)
        total = info.get('total_reachable', 0)
        test_env.render()
        time.sleep(0.5)
        test_env.close()
        print(f">>> Rendering complete: {steps} steps, reward: {total_reward:.1f}, discovered: {discovered}/{total}")


# [KEEP create_env FUNCTION EXACTLY THE SAME]
def create_env(hidden_size: int, env_id: int = 0):
    """Create a single environment for vectorization with improved rewards."""
    def _init():
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape
        sensor = CameraSensor(max_range=8, fov_deg=60, num_rays=24)
        base_env = MultiAgentSLAMEnv(
            width=actual_width, height=actual_height, num_agents=1, max_steps=2000,
            map_path=MAP_PATH, render_mode=None,
            sensor_config={0: sensor},
            discovery_reward=1.0, collision_penalty=-0.5,
            step_penalty=0.0, completion_bonus=50.0,
        )
        if actual_width == 32 and actual_height == 32:
            env = CurriculumWrapper(base_env, hidden_size=hidden_size)
            env = MultiDiscreteToDiscreteWrapper(env)
        else:
            env = MultiDiscreteToDiscreteWrapper(base_env)
        env = Monitor(env)
        env.reset(seed=42 + env_id)
        return env
    return _init


def train():
    """Train DQN with curriculum learning using multiple environments."""

    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("="*60)
    print(" DQN CURRICULUM TRAINING - TRANSFER LEARNING")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Steps per stage: {STEPS_PER_STAGE:,}")

    # CHANGE 1: Start from 12x12 since we're loading 10x10 model
    if map_width == 32 and map_height == 32:
        CURRICULUM = [12, 14, 16, 20, 24, 28, 32]  # Skip 8 and 10
        print(f"Curriculum stages (starting from 12): {CURRICULUM}")
    else:
        CURRICULUM = [None]
        print("No curriculum (map is not 32x32)")

    print("="*60 + "\n")

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

        print(f"Creating {N_ENVS} environments...")
        env_fns = [create_env(hidden_size if hidden_size else 0, i) for i in range(N_ENVS)]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

        # CHANGE 2: Load pre-trained model for first stage (12x12)
        if model is None:
            if os.path.exists(PRETRAINED_MODEL):
                print(f"Loading pre-trained 10x10 model from: {PRETRAINED_MODEL}")
                model = DQN.load(
                    PRETRAINED_MODEL,
                    env=vec_env,
                    exploration_fraction=0.7,  # Explore for 70% of training (7M steps)
                    exploration_initial_eps=0.6,  # Start with 60% exploration
                    exploration_final_eps=0.05,  # End with 5% exploration
                    learning_rate=5e-5,
                    target_update_interval=20000,
                )
                print(f" Model loaded on {model.device}")
                print(f" Using transfer learning parameters: ε 0.3→0.05 over 50% of training")
            else:
                # Fallback to creating new model if file not found
                print(f"WARNING: Pre-trained model not found at {PRETRAINED_MODEL}")
                print("Creating new DQN model instead...")
                model = DQN(
                    "MultiInputPolicy", vec_env,
                    policy_kwargs=dict(
                        features_extractor_class=SLAMCNNExtractor,
                        features_extractor_kwargs=dict(features_dim=256),
                        net_arch=[512, 512, 512, 512],
                    ),
                    learning_rate=1e-4, buffer_size=1_000_000, learning_starts=1000,
                    batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
                    target_update_interval=10000, exploration_fraction=0.7,
                    exploration_initial_eps=1.0, exploration_final_eps=0.05,
                    max_grad_norm=10, seed=42, device='auto',
                )
                print(f" Model created on {model.device}")
        else:
            model.set_env(vec_env)
            print("Model environment updated")

        # [KEEP CALLBACKS EXACTLY THE SAME]
        callbacks = []
        progress_callback = OptimizedProgressCallback(render_freq=2000, n_envs=N_ENVS)
        callbacks.append(progress_callback)
        checkpoint_callback = CheckpointCallback(
            save_freq=500000 // N_ENVS,
            save_path="./models/checkpoints/",
            name_prefix=f"transfer_stage_{stage_idx+1}_hidden_{hidden_size}"  # Changed prefix
        )
        callbacks.append(checkpoint_callback)
        callback = CallbackList(callbacks)

        print(f"\nStarting training for stage {stage_idx+1}...")
        print(f"Target: {STEPS_PER_STAGE:,} steps")

        # CHANGE 4: Use cumulative timesteps for consistency with original
        model.learn(
            total_timesteps=STEPS_PER_STAGE * (1 + stage_idx),
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False
        )

        stage_time = time.time() - stage_start
        stage_fps = STEPS_PER_STAGE / stage_time

        # CHANGE 5: Different naming for transfer models
        if hidden_size is not None:
            model_path = f"./models/transfer_stage_{stage_idx+1}_hidden_{hidden_size}"
        else:
            model_path = f"./models/transfer_checkpoint_{stage_idx+1}"
        model.save(model_path)

        print(f"\n Stage {stage_idx+1} complete!")
        print(f"   Time: {stage_time/60:.1f} minutes")
        print(f"   Average FPS: {stage_fps:.0f}")
        print(f"   Model saved to: {model_path}.zip")

        stages_left = len(CURRICULUM) - stage_idx - 1
        if stages_left > 0:
            est_remaining = (stages_left * STEPS_PER_STAGE) / stage_fps / 60
            print(f"   Estimated time remaining: {est_remaining:.0f} minutes")

        vec_env.close()

    # CHANGE 6: Different final model name
    model.save("./models/dqn_curriculum_transfer_final")
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print(f"Final model saved to ./models/dqn_curriculum_transfer_final.zip")
    print("="*60)


if __name__ == "__main__":
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")
    print(f" Map file verified: {MAP_PATH}")

    # CHANGE 7: Check for pre-trained model
    if not os.path.exists(PRETRAINED_MODEL):
        print(f" WARNING: Pre-trained model not found: {PRETRAINED_MODEL}")
        print(" Training will start from scratch if file is not found")
    else:
        print(f" Pre-trained model found: {PRETRAINED_MODEL}")

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/checkpoints", exist_ok=True)
    train()