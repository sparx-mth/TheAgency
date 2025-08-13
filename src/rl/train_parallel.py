"""
train_parallel.py - Parallel environment training for faster learning
Uses K parallel environments for accelerated training
"""

import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from environments.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor
from cnn_feature_extractor import SLAMCNNExtractor


class ProgressCallback(BaseCallback):
    """Enhanced callback to track training progress across multiple environments."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check for completed episodes across all environments
        if 'dones' in self.locals and self.locals['dones'] is not None:
            dones = self.locals['dones']
            infos = self.locals['infos']
            rewards = self.locals['rewards']

            for i, done in enumerate(dones):
                if done:
                    info = infos[i]
                    reward = rewards[i]

                    self.episode_count += 1
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(info.get('step', 0))

                    # Print every 50 episodes (more frequent due to parallel envs)
                    if self.episode_count % 50 == 0:
                        recent_rewards = self.episode_rewards[-50:]
                        print(f"Episode {self.episode_count}: "
                              f"Avg Reward={np.mean(recent_rewards):.2f}, "
                              f"Progress={info.get('progress', 0) * 100:.1f}%, "
                              f"Discovered={info.get('discovered_cells', 0)}")
        return True


def create_env():
    """Create a single environment instance."""
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
        render_mode=None,
        sensor_config={0: sensor},
        discovery_reward=1.0,
        collision_penalty=-0.1,
        step_penalty=0.0,
        completion_bonus=50.0,
    )

    return Monitor(env)


def create_parallel_envs(num_envs=8, use_subproc=True):
    """
    Create K parallel environments.

    Args:
        num_envs: Number of parallel environments
        use_subproc: Use SubprocVecEnv (faster) vs DummyVecEnv (simpler)
    """
    print(f"Creating {num_envs} parallel environments...")

    # Create list of environment creation functions
    env_fns = [create_env for _ in range(num_envs)]

    if use_subproc and num_envs > 1:
        # SubprocVecEnv: Each env runs in separate process (faster)
        vec_env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {num_envs} processes")
    else:
        # DummyVecEnv: All envs in same process (simpler, good for debugging)
        vec_env = DummyVecEnv(env_fns)
        print(f"Using DummyVecEnv with {num_envs} environments")

    return vec_env


def train_parallel(algorithm="ppo", num_envs=8, total_timesteps=2_000_000):
    """Train with K parallel environments."""

    print("=" * 60)
    print(f"PARALLEL {algorithm.upper()} TRAINING - {num_envs} ENVIRONMENTS")
    print("=" * 60)

    # Create parallel environments
    vec_env = create_parallel_envs(num_envs)

    # Add normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("\nEnvironments created successfully")
    print(f"Total parallel environments: {num_envs}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Create model based on algorithm
    print(f"\nCreating {algorithm.upper()} model...")

    if algorithm.lower() == "ppo":
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=dict(
                features_extractor_class=SLAMCNNExtractor,
                features_extractor_kwargs=dict(features_dim=256),
            ),
            # Adjusted for parallel training
            learning_rate=3e-4,
            n_steps=2048 // num_envs,  # Adjust steps per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=f"./logs/parallel_{algorithm}_tensorboard/",
            verbose=1,
            seed=42,
            device='auto',
        )
        model_dir = f"parallel_{algorithm}"

    elif algorithm.lower() == "a2c":
        model = A2C(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=dict(
                features_extractor_class=SLAMCNNExtractor,
                features_extractor_kwargs=dict(features_dim=256),
            ),
            # Adjusted for parallel training
            learning_rate=7e-4,
            n_steps=5,  # Keep small for A2C
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            use_rms_prop=True,
            tensorboard_log=f"./logs/parallel_{algorithm}_tensorboard/",
            verbose=1,
            seed=42,
            device='auto',
        )
        model_dir = f"parallel_{algorithm}"

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print("Model created successfully")
    print(f"Using device: {model.device}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./models/{model_dir}/checkpoints/",
        name_prefix=f"{algorithm}_house10_parallel"
    )

    progress_callback = ProgressCallback()

    # Train
    print("\n" + "=" * 60)
    print("Starting parallel training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Effective speedup: ~{num_envs}x faster data collection")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            log_interval=10,
            tb_log_name=f"{algorithm.upper()}_house10_parallel",
            reset_num_timesteps=True,
            progress_bar=False
        )

        # Save final model
        print("\n" + "=" * 60)
        print("Training completed!")
        model.save(f"./models/{model_dir}/final_model")
        vec_env.save(f"./models/{model_dir}/vec_normalize.pkl")
        print(f"Model saved to ./models/{model_dir}/final_model.zip")
        print(f"Normalizer saved to ./models/{model_dir}/vec_normalize.pkl")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        model.save(f"./models/{model_dir}/interrupted_model")
        vec_env.save(f"./models/{model_dir}/vec_normalize.pkl")
        print(f"Model saved to ./models/{model_dir}/interrupted_model.zip")

    finally:
        vec_env.close()


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    algorithm = "ppo"  # default
    num_envs = 8  # default
    timesteps = 2_000_000  # default

    if len(sys.argv) > 1:
        algorithm = sys.argv[1].lower()
    if len(sys.argv) > 2:
        num_envs = int(sys.argv[2])
    if len(sys.argv) > 3:
        timesteps = int(sys.argv[3])

    print(f"Training {algorithm.upper()} with {num_envs} parallel environments")
    print("Usage: python train_parallel.py [ppo|a2c] [num_envs] [timesteps]")
    print(f"Using: {algorithm}, {num_envs} envs, {timesteps:,} timesteps\n")

    # Create directories
    model_dir = f"parallel_{algorithm}"
    os.makedirs(f"./models/{model_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"./logs/parallel_{algorithm}_tensorboard", exist_ok=True)

    # Run parallel training
    train_parallel(algorithm, num_envs, timesteps)