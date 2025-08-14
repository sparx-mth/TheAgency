"""
train_simple.py - Simplified PPO training for house_map_10.txt
Focuses on single agent with optimized parameters for the specific map.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor
from cnn_feature_extractor import SLAMCNNExtractor


class ProgressCallback(BaseCallback):
    """Simple callback to track training progress."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            reward = self.locals['rewards'][0]

            self.episode_count += 1
            self.episode_rewards.append(reward)
            self.episode_lengths.append(info.get('step', 0))

            # Print every 100 episodes
            if self.episode_count % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                print(f"Episode {self.episode_count}: "
                      f"Avg Reward={np.mean(recent_rewards):.2f}, "
                      f"Progress={info.get('progress', 0)*100:.1f}%, "
                      f"Discovered={info.get('discovered_cells', 0)}")
        return True


def create_env():
    """Create the environment with optimized settings for house_map_10."""

    # Optimized sensor for 10x10 house map
    sensor = CameraSensor(
        max_range=5,      # Good range for 10x10
        fov_deg=90,       # Wide field of view
        num_rays=20       # Sufficient rays
    )

    env = MultiAgentSLAMEnv(
        width=10,
        height=10,
        num_agents=1,
        max_steps=500,    # Sufficient for 10x10
        map_path="/home/user/nadav/TheAgency/resources/planner/maps/house_map_10.txt",
        randomize=False,  # Always use the same map
        render_mode=None,
        sensor_config={0: sensor},
        # Optimized rewards for exploration
        discovery_reward=1.0,      # Good reward for discovery
        collision_penalty=-0.1,    # Small penalty
        step_penalty=0.0,          # No step penalty initially
        completion_bonus=50.0,     # Big bonus for completion
    )

    return Monitor(env)


def train():
    """Train PPO agent on house_map_10."""

    print("="*60)
    print("SIMPLE PPO TRAINING FOR HOUSE MAP 10")
    print("="*60)

    # Create environment
    print("\nCreating environment...")
    env = create_env()

    # Wrap in vectorized environment with normalization
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Environment created successfully")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create PPO model with CNN feature extractor
    print("\nCreating PPO model...")
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=SLAMCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        ),
        # Core parameters - mostly defaults
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        # Exploration
        ent_coef=0.01,        # Some exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Other
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/simple_tensorboard/",
        verbose=1,
        seed=42,
        device='auto',
    )

    print("Model created successfully")
    print(f"Using device: {model.device}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="../dqn/models/simple/checkpoints/",
        name_prefix="ppo_house10"
    )

    progress_callback = ProgressCallback()

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Total timesteps: 2,000,000")
    print("="*60 + "\n")

    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=[checkpoint_callback, progress_callback],
            log_interval=10,
            tb_log_name="PPO_house10",
            reset_num_timesteps=True,
            progress_bar=False
        )

        # Save final model
        print("\n" + "="*60)
        print("Training completed!")
        model.save("./models/simple/final_model")
        vec_env.save("./models/simple/vec_normalize.pkl")
        print("Model saved to ./models/simple/final_model.zip")
        print("Normalizer saved to ./models/simple/vec_normalize.pkl")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        model.save("./models/simple/interrupted_model")
        vec_env.save("./models/simple/vec_normalize.pkl")
        print("Model saved to ./models/simple/interrupted_model.zip")

    finally:
        vec_env.close()


if __name__ == "__main__":
    # Create directories
    os.makedirs("../dqn/models/simple/checkpoints", exist_ok=True)
    os.makedirs("../logs/simple_tensorboard", exist_ok=True)

    # Run training
    train()