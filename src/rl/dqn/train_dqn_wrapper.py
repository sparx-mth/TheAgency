"""
train_dqn_wrapper.py - DQN training using MultiDiscrete to Discrete wrapper
Uses the wrapper to make MultiDiscrete action space compatible with DQN.
"""

import os
import numpy as np
import warnings

from rl.feature_extractors.efficientnet_feature_extractor import SLAMCNNExtractor

warnings.filterwarnings("ignore")

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.base.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper


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
    """Create the environment with wrapper for DQN compatibility."""

    # Optimized sensor for 10x10 house map
    sensor = CameraSensor(
        max_range=5,      # Good range for 10x10
        fov_deg=90,       # Wide field of view
        num_rays=20       # Sufficient rays
    )

    # Create base environment
    base_env = MultiAgentSLAMEnv(
        width=10,
        height=10,
        num_agents=1,  # Start with single agent for DQN
        max_steps=500,
        map_path="/home/user/nadav/TheAgency/resources/planner/maps/house_map_10.txt",
        randomize=False,
        render_mode=None,
        sensor_config={0: sensor},
        # Optimized rewards for exploration
        discovery_reward=1.0,
        collision_penalty=-0.1,
        step_penalty=0.0,
        completion_bonus=50.0,
    )

    # Wrap with MultiDiscrete to Discrete converter
    wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

    return Monitor(wrapped_env)


def train():
    """Train DQN agent on house_map_10 using the wrapper."""

    print("="*60)
    print("DQN TRAINING FOR HOUSE MAP 10 (WITH MULTIDISCRETE WRAPPER)")
    print("="*60)

    # Create environment
    print("\nCreating environment...")
    env = create_env()

    # Wrap in vectorized environment (no normalization needed for DQN)
    vec_env = DummyVecEnv([lambda: env])

    print("Environment created successfully")
    print(f"Original observation space: {env.env.observation_space}")
    print(f"Wrapped action space: {env.action_space}")
    print(f"Total possible actions: {env.action_space.n}")

    # Show some action examples
    print("\nFirst 10 action mappings:")
    # Access the wrapper through the Monitor wrapper
    multidiscrete_wrapper = env.env  # Get the underlying wrapper
    meanings = multidiscrete_wrapper.get_action_meanings()
    for i in range(min(10, len(meanings))):
        print(f"  Action {i}: {meanings[i]}")

    # Create DQN model
    print("\nCreating DQN model...")
    model = DQN(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=SLAMCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[512, 512],  # Hidden layers for Q-network
        ),
        # DQN-specific parameters
        learning_rate=1e-4,
        buffer_size=100000,      # Replay buffer size
        learning_starts=1000,    # Start training after 1000 steps
        batch_size=32,           # Batch size for training
        tau=1.0,                 # Hard update (target network update frequency)
        gamma=0.99,              # Discount factor
        train_freq=4,            # Train every 4 steps
        gradient_steps=1,        # Gradient steps per training
        target_update_interval=1000,  # Update target network every 1000 steps
        # Exploration
        exploration_fraction=0.3,     # Fraction of training for exploration
        exploration_initial_eps=1.0,  # Initial epsilon
        exploration_final_eps=0.05,   # Final epsilon
        # Other
        max_grad_norm=10,
        tensorboard_log="./logs/dqn_tensorboard/",
        verbose=1,
        seed=42,
        device='auto',
    )

    print("Model created successfully")
    print(f"Using device: {model.device}")
    print(f"Replay buffer size: {model.buffer_size}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/dqn/checkpoints/",
        name_prefix="dqn_house10"
    )

    progress_callback = ProgressCallback()

    # Train
    print("\n" + "="*60)
    print("Starting DQN training...")
    print(f"Total timesteps: 1,000,000")
    print(f"Exploration will decay from 1.0 to 0.05 over {0.3 * 1_000_000:,.0f} steps")
    print("="*60 + "\n")

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=[checkpoint_callback, progress_callback],
            log_interval=10,
            tb_log_name="DQN_house10",
            reset_num_timesteps=True,
            progress_bar=False
        )

        # Save final model
        print("\n" + "="*60)
        print("Training completed!")
        model.save("./models/dqn/final_model")
        print("Model saved to ./models/dqn/final_model.zip")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        model.save("./models/dqn/interrupted_model")
        print("Model saved to ./models/dqn/interrupted_model.zip")

    finally:
        vec_env.close()




if __name__ == "__main__":
    # Create directories
    os.makedirs("./models/dqn/checkpoints", exist_ok=True)
    os.makedirs("./logs/dqn_tensorboard", exist_ok=True)

    # Run training
    train()
