"""
train_dqn.py

Simple DQN training script for the Single-Agent SLAM environment using Stable Baselines3.
This script provides a minimal setup to train a DQN agent for exploration and mapping.
"""

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.single_agent_wrapper import SingleAgentSLAMEnv
from sensors.camera_sensor import CameraSensor


def make_env():
    """Create and configure the SLAM environment."""
    # Configure sensor
    sensor = CameraSensor(
        max_range=8,
        fov_deg=60,
        num_rays=20
    )

    # Create environment
    env = SingleAgentSLAMEnv(
        width=20,
        height=20,
        max_steps=500,
        randomize=True,
        render_mode=None,  # Set to 'human' to visualize during training
        sensor=sensor,
        # Reward configuration
        discovery_reward=0.1,
        collision_penalty=-0.5,
        step_penalty=-0.001,
        completion_bonus=10.0,
    )

    return env


def train():
    """Main training function."""

    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create training environment
    print("Creating training environment...")
    train_env = Monitor(make_env(), filename="logs/training")
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = Monitor(make_env(), filename="logs/evaluation")
    eval_env = DummyVecEnv([lambda: eval_env])

    # Configure DQN model
    print("Initializing DQN model...")
    model = DQN(
        policy="MultiInputPolicy",  # Required for dict observation spaces
        env=train_env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_slam"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Train the model
    print("Starting training...")
    print("You can monitor progress with: tensorboard --logdir ./logs/tensorboard/")

    total_timesteps = 200000  # Adjust based on your needs

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        # Save final model
        print("Saving final model...")
        model.save("models/dqn_slam_final")

        print("Training complete!")
        print(f"Best model saved at: models/best/best_model.zip")
        print(f"Final model saved at: models/dqn_slam_final.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model...")
        model.save("models/dqn_slam_interrupted")
        print("Model saved at: models/dqn_slam_interrupted.zip")


def test_trained_model(model_path="models/best/best_model", num_episodes=5):
    """Test a trained model."""
    import os

    # Check if model exists
    model_file = model_path if model_path.endswith('.zip') else f"{model_path}.zip"
    if not os.path.exists(model_file):
        print(f"Error: Model not found at {model_file}")
        print("Please train a model first using: python train_dqn.py --train")
        return

    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # Create test environment with rendering
    env = make_env()
    env.render_mode = 'human'

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # Render
            env.render()

            # Print progress
            if step % 50 == 0:
                print(f"  Step {step}: Progress {info['progress']*100:.1f}%")

        print(f"  Episode finished: Total reward = {episode_reward:.2f}")
        print(f"  Final progress: {info['progress']*100:.1f}%")
        print(f"  Total collisions: {info['collision_counts'][0]}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN agent for SLAM")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--model", type=str, default="models/best/best_model",
                       help="Path to model for testing")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of test episodes")
    parser.add_argument("--quick", action="store_true",
                       help="Quick training (10k steps) for testing")

    args = parser.parse_args()

    if args.train:
        if args.quick:
            # Override total_timesteps for quick testing
            import sys
            sys.modules[__name__].total_timesteps = 10000
        train()
    elif args.test:
        test_trained_model(args.model, args.episodes)
    else:
        print("DQN Training Script for SLAM Environment")
        print("-" * 40)
        print("Usage:")
        print("  Train a model:    python train_dqn.py --train")
        print("  Quick training:   python train_dqn.py --train --quick")
        print("  Test a model:     python train_dqn.py --test")
        print("  Test specific:    python train_dqn.py --test --model models/dqn_slam_final")
        print("")
        print("Monitor training:   tensorboard --logdir ./logs/tensorboard/")