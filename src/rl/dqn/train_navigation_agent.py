"""
train_navigation_agent.py - Fixed DQN training for Navigation task with evaluation rendering
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
from rl.feature_extractors.cnn_feature_extractor import NavigationCNNExtractor  # CHANGED

# FIXED MAP PATH
MAP_PATH = "/home/user/nadav/TheAgency/resources/planner/maps/house_map_19.txt"

# TRAINING SETTINGS
N_ENVS = 4
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
                    loss = self.model.logger.name_to_value.get("train/loss", None)

                    print(f"Ep {self.episode_count:5d} | "
                          f"Steps: {self.total_steps:7,d} | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"Length: {avg_length:5.0f} | "
                          f"Success: {success_rate:5.1f}% | "
                          f"Final Dist: {avg_final_dist:4.1f} | "
                          f"ε: {epsilon:.3f} | "
                          f"FPS: {fps:4.0f} | "
                          f"Loss: {loss:.4f}" if loss is not None else "")

        return True


class EvaluationRenderingCallback(BaseCallback):
    """Callback for rendering evaluation episodes every N training episodes."""

    def __init__(self, eval_env_fn, eval_freq=500, n_eval_episodes=2, verbose=1):
        """
        Args:
            eval_env_fn: Function that creates a new evaluation environment
            eval_freq: Evaluate every N episodes
            n_eval_episodes: Number of episodes to render during evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn  # Store function instead of environment
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.episode_count = 0
        self.n_envs = None  # Will be set in _init_callback

    def _init_callback(self) -> None:
        """Initialize the callback."""
        if self.n_envs is None:
            self.n_envs = self.training_env.num_envs

    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        for i in range(self.n_envs):
            if self.locals.get('dones')[i]:
                self.episode_count += 1

                # Check if it's time to evaluate
                if self.episode_count % self.eval_freq == 0:
                    self._run_evaluation()

        return True

    def _run_evaluation(self):
        """Run and render evaluation episodes."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION at episode {self.episode_count} - Rendering {self.n_eval_episodes} episodes...")
        print("-" * 60)

        # Create a fresh evaluation environment
        eval_env = self.eval_env_fn()

        try:
            for ep_num in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                episode_length = 0

                print(f"\nEpisode {ep_num + 1}/{self.n_eval_episodes}")

                while not (done or truncated):
                    # Get action from model (deterministic for evaluation)
                    action, _ = self.model.predict(obs, deterministic=True)

                    # Step environment
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    # Render
                    eval_env.render()

                # Print episode summary
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Length: {episode_length}")
                if 'task_success' in info:
                    print(f"  Success: {info['task_success']}")
                if 'goal_position' in info:
                    print(f"  Goal was at: {info['goal_position']}")

        finally:
            # Always close the environment after evaluation
            eval_env.close()
            print("Evaluation environment closed.")

        print("=" * 60 + "\n")

def create_env(env_id: int = 0, exploration_steps: int = 20, render_mode: str = None):
    """Create single navigation environment with configurable exploration."""

    def _init():
        # Load map dimensions
        loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
        actual_height, actual_width = loaded_map.shape

        # Create sensor
        sensor = CameraSensor(
            max_range=4,
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
            'render_mode': render_mode,  # Pass render_mode through
            'sensor_config': {0: sensor},
            # Base rewards
            'discovery_reward': 0.5,
            'collision_penalty': -1.0,
            'step_penalty': -0.01,
            'completion_bonus': 0.0,
        }

        # Create navigation wrapper with configurable exploration
        env = NavigationWrapper(
            env_config=env_config,
            # Exploration - NOW CONFIGURABLE
            exploration_steps=exploration_steps,
            # Task parameters
            max_steps_to_goal=200,
            # Rewards
            goal_reached_reward=200.0,
            time_penalty=0.01,
        )

        # IMPORTANT: Reset once to ensure observation space is properly set up
        # This initializes goal_position in the observation
        env.reset(seed=42 + env_id)

        # Add monitor AFTER first reset
        env = Monitor(env)

        return env

    return _init


def train():
    """Train DQN for navigation task with progressive exploration curriculum."""

    # Verify map
    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    map_height, map_width = loaded_map.shape

    print("=" * 60)
    print("DQN NAVIGATION TRAINING - PROGRESSIVE EXPLORATION")
    print(f"Map: {MAP_PATH}")
    print(f"Map size: {map_width}x{map_height}")
    print(f"Parallel environments: {N_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")

    # PROGRESSIVE EXPLORATION CURRICULUM
    CURRICULUM = [
        ("Easy: 30 exploration steps", 30, int(TOTAL_TIMESTEPS * 0.01)),  # First 20%
        ("Medium: 40 exploration steps", 40, int(TOTAL_TIMESTEPS * 0.05)),  # Next 30%
        ("Hard: 50 exploration steps", 50, int(TOTAL_TIMESTEPS * 0.2)),  # Next 30%
        ("Expert: 100 exploration steps", 100, int(TOTAL_TIMESTEPS * 0.4)),  # Final 20%
    ]

    print("\nProgressive Exploration Curriculum:")
    for idx, (name, exploration, steps) in enumerate(CURRICULUM):
        print(f"  Stage {idx + 1}: {name} ({steps:,} steps)")
    print("=" * 60 + "\n")

    # Estimate time
    estimated_fps = N_ENVS * 450
    estimated_hours = TOTAL_TIMESTEPS / estimated_fps / 3600
    print(f"Estimated total time: {estimated_hours:.1f} hours")
    print(f"({TOTAL_TIMESTEPS / 1e6:.0f}M steps ÷ ~{estimated_fps} FPS)\n")

    model = None

    for stage_idx, (stage_name, exploration_steps, stage_timesteps) in enumerate(CURRICULUM):
        stage_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_idx + 1}/{len(CURRICULUM)}: {stage_name}")
        print(f"Exploration steps: {exploration_steps}")
        print(f"Training steps: {stage_timesteps:,}")
        print("-" * 40)

        # Create parallel environments with current exploration setting
        print(f"Creating {N_ENVS} environments...")
        env_fns = [create_env(i, exploration_steps, render_mode=None) for i in range(N_ENVS)]
        vec_env = SubprocVecEnv(env_fns)
        vec_env = VecMonitor(vec_env)

        # Create evaluation environment for rendering
        print("Creating evaluation environment for rendering...")
        eval_env = create_env(0, exploration_steps, render_mode='human')()  # Create single env with rendering

        if model is None:
            # Create model (only first time)
            print("Creating DQN model...")
            model = DQN(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=dict(
                    features_extractor_class=NavigationCNNExtractor,
                    features_extractor_kwargs=dict(features_dim=256),
                    net_arch=[512, 512],
                ),
                # Hyperparameters
                learning_rate=1e-4,
                buffer_size=1_000_000,
                learning_starts=10000,
                batch_size=256,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                # Exploration
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                # Other
                max_grad_norm=10,
                seed=42,
                device='auto',
            )
            print(f"Model created on {model.device}")
        else:
            # Update environment for existing model
            model.set_env(vec_env)
            print(f"Model environment updated for stage {stage_idx + 1}")

        # Create callbacks
        callbacks = []

        # Progress callback
        progress_callback = NavigationCallback(n_envs=N_ENVS)
        callbacks.append(progress_callback)

        # Create evaluation environment function (not the environment itself)
        eval_env_fn = create_env(0, exploration_steps, render_mode='human')

        # Evaluation rendering callback - pass the function, not an instantiated environment
        eval_rendering_callback = EvaluationRenderingCallback(
            eval_env_fn=eval_env_fn,  # Pass the function
            eval_freq=500,  # Every 15 episodes
            n_eval_episodes=2  # Render 2 episodes
        )
        callbacks.append(eval_rendering_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500_000 // N_ENVS,
            save_path="./models/navigation_checkpoints/",
            name_prefix=f"nav_stage_{stage_idx + 1}_exp{exploration_steps}"
        )
        callbacks.append(checkpoint_callback)

        callback = CallbackList(callbacks)

        # Train
        print(f"\nTraining stage {stage_idx + 1}...")

        model.learn(
            total_timesteps=stage_timesteps,
            callback=callback,
            reset_num_timesteps=False,  # Continue from previous training
            progress_bar=False
        )

        # Stage complete
        stage_time = time.time() - stage_start
        stage_fps = stage_timesteps / stage_time

        print(f"\nStage {stage_idx + 1} complete!")
        print(f"  Time: {stage_time / 60:.1f} minutes")
        print(f"  Average FPS: {stage_fps:.0f}")

        # Save stage model
        model_path = f"./models/nav_stage_{stage_idx + 1}_exp{exploration_steps}"
        model.save(model_path)
        print(f"  Model saved: {model_path}.zip")

        vec_env.close()
        eval_env.close()  # Clean up evaluation environment

    # Save final model
    model.save("./models/dqn_navigation_final")

    print("\n" + "=" * 60)
    print("NAVIGATION TRAINING COMPLETE!")
    print(f"Final model: ./models/dqn_navigation_final.zip")
    print("=" * 60)


if __name__ == "__main__":
    # Verify map
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"MAP FILE NOT FOUND: {MAP_PATH}")

    print(f"Map file verified: {MAP_PATH}")

    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/navigation_checkpoints", exist_ok=True)

    train()