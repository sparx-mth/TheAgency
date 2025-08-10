"""
Stable Baselines3 Training Script for Multi-Agent SLAM Environment

This script provides a clean implementation using Stable Baselines3 with support
for multiple RL algorithms (PPO, A2C, DQN, SAC, etc.)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from planner.simulation.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv


class SLAMTrainingConfig:
    """Configuration class for training parameters."""

    def __init__(
        self,
        # Environment parameters
        map_path: str,
        eval_map_path: Optional[str] = None,
        width: Optional[int] = None,  # Made optional - will be determined from map
        height: Optional[int] = None,  # Made optional - will be determined from map
        num_drones: int = 3,
        camera_range: int = 3,
        fov: int = 90,
        max_steps: int = 1000,

        # Training parameters
        algorithm: str = "PPO",  # PPO, A2C, DQN, SAC
        total_timesteps: int = 100_000,
        n_envs: int = 4,  # Number of parallel environments

        # Model parameters
        policy: str = "MlpPolicy",  # or "CnnPolicy" for image observations
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,  # For PPO/A2C
        gamma: float = 0.99,

        # Evaluation parameters
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,

        # Saving parameters
        save_freq: int = 10_000,
        log_dir: str = "./logs",
        save_dir: str = "./models",

        # Other parameters
        seed: int = 42,
        device: str = "auto",
        verbose: int = 1
    ):
        # Convert relative paths to absolute paths
        self.map_path = self._get_absolute_path(map_path)
        self.eval_map_path = self._get_absolute_path(eval_map_path) if eval_map_path else self.map_path

        # Load map dimensions if not provided
        if width is None or height is None:
            self.height, self.width = self._get_map_dimensions(self.map_path)
        else:
            self.width = width
            self.height = height

        self.num_drones = num_drones
        self.camera_range = camera_range
        self.fov = fov
        self.max_steps = max_steps

        self.algorithm = algorithm.upper()
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs

        self.policy = policy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma

        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_dir = save_dir

        self.seed = seed
        self.device = device
        self.verbose = verbose

        # Create unique run identifier
        self.run_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Update paths with run_id
        self.log_path = Path(log_dir) / self.run_id
        self.save_path = Path(save_dir) / self.run_id

        # Create directories
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _get_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path."""
        if os.path.isabs(path):
            return path

        # Try different base directories
        possible_bases = [
            os.getcwd(),  # Current working directory
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # Project root
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources"),
        ]

        for base in possible_bases:
            full_path = os.path.join(base, path)
            if os.path.exists(full_path):
                return full_path

        # If not found, return original path (will error later with clear message)
        return path

    def _get_map_dimensions(self, map_path: str) -> Tuple[int, int]:
        """Load map dimensions from file."""
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")

        map_data = np.loadtxt(map_path, dtype=np.int8)
        return map_data.shape


class GymnasiumAdapter:
    """
    Adapter to make old gym environment compatible with gymnasium interface.
    This is not a gym.Wrapper but a standalone adapter.
    """

    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, 'metadata', {'render_modes': ['human', 'rgb_array']})
        self.render_mode = getattr(env, 'render_mode', None)
        self.spec = getattr(env, 'spec', None)

        # Convert spaces if needed
        self.observation_space = self._convert_space(env.observation_space) if hasattr(env, 'observation_space') else None
        self.action_space = self._convert_space(env.action_space) if hasattr(env, 'action_space') else None

    def _convert_space(self, space):
        """Convert gym.Space to gymnasium.Space."""
        # Check if it's already a gymnasium space
        if hasattr(space, '__module__') and 'gymnasium' in space.__module__:
            return space

        # Import old gym to check types
        try:
            import gym as old_gym

            if isinstance(space, old_gym.spaces.Box):
                return spaces.Box(
                    low=space.low,
                    high=space.high,
                    shape=space.shape,
                    dtype=space.dtype
                )
            elif isinstance(space, old_gym.spaces.Discrete):
                return spaces.Discrete(space.n)
            elif isinstance(space, old_gym.spaces.MultiDiscrete):
                return spaces.MultiDiscrete(space.nvec)
            elif isinstance(space, old_gym.spaces.Dict):
                return spaces.Dict({
                    key: self._convert_space(subspace)
                    for key, subspace in space.spaces.items()
                })
        except ImportError:
            pass

        # If we can't convert, return as-is
        return space

    def reset(self, **kwargs):
        """Reset with gymnasium interface."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            # Old gym interface returns only observation
            return result, {}

    def step(self, action):
        """Step with gymnasium interface."""
        result = self.env.step(action)
        if len(result) == 4:
            # Old gym interface: obs, reward, done, info
            obs, reward, done, info = result
            return obs, reward, done, False, info  # Add truncated=False
        elif len(result) == 5:
            # Already gymnasium interface
            return result
        else:
            raise ValueError(f"Unexpected step result format: {len(result)} values")

    def render(self):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        """Forward any other attributes to the wrapped environment."""
        return getattr(self.env, name)


class SingleAgentSLAMWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper to convert multi-agent environment to single-agent for Stable Baselines3.

    This wrapper controls a single drone at a time while keeping others inactive,
    or can implement a centralized controller for all drones.
    """

    def __init__(self, env, mode: str = "single"):
        """
        Args:
            env: Multi-agent SLAM environment (already adapted to gymnasium)
            mode: "single" for single drone, "centralized" for controlling all drones
        """
        super().__init__()
        self.env = env
        self.mode = mode
        self.active_agent_id = 0  # For single mode

        # Get base environment (unwrap if needed)
        base_env = env.env if hasattr(env, 'env') else env

        # Get number of drones
        self.num_drones = getattr(base_env, 'num_drones', 3)

        # Copy metadata
        self.metadata = getattr(env, 'metadata', {'render_modes': ['human', 'rgb_array']})
        self.render_mode = getattr(env, 'render_mode', None)
        self.spec = getattr(env, 'spec', None)

        if mode == "single":
            # Use single drone's observation and action space
            if hasattr(base_env, 'observation_spaces'):
                base_obs_space = base_env.observation_spaces[0]
            else:
                # Fallback for different environment versions
                base_obs_space = self._create_default_obs_space()

            # Calculate the actual observation size by creating a dummy observation
            # This ensures we get the exact size
            dummy_obs = {
                'local_map': np.zeros((getattr(base_env, 'height', 10),
                                      getattr(base_env, 'width', 10)), dtype=np.int8),
                'position': np.array([0, 0]),
                'facing_direction': 0,
                'active': 1,
                'collided': 0,
                'entry_time': np.array([0]),
                'new_discoveries': np.array([0])
            }

            # Get the actual flattened size
            dummy_flat = self._flatten_observation(dummy_obs)
            obs_size = len(dummy_flat)

            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(obs_size,), dtype=np.float32
            )

            if hasattr(base_env, 'action_spaces'):
                base_action_space = base_env.action_spaces[0]
                # Convert to gymnasium Discrete if needed
                if hasattr(base_action_space, 'n'):
                    self.action_space = spaces.Discrete(base_action_space.n)
                else:
                    self.action_space = base_action_space
            else:
                self.action_space = spaces.Discrete(4)  # Default: 4 actions

        elif mode == "centralized":
            # Flatten all observations and actions

            # Calculate total observation size
            if hasattr(base_env, 'observation_spaces'):
                single_obs_size = self._calculate_obs_size(base_env.observation_spaces[0])
            else:
                single_obs_size = self._calculate_obs_size(self._create_default_obs_space())

            total_obs_size = single_obs_size * self.num_drones

            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(total_obs_size,), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([4] * self.num_drones)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _create_default_obs_space(self):
        """Create default observation space structure."""
        return {
            'local_map': spaces.Box(low=-1, high=6, shape=(32, 32), dtype=np.int8),
            'position': spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32),
            'facing_direction': spaces.Discrete(4),
            'active': spaces.Discrete(2),
            'collided': spaces.Discrete(2),
            'entry_time': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'new_discoveries': spaces.Box(low=-1, high=1000, shape=(1,), dtype=np.int32)
        }

    def _calculate_obs_size(self, obs_space):
        """Calculate flattened observation size."""
        if isinstance(obs_space, dict):
            # Dictionary space - sum all components
            total = 0
            if 'local_map' in obs_space:
                total += np.prod(obs_space['local_map'].shape)
            # Add other features
            total += 2  # position (x, y)
            total += 1  # facing_direction
            total += 1  # active
            total += 1  # collided
            total += 1  # entry_time
            total += 1  # new_discoveries
            return int(total)
        elif isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        else:
            # Default fallback - assuming 10x10 map + 7 features
            return 107

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return observation for the wrapper."""
        super().reset(seed=seed)
        obs_dict, info = self.env.reset(**{'seed': seed, 'options': options} if seed or options else {})
        return self._process_observation(obs_dict, info), info

    def step(self, action):
        """Step the environment with the wrapper's action format."""
        # Convert action to multi-agent format
        actions = self._process_action(action)

        # Step the underlying environment
        obs_dict, rewards, dones, truncated, info = self.env.step(actions)

        # Process outputs for single-agent interface
        obs = self._process_observation(obs_dict, info)
        reward = self._process_reward(rewards)
        done = self._process_done(dones)
        trunc = self._process_truncated(truncated)

        return obs, reward, done, trunc, info

    def _process_observation(self, obs_dict: Dict, info: Dict) -> np.ndarray:
        """Convert multi-agent observations to single-agent format."""
        if self.mode == "single":
            # Return observation for active agent
            obs = obs_dict[self.active_agent_id]
            # Flatten the observation
            return self._flatten_observation(obs)
        elif self.mode == "centralized":
            # Concatenate all observations
            all_obs = []
            for agent_id in range(self.num_drones):
                if agent_id in obs_dict:
                    all_obs.append(self._flatten_observation(obs_dict[agent_id]))
                else:
                    # Padding for inactive agents
                    all_obs.append(np.zeros(self._get_obs_size()))
            return np.concatenate(all_obs)

    def _flatten_observation(self, obs: Dict) -> np.ndarray:
        """Flatten a single agent's observation."""
        # Get the local map
        local_map = obs['local_map'].astype(np.float32)

        # Normalize map values to [-1, 1]
        local_map = np.clip(local_map / 3.0, -1.0, 1.0)
        flat_map = local_map.flatten()

        # Extract and normalize other features
        features = []

        # Position (normalized to [0, 1])
        if 'position' in obs:
            features.append(obs['position'][0] / 100.0)
            features.append(obs['position'][1] / 100.0)

        # Facing direction (normalized to [0, 1])
        if 'facing_direction' in obs:
            # Handle both direct value and array format
            facing = obs['facing_direction']
            if hasattr(facing, '__len__'):
                facing = facing[0] if len(facing) > 0 else 0
            features.append(float(facing) / 3.0)

        # Active flag
        if 'active' in obs:
            active = obs['active']
            if hasattr(active, '__len__'):
                active = active[0] if len(active) > 0 else 0
            features.append(float(active))

        # Collided flag
        if 'collided' in obs:
            collided = obs['collided']
            if hasattr(collided, '__len__'):
                collided = collided[0] if len(collided) > 0 else 0
            features.append(float(collided))

        # Entry time (normalized)
        if 'entry_time' in obs:
            entry_time = obs['entry_time']
            if hasattr(entry_time, '__len__'):
                entry_time = entry_time[0] if len(entry_time) > 0 else 0
            features.append(float(entry_time) / 1000.0)

        # New discoveries (normalized)
        if 'new_discoveries' in obs:
            discoveries = obs['new_discoveries']
            if hasattr(discoveries, '__len__'):
                discoveries = discoveries[0] if len(discoveries) > 0 else 0
            features.append(np.clip(float(discoveries) / 100.0, 0, 1))

        # Concatenate everything
        features_array = np.array(features, dtype=np.float32)
        result = np.concatenate([flat_map, features_array])

        return result

    def _get_obs_size(self) -> int:
        """Get the size of a flattened observation."""
        return self.observation_space.shape[0] // (self.num_drones if self.mode == "centralized" else 1)

    def _process_action(self, action) -> Dict[int, int]:
        """Convert single-agent action to multi-agent format."""

        if self.mode == "single":
            # Only control active agent, others stay
            return {
                agent_id: int(action) if agent_id == self.active_agent_id else 3  # STAY
                for agent_id in range(self.num_drones)
            }
        elif self.mode == "centralized":
            # Each agent gets its corresponding action
            return {
                agent_id: int(action[agent_id])
                for agent_id in range(self.num_drones)
            }

    def _process_reward(self, rewards: Dict[int, float]) -> float:
        """Aggregate multi-agent rewards."""
        if self.mode == "single":
            return rewards.get(self.active_agent_id, 0.0)
        elif self.mode == "centralized":
            # Sum of all rewards (you can also use mean or other aggregation)
            return sum(rewards.values())

    def _process_done(self, dones: Dict[int, bool]) -> bool:
        """Aggregate multi-agent done signals."""
        if self.mode == "single":
            return dones.get(self.active_agent_id, False)
        elif self.mode == "centralized":
            # Episode ends when all agents are done
            return all(dones.values())

    def _process_truncated(self, truncated: Dict[int, bool]) -> bool:
        """Aggregate multi-agent truncated signals."""
        if self.mode == "single":
            return truncated.get(self.active_agent_id, False)
        elif self.mode == "centralized":
            return any(truncated.values())


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_progress = []

    def _on_step(self) -> bool:
        # Log additional metrics from info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "exploration_progress" in info:
                    self.logger.record("env/exploration_progress", info["exploration_progress"])
                if "drone_discoveries" in info:
                    total_discoveries = sum(info["drone_discoveries"].values())
                    self.logger.record("env/total_discoveries", total_discoveries)
        return True

    def _on_rollout_end(self) -> None:
        # Log episode statistics
        if len(self.episode_rewards) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
            self.episode_rewards.clear()
            self.episode_lengths.clear()


class SLAMTrainer:
    """Main trainer class for SLAM environment using Stable Baselines3."""

    def __init__(self, config: SLAMTrainingConfig):
        self.config = config
        self.model = None
        self.env = None
        self.eval_env = None

    def create_env(self, map_path: str, n_envs: int = 1) -> gym.Env:
        """Create vectorized environment."""
        # Verify map exists
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}\nCurrent directory: {os.getcwd()}")

        # Load map dimensions
        map_data = np.loadtxt(map_path, dtype=np.int8)
        height, width = map_data.shape

        def make_env(rank: int):
            def _init():
                # Create multi-agent environment
                multi_env = MultiAgentSLAMGymEnv(
                    width=width,
                    height=height,
                    num_drones=self.config.num_drones,
                    num_entry_points=self.config.num_drones,
                    camera_range=self.config.camera_range,
                    fov=self.config.fov,
                    max_steps=self.config.max_steps,
                    render_mode=None,
                    randomize=False,
                    map_path=map_path
                )

                # Debug: Print environment info
                if rank == 0 and self.config.verbose > 0:
                    print(f"Created MultiAgentSLAMGymEnv: {type(multi_env)}")

                # Wrap for gymnasium compatibility if needed
                multi_env = GymnasiumAdapter(multi_env)

                # Debug: Print after adapter
                if rank == 0 and self.config.verbose > 0:
                    print(f"After GymnasiumAdapter: {type(multi_env)}")

                # Wrap for single-agent interface
                single_env = SingleAgentSLAMWrapper(multi_env, mode="single")

                # Debug: Print wrapper info
                if rank == 0 and self.config.verbose > 0:
                    print(f"SingleAgentSLAMWrapper created: {type(single_env)}")
                    print(f"Observation space: {single_env.observation_space}")
                    print(f"Action space: {single_env.action_space}")

                # Add monitor wrapper for logging
                single_env = Monitor(single_env)

                return single_env
            return _init

        if n_envs == 1:
            return DummyVecEnv([make_env(0)])
        else:
            # Use DummyVecEnv for now to avoid multiprocessing issues
            # You can switch to SubprocVecEnv once environment is stable
            return DummyVecEnv([make_env(i) for i in range(n_envs)])

    def get_algorithm(self):
        """Get the RL algorithm class based on config."""
        algorithms = {
            "PPO": PPO,
            "A2C": A2C,
            "DQN": DQN,
            "SAC": SAC
        }

        if self.config.algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        return algorithms[self.config.algorithm]

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get algorithm-specific parameters."""
        common_params = {
            "policy": self.config.policy,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "verbose": self.config.verbose,
            "device": self.config.device,
            "seed": self.config.seed,
            "tensorboard_log": str(self.config.log_path / "tensorboard")
        }

        # Algorithm-specific parameters
        if self.config.algorithm in ["PPO", "A2C"]:
            common_params["n_steps"] = self.config.n_steps

        if self.config.algorithm == "PPO":
            common_params["batch_size"] = self.config.batch_size
            common_params["n_epochs"] = 10
            common_params["clip_range"] = 0.2

        if self.config.algorithm == "DQN":
            common_params["batch_size"] = self.config.batch_size
            common_params["buffer_size"] = 50_000
            common_params["exploration_fraction"] = 0.1
            common_params["exploration_initial_eps"] = 1.0
            common_params["exploration_final_eps"] = 0.05

        if self.config.algorithm == "SAC":
            common_params["batch_size"] = self.config.batch_size
            common_params["buffer_size"] = 50_000

        return common_params

    def train(self):
        """Main training function."""
        print("=" * 80)
        print(f"TRAINING WITH {self.config.algorithm}")
        print("=" * 80)
        print(f"Training map: {self.config.map_path}")
        print(f"Evaluation map: {self.config.eval_map_path}")
        print(f"Map dimensions: {self.config.width}x{self.config.height}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Number of environments: {self.config.n_envs}")
        print(f"Run ID: {self.config.run_id}")
        print("=" * 80)

        # Create environments
        print("\nCreating training environments...")
        self.env = self.create_env(self.config.map_path, self.config.n_envs)

        print("Creating evaluation environment...")
        self.eval_env = self.create_env(self.config.eval_map_path, n_envs=1)

        # Get algorithm class and parameters
        AlgorithmClass = self.get_algorithm()
        algorithm_params = self.get_algorithm_params()

        # Create model
        print(f"\nInitializing {self.config.algorithm} model...")
        self.model = AlgorithmClass(env=self.env, **algorithm_params)

        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Train
        print("\nStarting training...")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )

        # Save final model
        final_model_path = self.config.save_path / "final_model"
        self.model.save(final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}")

        # Final evaluation
        print("\nRunning final evaluation...")
        mean_reward, std_reward = self.evaluate(n_eval_episodes=20)
        print(f"Final performance: {mean_reward:.2f} ± {std_reward:.2f}")

        return self.model

    def _setup_callbacks(self) -> CallbackList:
        """Setup training callbacks."""
        callbacks = []

        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.config.save_path / "best_model"),
            log_path=str(self.config.log_path / "evaluations"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.config.save_path / "checkpoints"),
            name_prefix=f"{self.config.algorithm}_checkpoint",
            save_replay_buffer=True if self.config.algorithm in ["DQN", "SAC"] else False,
            save_vecnormalize=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        # Custom tensorboard callback
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)

        return CallbackList(callbacks)

    def evaluate(
        self,
        model_path: Optional[str] = None,
        n_eval_episodes: int = 10,
        render: bool = False
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        # Load model if path provided
        if model_path:
            AlgorithmClass = self.get_algorithm()
            model = AlgorithmClass.load(model_path, env=self.eval_env)
        else:
            model = self.model

        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model,
            self.eval_env,
            n_eval_episodes=n_eval_episodes,
            render=render,
            deterministic=True,
            return_episode_rewards=False
        )

        return mean_reward, std_reward

    def visualize_episode(
        self,
        model_path: Optional[str] = None,
        map_path: Optional[str] = None
    ):
        """Visualize a single episode with rendering."""
        print("\n" + "=" * 80)
        print("EPISODE VISUALIZATION")
        print("=" * 80)

        # Use provided map or default to eval map
        map_to_use = map_path or self.config.eval_map_path

        # Load map dimensions
        map_data = np.loadtxt(map_to_use, dtype=np.int8)
        height, width = map_data.shape

        # Create environment with rendering
        multi_env = MultiAgentSLAMGymEnv(
            width=width,
            height=height,
            num_drones=self.config.num_drones,
            num_entry_points=self.config.num_drones,
            camera_range=self.config.camera_range,
            fov=self.config.fov,
            max_steps=self.config.max_steps,
            render_mode='human',
            randomize=False,
            map_path=map_to_use
        )

        env = SingleAgentSLAMWrapper(multi_env, mode="single")

        # Load model
        if model_path:
            AlgorithmClass = self.get_algorithm()
            model = AlgorithmClass.load(model_path)
        else:
            model = self.model

        # Run episode
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        print("Running episode... (close window to stop)")

        while steps < self.config.max_steps:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Render
            env.render()

            # Check termination
            if done or truncated:
                print(f"Episode finished!")
                break

            # Print progress periodically
            if steps % 50 == 0:
                print(f"Step {steps}: Progress = {info['exploration_progress']:.1%}")

        print(f"\nEpisode Summary:")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final exploration: {info['exploration_progress']:.1%}")

        # Keep window open
        import time
        print("\nKeeping window open for 5 seconds...")
        for _ in range(50):
            env.render()
            time.sleep(0.1)

        env.close()


def main():
    """Main function to run training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agents for Multi-Agent SLAM")
    parser.add_argument("--algorithm", type=str, default="DQN",
                       choices=["PPO", "A2C", "DQN", "SAC"],
                       help="RL algorithm to use")
    parser.add_argument("--train-map", type=str,
                       default="/home/user/nadav/TheAgency/resources/planner/maps/house_map_10.txt",
                       help="Path to training map (relative to resources/ or absolute)")
    parser.add_argument("--eval-map", type=str,
                       default=None,
                       help="Path to evaluation map (defaults to train-map)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--num-drones", type=int, default=1,
                       help="Number of drones in the environment")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize after training")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to load a pretrained model")

    args = parser.parse_args()

    # If eval map not specified, use train map
    if args.eval_map is None:
        args.eval_map = args.train_map

    # Create configuration
    config = SLAMTrainingConfig(
        map_path=args.train_map,
        eval_map_path=args.eval_map,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        num_drones=args.num_drones,
        camera_range=3,
        fov=90,
        max_steps=1000
    )

    # Create trainer
    trainer = SLAMTrainer(config)

    if args.load_model:
        # Just visualize with loaded model
        trainer.visualize_episode(model_path=args.load_model)
    else:
        # Train
        model = trainer.train()

        # Visualize if requested
        if args.visualize:
            trainer.visualize_episode()


if __name__ == "__main__":
    main()