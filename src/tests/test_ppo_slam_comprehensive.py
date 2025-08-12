"""
test_ppo_slam_comprehensive.py

Comprehensive test suite to diagnose PPO learning issues in the SLAM environment.
This file tests all critical components that affect PPO training.
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# Import environment and dependencies
from environments.slam_env import MultiAgentSLAMEnv
from environments.constants import TileType, Action
from sensors.camera_sensor import CameraSensor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class SLAMEnvironmentTester:
    """Comprehensive tester for SLAM environment with PPO."""

    def __init__(self):
        self.test_results = []
        self.verbose = True

    def log(self, message, level="INFO"):
        """Log message with formatting."""
        prefix = "✓" if level == "PASS" else "✗" if level == "FAIL" else "→"
        print(f"{prefix} [{level}] {message}")
        self.test_results.append((level, message))

    def create_simple_env(self, num_agents=1, grid_size=5):
        """Create a very simple environment for testing."""
        env = MultiAgentSLAMEnv(
            width=grid_size,
            height=grid_size,
            num_agents=num_agents,
            max_steps=100,
            map_path=None,
            randomize=False,
            render_mode=None,
            sensor_config=None,
            default_sensor_params={'max_range': 3, 'fov_deg': 90, 'num_rays': 5},
            discovery_reward=1.0,  # High reward for testing
            collision_penalty=-0.5,
            step_penalty=-0.01,
            completion_bonus=10.0,
        )
        return env

    def test_1_action_space(self):
        """Test 1: Verify action space configuration."""
        self.log("=" * 60)
        self.log("TEST 1: ACTION SPACE VERIFICATION")
        self.log("=" * 60)

        try:
            # Test single agent
            env = self.create_simple_env(num_agents=1)
            obs, _ = env.reset()

            self.log(f"Single agent action space: {env.action_space}")
            self.log(f"Action space type: {type(env.action_space)}")
            self.log(f"Action space shape: {env.action_space.shape}")
            self.log(f"Action space nvec: {env.action_space.nvec}")

            # Verify it's MultiDiscrete
            from gymnasium.spaces import MultiDiscrete
            if isinstance(env.action_space, MultiDiscrete):
                self.log("Action space is MultiDiscrete", "PASS")
            else:
                self.log(f"Action space is NOT MultiDiscrete: {type(env.action_space)}", "FAIL")

            # Test sample actions
            for _ in range(5):
                action = env.action_space.sample()
                self.log(f"Sample action: {action}, shape: {action.shape}, dtype: {action.dtype}")

                # Verify action is valid
                obs, reward, term, trunc, info = env.step(action)
                self.log(f"Step executed successfully with reward: {reward:.3f}")

            env.close()

            # Test multi-agent
            env = self.create_simple_env(num_agents=3)
            obs, _ = env.reset()

            self.log(f"\nMulti-agent (3) action space: {env.action_space}")
            self.log(f"Action space shape: {env.action_space.shape}")
            self.log(f"Expected shape for 3 agents: (3,)")

            if env.action_space.shape == (3,):
                self.log("Multi-agent action space shape correct", "PASS")
            else:
                self.log(f"Multi-agent action space shape incorrect", "FAIL")

            env.close()

        except Exception as e:
            self.log(f"Action space test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_2_observation_space(self):
        """Test 2: Verify observation space and actual observations."""
        self.log("\n" + "=" * 60)
        self.log("TEST 2: OBSERVATION SPACE VERIFICATION")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=2, grid_size=5)
            obs, _ = env.reset()

            # Check observation space structure
            self.log(f"Observation space: {env.observation_space}")
            self.log(f"Observation space type: {type(env.observation_space)}")

            from gymnasium.spaces import Dict, Box
            if isinstance(env.observation_space, Dict):
                self.log("Observation space is Dict", "PASS")

                # Check each component
                for key, space in env.observation_space.spaces.items():
                    self.log(f"  '{key}': {space}")
                    if isinstance(space, Box):
                        self.log(f"    - Shape: {space.shape}, dtype: {space.dtype}")
                        self.log(f"    - Low: {space.low.min()}, High: {space.high.max()}")
            else:
                self.log("Observation space is NOT Dict", "FAIL")

            # Check actual observation
            self.log("\nActual observation structure:")
            for key, value in obs.items():
                self.log(f"  '{key}': shape={value.shape}, dtype={value.dtype}")
                self.log(f"    - Min: {value.min()}, Max: {value.max()}")

                # Verify shape matches space
                expected_shape = env.observation_space[key].shape
                if value.shape == expected_shape:
                    self.log(f"    - Shape matches space definition", "PASS")
                else:
                    self.log(f"    - Shape mismatch! Expected {expected_shape}", "FAIL")

            # Test observation consistency over steps
            self.log("\nTesting observation consistency over 10 steps:")
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)

                # Verify observation structure
                for key in env.observation_space.spaces.keys():
                    if key not in obs:
                        self.log(f"  Step {i}: Missing key '{key}' in observation", "FAIL")
                    else:
                        expected_shape = env.observation_space[key].shape
                        if obs[key].shape != expected_shape:
                            self.log(f"  Step {i}: Shape mismatch for '{key}'", "FAIL")

                if term or trunc:
                    obs, _ = env.reset()
                    self.log(f"  Episode reset at step {i}")

            self.log("Observation consistency maintained", "PASS")
            env.close()

        except Exception as e:
            self.log(f"Observation space test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_3_reward_structure(self):
        """Test 3: Analyze reward structure and distribution."""
        self.log("\n" + "=" * 60)
        self.log("TEST 3: REWARD STRUCTURE ANALYSIS")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=1, grid_size=5)

            # Collect rewards over multiple episodes
            all_rewards = []
            episode_rewards = []
            action_rewards = defaultdict(list)

            for episode in range(20):
                obs, _ = env.reset()
                episode_reward = 0
                step_rewards = []

                done = False
                steps = 0
                while not done and steps < 50:
                    action = env.action_space.sample()
                    obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc

                    all_rewards.append(reward)
                    step_rewards.append(reward)
                    episode_reward += reward
                    action_rewards[action[0]].append(reward)
                    steps += 1

                episode_rewards.append(episode_reward)

                # Analyze step rewards
                if step_rewards:
                    self.log(f"Episode {episode}: Total={episode_reward:.2f}, "
                            f"Steps={len(step_rewards)}, "
                            f"Avg={np.mean(step_rewards):.3f}, "
                            f"Progress={info.get('progress', 0)*100:.1f}%")

            # Statistical analysis
            self.log("\nReward Statistics:")
            self.log(f"  All rewards: Mean={np.mean(all_rewards):.3f}, "
                    f"Std={np.std(all_rewards):.3f}, "
                    f"Min={np.min(all_rewards):.3f}, "
                    f"Max={np.max(all_rewards):.3f}")
            self.log(f"  Episode rewards: Mean={np.mean(episode_rewards):.3f}, "
                    f"Std={np.std(episode_rewards):.3f}")

            # Check reward distribution by action
            self.log("\nReward by Action:")
            for action_id, rewards in action_rewards.items():
                action_name = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "STAY"][action_id]
                self.log(f"  {action_name}: Mean={np.mean(rewards):.3f}, "
                        f"Count={len(rewards)}, "
                        f"Positive={sum(r > 0 for r in rewards)}")

            # Check for reward issues
            if np.std(all_rewards) < 0.001:
                self.log("WARNING: Very low reward variance - agent may not learn", "FAIL")
            else:
                self.log("Reward variance appears sufficient", "PASS")

            if all(r <= 0 for r in all_rewards):
                self.log("WARNING: No positive rewards - agent unlikely to learn", "FAIL")
            else:
                positive_ratio = sum(r > 0 for r in all_rewards) / len(all_rewards)
                self.log(f"Positive reward ratio: {positive_ratio:.2%}", "PASS" if positive_ratio > 0.1 else "FAIL")

            env.close()

        except Exception as e:
            self.log(f"Reward structure test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_4_movement_and_sensing(self):
        """Test 4: Verify movement mechanics and sensing."""
        self.log("\n" + "=" * 60)
        self.log("TEST 4: MOVEMENT AND SENSING VERIFICATION")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=1, grid_size=5)
            obs, _ = env.reset()

            # Get initial position
            initial_pos = tuple(obs['positions'][0])
            self.log(f"Initial position: {initial_pos}")
            self.log(f"Initial facing: {obs['facings'][0]}")

            # Test each action type
            self.log("\nTesting movement actions:")

            # Test FORWARD
            action = np.array([Action.FORWARD])
            obs, reward, _, _, info = env.step(action)
            new_pos = tuple(obs['positions'][0])

            if new_pos != initial_pos:
                self.log(f"FORWARD: Moved from {initial_pos} to {new_pos}", "PASS")
            else:
                self.log(f"FORWARD: Position unchanged (might be blocked)", "INFO")

            # Test turning
            facing_before = obs['facings'][0]
            action = np.array([Action.TURN_LEFT])
            obs, reward, _, _, info = env.step(action)
            facing_after = obs['facings'][0]

            if facing_before != facing_after:
                self.log(f"TURN_LEFT: Facing changed from {facing_before} to {facing_after}", "PASS")
            else:
                self.log(f"TURN_LEFT: Facing unchanged", "FAIL")

            # Test map discovery
            self.log("\nTesting map discovery:")
            initial_unknown = np.sum(obs['global_map'] == TileType.UNKNOWN)

            # Move around to discover
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            final_unknown = np.sum(obs['global_map'] == TileType.UNKNOWN)
            discovered = initial_unknown - final_unknown

            self.log(f"Initial unknown cells: {initial_unknown}")
            self.log(f"Final unknown cells: {final_unknown}")
            self.log(f"Discovered cells: {discovered}")

            if discovered > 0:
                self.log("Map discovery working", "PASS")
            else:
                self.log("No map discovery detected", "FAIL")

            env.close()

        except Exception as e:
            self.log(f"Movement and sensing test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_5_ppo_compatibility(self):
        """Test 5: Check PPO compatibility and basic training."""
        self.log("\n" + "=" * 60)
        self.log("TEST 5: PPO COMPATIBILITY CHECK")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=1, grid_size=5)

            # Use SB3's environment checker
            self.log("Running Stable Baselines3 environment checker...")
            try:
                check_env(env, warn=True)
                self.log("Environment passes SB3 compatibility check", "PASS")
            except Exception as e:
                self.log(f"Environment fails SB3 check: {e}", "FAIL")

            # Create PPO model
            self.log("\nCreating PPO model...")
            vec_env = DummyVecEnv([lambda: env])

            model = PPO(
                "MultiInputPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=32,
                n_epochs=5,
                gamma=0.99,
                verbose=0
            )
            self.log("PPO model created successfully", "PASS")

            # Test prediction
            self.log("\nTesting PPO prediction...")
            obs = vec_env.reset()
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=False)
                self.log(f"  Predicted action: {action}, shape: {action.shape}, dtype: {action.dtype}")
                obs, rewards, dones, infos = vec_env.step(action)
            self.log("PPO prediction working", "PASS")

            # Quick training test - simplified without torch dependency
            self.log("\nTesting quick training (1000 steps)...")

            # Get initial performance
            eval_env = self.create_simple_env(num_agents=1, grid_size=5)
            obs, _ = eval_env.reset()
            initial_reward = 0
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = eval_env.step(action)
                initial_reward += reward
                if term or trunc:
                    break
            eval_env.close()

            # Train
            model.learn(total_timesteps=1000, progress_bar=False)

            # Get final performance
            eval_env = self.create_simple_env(num_agents=1, grid_size=5)
            obs, _ = eval_env.reset()
            final_reward = 0
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = eval_env.step(action)
                final_reward += reward
                if term or trunc:
                    break
            eval_env.close()

            self.log(f"Reward change: {initial_reward:.2f} -> {final_reward:.2f}")
            if final_reward > initial_reward - 1.0:  # Allow small degradation
                self.log("Model performance maintained or improved", "PASS")
            else:
                self.log("Model performance degraded significantly", "FAIL")

            vec_env.close()

        except Exception as e:
            self.log(f"PPO compatibility test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_6_normalized_observations(self):
        """Test 6: Check if observations need normalization."""
        self.log("\n" + "=" * 60)
        self.log("TEST 6: OBSERVATION NORMALIZATION CHECK")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=2, grid_size=10)

            # Collect observations
            all_obs = defaultdict(list)

            for _ in range(10):
                obs, _ = env.reset()
                for _ in range(50):
                    action = env.action_space.sample()
                    obs, _, term, trunc, _ = env.step(action)

                    for key, value in obs.items():
                        all_obs[key].append(value.flatten())

                    if term or trunc:
                        break

            # Analyze ranges
            self.log("Observation value ranges:")
            for key, values in all_obs.items():
                if values:
                    concat = np.concatenate(values)
                    self.log(f"  {key}:")
                    self.log(f"    Min: {concat.min():.2f}, Max: {concat.max():.2f}")
                    self.log(f"    Mean: {concat.mean():.2f}, Std: {concat.std():.2f}")

                    # Check if normalization might help
                    if key == 'global_map':
                        if concat.max() > 10 or concat.min() < -10:
                            self.log(f"    → Consider normalizing (large range)", "INFO")
                    elif key == 'positions':
                        if concat.max() > 100:
                            self.log(f"    → Consider normalizing positions", "INFO")

            env.close()

        except Exception as e:
            self.log(f"Normalization check failed: {e}", "FAIL")

    def test_7_exploration_behavior(self):
        """Test 7: Analyze exploration behavior."""
        self.log("\n" + "=" * 60)
        self.log("TEST 7: EXPLORATION BEHAVIOR ANALYSIS")
        self.log("=" * 60)

        try:
            env = self.create_simple_env(num_agents=1, grid_size=7)

            # Test random exploration
            self.log("Random exploration test:")
            obs, _ = env.reset()

            positions_visited = set()
            discoveries_per_step = []

            for step in range(100):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)

                pos = tuple(obs['positions'][0])
                positions_visited.add(pos)

                discovered = np.sum(obs['global_map'] != TileType.UNKNOWN)
                discoveries_per_step.append(discovered)

                if (step + 1) % 20 == 0:
                    self.log(f"  Step {step+1}: Visited {len(positions_visited)} positions, "
                            f"Discovered {discovered} cells, "
                            f"Progress {info.get('progress', 0)*100:.1f}%")

                if term or trunc:
                    break

            # Analyze exploration efficiency
            if len(discoveries_per_step) > 1:
                discovery_rate = np.diff(discoveries_per_step)
                avg_rate = np.mean(discovery_rate[discovery_rate > 0]) if any(discovery_rate > 0) else 0

                self.log(f"\nExploration metrics:")
                self.log(f"  Unique positions visited: {len(positions_visited)}")
                self.log(f"  Total cells discovered: {discoveries_per_step[-1]}")
                self.log(f"  Average discovery rate: {avg_rate:.2f} cells/step")

                if avg_rate < 0.1:
                    self.log("  WARNING: Very low discovery rate", "FAIL")

            env.close()

        except Exception as e:
            self.log(f"Exploration behavior test failed: {e}", "FAIL")

    def test_8_fixed_map_learning(self):
        """Test 8: Test learning on a fixed simple map."""
        self.log("\n" + "=" * 60)
        self.log("TEST 8: FIXED MAP LEARNING TEST")
        self.log("=" * 60)

        try:
            # Create a very simple fixed map
            simple_map = np.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 2, 0, 1],  # 2 is entry point
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ], dtype=np.int8)

            # Save map temporarily
            np.savetxt('test_map.txt', simple_map, fmt='%d')

            env = MultiAgentSLAMEnv(
                width=5,
                height=5,
                num_agents=1,
                max_steps=50,
                map_path='test_map.txt',
                randomize=False,
                discovery_reward=1.0,
                collision_penalty=-0.1,
                step_penalty=0.0,  # No step penalty for clearer signal
                completion_bonus=10.0,
                default_sensor_params={'max_range': 2, 'fov_deg': 90, 'num_rays': 5}
            )

            self.log("Testing on 5x5 fixed map with single agent")

            # Test environment
            obs, _ = env.reset()
            self.log(f"Starting position: {obs['positions'][0]}")
            self.log(f"Reachable cells: {env.total_reachable}")

            # Optimal sequence test
            self.log("\nTesting optimal action sequence:")
            optimal_actions = [
                Action.FORWARD,
                Action.TURN_LEFT,
                Action.FORWARD,
                Action.TURN_RIGHT,
                Action.FORWARD
            ]

            total_reward = 0
            for i, action_type in enumerate(optimal_actions):
                action = np.array([action_type])
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                self.log(f"  Step {i+1}: Action={action_type.name}, "
                        f"Reward={reward:.2f}, Progress={info['progress']*100:.1f}%")

            self.log(f"Total reward from optimal sequence: {total_reward:.2f}")

            if total_reward > 0:
                self.log("Environment provides positive feedback for exploration", "PASS")
            else:
                self.log("Environment not rewarding exploration properly", "FAIL")

            env.close()

            # Clean up
            import os
            if os.path.exists('test_map.txt'):
                os.remove('test_map.txt')

        except Exception as e:
            self.log(f"Fixed map learning test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def test_9_training_convergence(self):
        """Test 9: Extended training convergence test."""
        self.log("\n" + "=" * 60)
        self.log("TEST 9: TRAINING CONVERGENCE TEST")
        self.log("=" * 60)

        try:
            # Try importing torch to check parameters
            import torch
            torch_available = True
        except ImportError:
            torch_available = False
            self.log("PyTorch not available for parameter inspection", "INFO")

        try:
            # Create simple environment
            env = self.create_simple_env(num_agents=1, grid_size=5)
            vec_env = DummyVecEnv([lambda: env])

            # Create PPO model with aggressive learning settings
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                learning_rate=1e-3,  # Higher learning rate
                n_steps=64,
                batch_size=32,
                n_epochs=10,
                gamma=0.95,
                ent_coef=0.05,  # Higher entropy for exploration
                verbose=0
            )

            # Track training progress
            self.log("Training for 5000 steps with evaluation...")

            eval_rewards = []
            for i in range(5):
                # Train
                model.learn(total_timesteps=1000, progress_bar=False)

                # Evaluate
                eval_env = self.create_simple_env(num_agents=1, grid_size=5)
                obs, _ = eval_env.reset()
                total_reward = 0
                done = False
                steps = 0

                while not done and steps < 50:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, _ = eval_env.step(action)
                    total_reward += reward
                    done = term or trunc
                    steps += 1

                eval_rewards.append(total_reward)
                self.log(f"  Checkpoint {i+1}: Reward={total_reward:.2f}, Steps={steps}")
                eval_env.close()

            # Check for improvement
            if len(eval_rewards) > 1:
                improvement = eval_rewards[-1] - eval_rewards[0]
                self.log(f"\nTraining improvement: {improvement:.2f}")

                if improvement > 0:
                    self.log("Model shows improvement", "PASS")
                else:
                    self.log("No improvement detected - check reward structure", "FAIL")

            vec_env.close()

        except Exception as e:
            self.log(f"Training convergence test failed: {e}", "FAIL")
            import traceback
            traceback.print_exc()

    def run_all_tests(self):
        """Run all tests in sequence."""
        self.log("=" * 80)
        self.log("COMPREHENSIVE PPO SLAM ENVIRONMENT TEST SUITE")
        self.log("=" * 80)

        test_methods = [
            self.test_1_action_space,
            self.test_2_observation_space,
            self.test_3_reward_structure,
            self.test_4_movement_and_sensing,
            self.test_5_ppo_compatibility,
            self.test_6_normalized_observations,
            self.test_7_exploration_behavior,
            self.test_8_fixed_map_learning,
            self.test_9_training_convergence
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log(f"Test {test_method.__name__} crashed: {e}", "FAIL")

        # Summary
        self.log("\n" + "=" * 80)
        self.log("TEST SUMMARY")
        self.log("=" * 80)

        passes = sum(1 for level, _ in self.test_results if level == "PASS")
        fails = sum(1 for level, _ in self.test_results if level == "FAIL")
        infos = sum(1 for level, _ in self.test_results if level == "INFO")

        self.log(f"Total PASS: {passes}")
        self.log(f"Total FAIL: {fails}")
        self.log(f"Total INFO: {infos}")

        if fails > 0:
            self.log("\n⚠️  CRITICAL ISSUES FOUND - Review FAIL items above")
            self.log("\nCommon issues that prevent learning:")
            self.log("1. Reward signal too sparse or always negative")
            self.log("2. Observation space not properly normalized")
            self.log("3. Action effects not observable in state")
            self.log("4. Environment too difficult for initial exploration")
            self.log("5. Bug in movement or sensing mechanics")
        else:
            self.log("\n✅ All tests passed - environment should be trainable")

    def analyze_specific_issue(self, issue_type="reward"):
        """Deep dive into specific issues."""
        if issue_type == "reward":
            self.log("\n" + "=" * 60)
            self.log("DEEP DIVE: REWARD ANALYSIS")
            self.log("=" * 60)

            env = self.create_simple_env(num_agents=1, grid_size=5)

            # Analyze reward components
            env.discovery_reward = 1.0
            env.collision_penalty = -0.5
            env.step_penalty = -0.01
            env.completion_bonus = 10.0

            obs, _ = env.reset()

            self.log("Testing reward components individually:")

            # Test discovery reward
            initial_discovered = np.sum(obs['global_map'] != TileType.UNKNOWN)

            # Move to discover new cells
            for _ in range(10):
                action = np.array([Action.FORWARD])
                obs, reward, _, _, _ = env.step(action)
                new_discovered = np.sum(obs['global_map'] != TileType.UNKNOWN)

                if new_discovered > initial_discovered:
                    self.log(f"  Discovery: {new_discovered - initial_discovered} cells, "
                            f"Reward: {reward:.3f}")
                    break

                # Try turning if blocked
                action = np.array([Action.TURN_RIGHT])
                env.step(action)

            env.close()


def test_slam_environment():
    """Pytest-compatible test function."""
    tester = SLAMEnvironmentTester()
    tester.run_all_tests()

    # Check for failures
    fails = sum(1 for level, _ in tester.test_results if level == "FAIL")
    assert fails == 0, f"Found {fails} test failures"


def main():
    """Run as standalone script."""
    print("\n🔍 Starting Comprehensive SLAM Environment Diagnostic...\n")

    tester = SLAMEnvironmentTester()

    # Run all tests
    tester.run_all_tests()

    # Optional: Deep dive into specific issues
    # tester.analyze_specific_issue("reward")

    print("\n✅ Diagnostic complete. Check results above for issues.")
    print("\nRECOMMENDATIONS:")
    print("1. If rewards are always negative: Increase discovery_reward, reduce penalties")
    print("2. If no exploration: Increase entropy coefficient (ent_coef)")
    print("3. If observations are large: Consider normalizing or using VecNormalize wrapper")
    print("4. If map is too complex: Start with smaller maps (5x5 or 7x7)")
    print("5. If sensor range too small: Increase max_range to at least 3-5 cells")


if __name__ == "__main__":
    main()