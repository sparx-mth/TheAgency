"""
Comprehensive test suite for RoomEntryWrapper environment.
Tests all aspects including base environment interface, rewards, termination, and SB3 compatibility.
"""

import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from typing import Dict, Tuple
import os
import time

# Import your modules - adjust paths as needed
from environments.tasks.room_entry_wrapper import RoomEntryWrapper
from environments.tasks.doorway_utils import precompute_doorways
from environments.base.constants import TileType, Action


class TestRoomEntryEnvironment:
    """Comprehensive test suite for Room Entry Environment."""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        cls.map_path = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

        # Pre-compute doorways once for all tests
        if os.path.exists(cls.map_path):
            cls.precomputed_doorways = precompute_doorways(cls.map_path)
        else:
            # Fallback for testing without the specific map
            cls.precomputed_doorways = {(5, 5): 'horizontal', (10, 10): 'vertical'}

        cls.default_config = {
            'width': 32,
            'height': 32,
            'num_agents': 1,
            'max_steps': 500,
            'map_path': cls.map_path if os.path.exists(cls.map_path) else None,
            'randomize': False if os.path.exists(cls.map_path) else True,
            'render_mode': None
        }

    def create_env(self, **kwargs):
        """Helper to create environment with custom parameters."""
        # Separate base environment config from wrapper parameters
        base_env_params = ['width', 'height', 'num_agents', 'max_steps', 'map_path', 'randomize', 'render_mode']

        # Extract base environment config
        env_config = {}
        for key in base_env_params:
            if key in self.default_config:
                env_config[key] = self.default_config[key]
            if key in kwargs:
                env_config[key] = kwargs.pop(key)

        # Remaining kwargs are wrapper-specific parameters
        return RoomEntryWrapper(
            env_config=env_config,
            precomputed_doorways=self.precomputed_doorways,
            **kwargs  # Only wrapper-specific params now
        )

    # ============== Basic Environment Tests ==============

    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        env = self.create_env()

        assert env is not None
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'all_doorways')
        assert len(env.all_doorways) == len(self.precomputed_doorways)

        # Check doorway data structures
        assert isinstance(env.all_doorways, dict)
        assert isinstance(env.all_doorways_array, np.ndarray)
        assert len(env.doorway_orientations) == len(env.all_doorways)

        env.close()

    def test_action_space(self):
        """Test action space is correct for single agent DQN."""
        env = self.create_env()

        # Should be Discrete for single agent
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4  # FORWARD, TURN_LEFT, TURN_RIGHT, STAY

        # Test sampling
        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < 4
            assert isinstance(action, (int, np.integer))

        env.close()

    def test_observation_space(self):
        """Test observation space structure."""
        env = self.create_env()
        obs, info = env.reset()

        # Check observation structure
        assert isinstance(obs, dict)
        assert 'global_map' in obs
        assert 'positions' in obs
        assert 'facings' in obs
        assert 'active' in obs

        # Check shapes
        assert obs['global_map'].shape == (env.env.height, env.env.width)
        assert obs['positions'].shape == (1, 2)  # Single agent
        assert obs['facings'].shape == (1,)
        assert obs['active'].shape == (1,)

        # Check types
        assert obs['global_map'].dtype == np.int8
        assert obs['positions'].dtype == np.int32
        assert obs['facings'].dtype == np.int32
        assert obs['active'].dtype == np.int8

        # Check observation is in space
        assert env.observation_space.contains(obs)

        env.close()

    # ============== Reset and Initialization Tests ==============

    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        env = self.create_env(auto_explore=False)

        # Multiple resets should work
        for _ in range(5):
            obs, info = env.reset()

            # Check initial state
            assert env.task_step == 0
            assert env.task_status == 0  # IN_PROGRESS
            assert env.target_doorway is None  # No target until doorway found
            assert not env.has_passed_through
            assert not env.is_exploring  # auto_explore is False

            # Check observations are valid
            assert env.observation_space.contains(obs)

        env.close()

    def test_auto_exploration_on_reset(self):
        """Test auto-exploration functionality during reset."""
        env = self.create_env(
            auto_explore=True,
            max_exploration_steps=100,
            min_doorways_to_discover=1
        )

        obs, info = env.reset()

        # After auto-exploration
        assert not env.is_exploring  # Should be done exploring
        assert env.exploration_steps > 0  # Should have taken some steps

        # Check if doorways were discovered
        if len(env.discovered_doorway_indices) > 0:
            assert env.target_doorway is not None  # Should have selected a target
            assert env.target_doorway_idx is not None
            assert env.initial_distance is not None

        # Info should contain exploration data
        assert 'exploration_steps' in info
        assert 'discovered_doorways' in info

        env.close()

    # ============== Step Mechanics Tests ==============

    def test_step_action_conversion(self):
        """Test that single agent actions are properly converted."""
        env = self.create_env(auto_explore=False)
        obs, info = env.reset()

        # Test different action formats
        action_formats = [
            0,  # Plain integer
            np.int32(1),  # Numpy integer
            np.array(2),  # Scalar array
            np.array([3]),  # 1D array
        ]

        for action in action_formats:
            obs, reward, terminated, truncated, info = env.step(action)

            # Should not crash and return valid data
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert env.observation_space.contains(obs)

        env.close()

    def test_step_counter_accuracy(self):
        """Test that step counting is accurate and starts correctly."""
        env = self.create_env(
            auto_explore=True,
            max_exploration_steps=50,
            max_task_steps=100
        )

        obs, info = env.reset()

        # Task step should be 0 after reset (not counting exploration)
        assert env.task_step == 0

        # Exploration steps should be recorded separately
        exploration_steps_taken = env.exploration_steps
        assert exploration_steps_taken > 0

        # Take some steps and verify counting
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            assert env.task_step == i + 1  # Task steps start from 1
            assert info['task_step'] == i + 1

            if terminated or truncated:
                break

        env.close()

    # ============== Doorway Discovery Tests ==============

    def test_doorway_discovery_mechanism(self):
        """Test that doorways are discovered correctly."""
        env = self.create_env(auto_explore=False)
        obs, info = env.reset()

        initial_discovered = len(env.discovered_doorway_indices)

        # Simulate exploration to discover doorways
        max_steps = 200
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Check doorway discovery
            env._fast_check_doorways()

            if len(env.discovered_doorway_indices) > initial_discovered:
                # Doorway discovered!
                assert env.doorway_visible[env.discovered_doorway_indices[-1]]

                # Should select target if not already selected
                if env.target_doorway is None:
                    env._select_target_doorway_fast()
                    if len(env.discovered_doorway_indices) > 0:
                        assert env.target_doorway is not None
                break

            if terminated or truncated:
                break

        env.close()

    def test_target_doorway_selection(self):
        """Test that nearest doorway is selected as target."""
        env = self.create_env(auto_explore=False)
        obs, info = env.reset()

        # Manually mark some doorways as discovered
        if len(env.all_doorways) >= 2:
            env.discovered_doorway_indices = [0, 1]
            env.doorway_visible[0] = True
            env.doorway_visible[1] = True

            # Select target
            env._select_target_doorway_fast()

            # Should have selected the nearest one
            assert env.target_doorway is not None
            assert env.target_doorway_idx in [0, 1]

            # Verify it's actually the nearest
            drone_pos = env.env.drones[0].pos
            selected_dist = abs(env.target_doorway[0] - drone_pos[0]) + \
                          abs(env.target_doorway[1] - drone_pos[1])

            for idx in [0, 1]:
                door_pos = tuple(env.all_doorways_array[idx])
                dist = abs(door_pos[0] - drone_pos[0]) + abs(door_pos[1] - drone_pos[1])
                assert dist >= selected_dist - 0.1  # Account for floating point

        env.close()

    # ============== Reward Structure Tests ==============

    def test_reward_structure(self):
        """Test that rewards are computed correctly."""
        env = self.create_env(
            auto_explore=False,
            success_reward=10.0,
            progress_reward=0.5,
            collision_penalty=-0.5,
            step_penalty=-0.01
        )

        obs, info = env.reset()

        # Test step penalty
        obs, reward, _, _, _ = env.step(Action.STAY)
        assert reward <= 0  # Should have step penalty
        assert abs(reward - (-0.01)) < 0.1  # Approximately step penalty

        # Test collision penalty
        # Move into a wall repeatedly to trigger collision
        for _ in range(10):
            old_pos = obs['positions'][0].copy()
            obs, reward, _, _, _ = env.step(Action.FORWARD)
            new_pos = obs['positions'][0]

            if np.array_equal(old_pos, new_pos):
                # Collision occurred
                assert reward < -0.01  # Should be more negative than just step penalty
                break

        env.close()

    def test_success_reward(self):
        """Test that success reward is given when passing through doorway."""
        # This test requires a known map with accessible doorways
        env = self.create_env(
            auto_explore=True,
            success_reward=10.0,
            max_task_steps=500
        )

        obs, info = env.reset()

        if env.target_doorway is not None:
            # Manually simulate successful doorway passage
            env.position_before_doorway = (env.target_doorway[0] - 1, env.target_doorway[1])
            env.previous_pos = env.target_doorway

            # Compute reward for passing through
            mock_obs = obs.copy()
            mock_obs['positions'][0] = [env.target_doorway[0] + 1, env.target_doorway[1]]

            old_has_passed = env.has_passed_through
            reward = env._compute_task_reward(mock_obs, Action.FORWARD, 0.0)

            if env.has_passed_through and not old_has_passed:
                assert reward > 5.0  # Should include success reward

        env.close()

    # ============== Task Completion Tests ==============

    def test_task_completion_detection(self):
        """Test that task completion is detected correctly."""
        env = self.create_env(auto_explore=True, max_task_steps=100)
        obs, info = env.reset()

        if env.target_doorway is not None:
            # Simulate successful passage
            env.has_passed_through = True

            status = env._check_task_status(obs, Action.FORWARD)
            assert status == 1  # TaskStatus.SUCCESS

            # Reset and test failure condition
            env.has_passed_through = False
            env.task_step = 100  # Should be >= max_task_steps for failure

            status = env._check_task_status(obs, Action.FORWARD)
            assert status == 2  # TaskStatus.FAILURE

        env.close()

    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        env = self.create_env(
            auto_explore=True,
            max_task_steps=50
        )

        obs, info = env.reset()

        # Run until termination
        total_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and total_steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

        # Should have terminated or truncated
        assert terminated or truncated

        # Check termination reason
        if terminated:
            assert 'task_success' in info
            if info['task_success']:
                assert env.has_passed_through

        if truncated:
            if 'task_success' in info:
                assert not info['task_success']

        env.close()



    def test_doorway_approach_tracking(self):
        """Test that doorway approach is tracked correctly."""
        env = self.create_env(auto_explore=True)
        obs, info = env.reset()

        if env.target_doorway is not None:
            # Test distance tracking
            initial_distance = env.initial_distance
            assert initial_distance is not None
            assert initial_distance > 0

            # Take a step
            obs, reward, _, _, _ = env.step(Action.FORWARD)

            # Distance should be updated
            if env.previous_distance is not None:
                current_pos = tuple(obs['positions'][0])
                actual_distance = abs(current_pos[0] - env.target_doorway[0]) + \
                                abs(current_pos[1] - env.target_doorway[1])
                assert abs(env.previous_distance - actual_distance) < 2  # Allow for movement

        env.close()

    # ============== Stable Baselines3 Compatibility Tests ==============

    def test_sb3_env_checker(self):
        """Test that environment passes SB3 environment checker."""
        env = self.create_env(auto_explore=True)

        try:
            check_env(env, warn=True, skip_render_check=True)
            env_check_passed = True
        except Exception as e:
            print(f"Environment check failed: {e}")
            env_check_passed = False

        assert env_check_passed, "Environment failed SB3 compatibility check"
        env.close()

    def test_sb3_dqn_compatibility(self):
        """Test that environment works with SB3 DQN."""
        env = self.create_env(auto_explore=True, max_task_steps=100)

        try:
            # Create a DQN model
            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=0,
                learning_rate=1e-4,
                buffer_size=1000,
                learning_starts=50,
                batch_size=32,
                train_freq=4,
                target_update_interval=100
            )

            # Try to train for a few steps
            model.learn(total_timesteps=100)

            # Try prediction
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)

            # Action should be valid
            assert env.action_space.contains(action)

            dqn_compatible = True
        except Exception as e:
            print(f"DQN compatibility test failed: {e}")
            dqn_compatible = False

        assert dqn_compatible, "Environment not compatible with SB3 DQN"
        env.close()

    # ============== Edge Cases and Stress Tests ==============

    def test_no_doorways_scenario(self):
        """Test environment behavior when no doorways exist."""
        env = RoomEntryWrapper(
            env_config=self.default_config,
            precomputed_doorways={},  # No doorways
            auto_explore=True,
            max_exploration_steps=50
        )

        obs, info = env.reset()

        # Should complete exploration without crashing
        assert env.exploration_steps > 0
        assert len(env.discovered_doorway_indices) == 0
        assert env.target_doorway is None

        # Should still be able to step
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        env.close()

    def test_multiple_resets(self):
        """Test that multiple resets don't cause issues."""
        env = self.create_env(auto_explore=True)

        for i in range(5):
            obs, info = env.reset()

            # Each reset should properly initialize
            assert env.task_step == 0
            assert not env.has_passed_through

            # Take a few steps
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        env.close()

    def test_rapid_action_switching(self):
        """Test rapid switching between different actions."""
        env = self.create_env(auto_explore=False)
        obs, info = env.reset()

        # Rapidly switch between all actions
        for _ in range(50):
            for action in [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STAY]:
                obs, reward, terminated, truncated, info = env.step(action)

                # Should handle rapid switching without issues
                assert env.observation_space.contains(obs)
                assert isinstance(reward, (float, np.floating))

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        env.close()

    # ============== Performance Tests ==============

    def test_step_performance(self):
        """Test that steps execute in reasonable time."""
        env = self.create_env(auto_explore=False)
        obs, info = env.reset()

        step_times = []
        for _ in range(100):
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(0)
            step_time = time.time() - start_time
            step_times.append(step_time)

            if terminated or truncated:
                break

        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)

        # Steps should be fast
        assert avg_step_time < 0.01, f"Average step time too slow: {avg_step_time:.4f}s"
        assert max_step_time < 0.1, f"Max step time too slow: {max_step_time:.4f}s"

        env.close()

    def test_reset_performance(self):
        """Test that reset executes in reasonable time."""
        env = self.create_env(auto_explore=True, max_exploration_steps=100)

        reset_times = []
        for _ in range(5):
            start_time = time.time()
            obs, info = env.reset()
            reset_time = time.time() - start_time
            reset_times.append(reset_time)

        avg_reset_time = np.mean(reset_times)

        # Reset with auto-exploration should still be reasonably fast
        assert avg_reset_time < 2.0, f"Average reset time too slow: {avg_reset_time:.2f}s"

        env.close()


# ============== Integration Test ==============

def test_full_episode_integration():
    """Run a complete episode to test full integration."""
    # Setup
    map_path = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

    if os.path.exists(map_path):
        doorways = precompute_doorways(map_path)
    else:
        doorways = {(5, 5): 'horizontal', (10, 10): 'vertical'}

    config = {
        'width': 32,
        'height': 32,
        'num_agents': 1,
        'max_steps': 500,
        'map_path': map_path if os.path.exists(map_path) else None,
        'randomize': False if os.path.exists(map_path) else True,
    }

    env = RoomEntryWrapper(
        env_config=config,
        precomputed_doorways=doorways,
        auto_explore=True,
        max_task_steps=300
    )

    # Run episode
    obs, info = env.reset()

    episode_reward = 0
    episode_steps = 0
    success = False

    while episode_steps < 500:
        # Simple policy: random with forward bias
        if np.random.random() < 0.6:
            action = Action.FORWARD
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1

        if terminated or truncated:
            if 'task_success' in info:
                success = info['task_success']
            break

    # Verify episode completed properly
    assert episode_steps > 0
    assert 'task_status' in info

    print(f"Episode completed: Steps={episode_steps}, Reward={episode_reward:.2f}, Success={success}")
    print(f"Discovered doorways: {info.get('discovered_doorways', 0)}")
    print(f"Target doorway: {info.get('target_doorway', None)}")
    print(f"Has passed through: {info.get('has_passed_through', False)}")

    env.close()


if __name__ == "__main__":
    # Run all tests
    print("Starting comprehensive environment tests...")

    test_suite = TestRoomEntryEnvironment()
    test_suite.setup_class()

    # Run each test method
    test_methods = [
        test_suite.test_environment_initialization,
        test_suite.test_action_space,
        test_suite.test_observation_space,
        test_suite.test_reset_functionality,
        test_suite.test_auto_exploration_on_reset,
        test_suite.test_step_action_conversion,
        test_suite.test_step_counter_accuracy,
        test_suite.test_doorway_discovery_mechanism,
        test_suite.test_target_doorway_selection,
        test_suite.test_reward_structure,
        test_suite.test_success_reward,
        test_suite.test_task_completion_detection,
        test_suite.test_episode_termination,
        test_suite.test_doorway_approach_tracking,
        test_suite.test_sb3_env_checker,
        test_suite.test_sb3_dqn_compatibility,
        test_suite.test_no_doorways_scenario,
        test_suite.test_multiple_resets,
        test_suite.test_rapid_action_switching,
        test_suite.test_step_performance,
        test_suite.test_reset_performance,
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        test_name = test_method.__name__
        try:
            print(f"\nRunning {test_name}...")
            test_method()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1

    # Run integration test
    print("\nRunning integration test...")
    try:
        test_full_episode_integration()
        print("✓ Integration test PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ Integration test FAILED: {e}")
        failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")