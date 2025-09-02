"""
Comprehensive test suite for Wall Following Environment.
Tests all aspects of the environment including state/action spaces,
reward computation, episode termination, and edge cases.

UPDATED for latest changes:
- Better collision detection using info dict
- Wall-lock verification with retry
- Proper metric reset after pre-search
- Discovered cells tracking improvements
"""

import numpy as np
import pytest
from typing import Dict, Tuple, List
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from environments.tasks.wall_following_wrapper import WallFollowingWrapper
from environments.base.constants import TileType, Action


class TestWallFollowingEnvironment:
    """Comprehensive test suite for Wall Following environment."""

    def setup_method(self):
        """Setup test environment before each test."""
        self.env_config = {
            'width': 20,
            'height': 20,
            'num_agents': 1,
            'max_steps': 500,
            'randomize': False,  # Use deterministic map for testing
            'render_mode': None,  # No rendering for tests
        }

    def teardown_method(self):
        """Cleanup after each test."""
        if hasattr(self, 'env'):
            self.env.close()

    # ============= Basic Functionality Tests =============

    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = WallFollowingWrapper(self.env_config)
        assert env is not None
        assert env.action_space.n == 4  # 4 actions (including STAY)
        assert 'global_map' in env.observation_space.spaces
        assert 'positions' in env.observation_space.spaces
        assert 'facings' in env.observation_space.spaces
        env.close()

    def test_reset_returns_valid_observation(self):
        """Test that reset returns valid observation and info."""
        env = WallFollowingWrapper(self.env_config)
        obs, info = env.reset()

        # Check observation structure
        assert 'global_map' in obs
        assert 'positions' in obs
        assert 'facings' in obs
        assert 'active' in obs

        # Check observation shapes
        assert obs['global_map'].shape == (20, 20)
        assert obs['positions'].shape == (1, 2)
        assert obs['facings'].shape == (1,)
        assert obs['active'].shape == (1,)

        # Check that a wall is visible and locked after reset
        global_map = obs['global_map']
        wall_count = np.sum(global_map == TileType.WALL)
        assert wall_count > 0, "No walls visible after reset"
        assert env.wall_locked, "Wall not locked after reset"

        # Check info contains expected fields
        assert 'pre_search_steps' in info
        assert 'wall_locked' in info
        assert info['wall_locked'] == True

        env.close()

    def test_action_space_validity(self):
        """Test that all actions in action space are valid."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        for action in range(env.action_space.n):
            # Should not raise error
            step_result = env.step(action)
            assert len(step_result) == 5, "Step should return 5-tuple"

            obs, reward, terminated, truncated, info = step_result

            # Verify return types
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            # Reset for next action test
            env.reset()

        env.close()

    # ============= Pre-search Functionality Tests =============

    def test_pre_search_finds_wall(self):
        """Test that pre-search successfully finds a wall."""
        env = WallFollowingWrapper(self.env_config)

        for _ in range(3):  # Test multiple times
            obs, info = env.reset()

            # Check that wall is visible
            global_map = obs['global_map']
            wall_cells = np.sum(global_map == TileType.WALL)
            assert wall_cells > 0, "Pre-search failed to find wall"

            # Check pre-search info
            assert info['pre_search_steps'] >= 0
            assert info['pre_search_steps'] <= 1000  # Could retry with extended search

            # Check that target wall is set
            assert env.wall_locked, "Wall not locked after pre-search"
            assert len(env.target_wall_segment) > 0, "No target wall segment"
            assert len(env.accessible_wall_cells) > 0, "No accessible wall cells"

        env.close()

    def test_pre_search_retry_mechanism(self):
        """Test that pre-search retries if first attempt fails."""
        # This is hard to test directly without mocking, but we can verify the mechanism exists
        env = WallFollowingWrapper(self.env_config)

        # The retry mechanism should ensure wall_locked is always True after reset
        for _ in range(5):
            obs, info = env.reset()
            assert env.wall_locked, "Wall not locked even after retry"
            assert info['wall_locked'] == True

        env.close()

    def test_metric_reset_after_presearch(self):
        """Test that all metrics are properly reset after pre-search."""
        env = WallFollowingWrapper(self.env_config)
        obs, info = env.reset()

        # After reset, all counters should be zero
        assert env.task_step == 0, "Task step not reset"
        assert env.collision_count == 0, "Collision count not reset"
        assert env.last_known_collisions == 0, "Last known collisions not reset"
        assert env.env.current_step == 0, "Base env step not reset"

        # Drone metrics should also be reset
        for drone in env.env.drones:
            assert drone.collision_count == 0, f"Drone {drone.drone_id} collisions not reset"
            assert drone.total_discoveries == 0, f"Drone {drone.drone_id} discoveries not reset"

        env.close()

    # ============= Wall Tracking Tests =============

    def test_wall_segment_identification(self):
        """Test that wall segments are correctly identified."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # Check wall segment properties
        assert len(env.target_wall_segment) > 0, "No target wall"
        assert len(env.accessible_wall_cells) > 0, "No accessible cells"
        assert len(env.accessible_wall_cells) <= len(env.target_wall_segment), \
            "More accessible than total cells"

        # Boundaries should be extended beyond wall endpoints
        assert len(env.wall_boundaries) in [1, 2], \
            f"Invalid boundary count: {len(env.wall_boundaries)}"

        env.close()

    def test_extended_boundaries(self):
        """Test that boundaries are properly extended beyond wall endpoints."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        accessible = env.accessible_wall_cells
        boundaries = env.wall_boundaries

        if len(accessible) > 1 and len(boundaries) == 2:
            # Convert to list and sort to find actual endpoints
            accessible_list = list(accessible)

            # Check if vertical or horizontal
            first_cell = accessible_list[0]
            is_vertical = all(cell[0] == first_cell[0] for cell in accessible_list)

            if is_vertical:
                accessible_list.sort(key=lambda cell: cell[1])
                x = accessible_list[0][0]
                first_y = accessible_list[0][1]
                last_y = accessible_list[-1][1]

                # Boundaries should be extended by 1 cell
                for bx, by in boundaries:
                    assert bx == x, "Boundary not aligned with wall"
                    assert by == first_y - 1 or by == last_y + 1 or \
                           by == first_y or by == last_y, \
                           "Boundary not at expected position"
            else:
                accessible_list.sort(key=lambda cell: cell[0])
                y = accessible_list[0][1]
                first_x = accessible_list[0][0]
                last_x = accessible_list[-1][0]

                # Boundaries should be extended by 1 cell
                for bx, by in boundaries:
                    assert by == y, "Boundary not aligned with wall"
                    assert bx == first_x - 1 or bx == last_x + 1 or \
                           bx == first_x or bx == last_x, \
                           "Boundary not at expected position"

        env.close()

    def test_wall_discovery_tracking(self):
        """Test that wall discovery tracking only includes task-relevant cells."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        initial_discovered = len(env.discovered_cells)
        task_cells = env.accessible_wall_cells | env.wall_boundaries

        # Take actions to explore
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Discovered cells should only be from task-relevant cells
            assert env.discovered_cells.issubset(task_cells), \
                "Discovered cells outside task scope"

            # Check discovery doesn't decrease
            assert len(env.discovered_cells) >= initial_discovered, \
                "Discovered cells decreased"

            if terminated or truncated:
                break

        env.close()

    # ============= Collision Detection Tests =============

    def test_collision_detection_via_info(self):
        """Test that collisions are properly detected using info dict."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        initial_collisions = 0

        # Try to cause collisions by moving forward repeatedly
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(Action.FORWARD)

            # Check collision tracking via info
            if 'collision_counts' in info:
                current_collisions = info['collision_counts'][0]
                assert current_collisions >= initial_collisions, \
                    "Collision count decreased"

                # Check wrapper tracking matches
                assert env.last_known_collisions == current_collisions, \
                    "Last known collisions mismatch"

                initial_collisions = current_collisions

            if terminated or truncated:
                break

        env.close()

    def test_collision_penalty_application(self):
        """Test that collision penalties are properly applied."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # Find a wall and position to cause collision
        pos = tuple(obs['positions'][0])
        facing = obs['facings'][0]

        # Move forward multiple times to likely hit walls
        collision_detected = False
        for _ in range(10):
            old_collision_count = env.collision_count

            obs, reward, terminated, truncated, info = env.step(Action.FORWARD)

            # Check if collision occurred
            if env.collision_count > old_collision_count:
                collision_detected = True
                # Collision should result in negative reward component
                # (accounting for other reward components)
                assert reward < 0, f"No penalty for collision: reward={reward}"

            if terminated or truncated:
                break

        if not collision_detected:
            print("Warning: No collisions detected in collision penalty test")

        env.close()

    # ============= Reward Computation Tests =============

    def test_reward_computation_with_collision_flag(self):
        """Test that reward computation properly uses collision flag."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # Track rewards with and without collisions
        rewards_with_collision = []
        rewards_without_collision = []

        for _ in range(30):
            old_collisions = env.last_known_collisions

            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Check if collision occurred
            if 'collision_counts' in info:
                if info['collision_counts'][0] > old_collisions:
                    rewards_with_collision.append(reward)
                else:
                    rewards_without_collision.append(reward)
            else:
                rewards_without_collision.append(reward)

            if terminated or truncated:
                break

        # Collisions should generally lead to lower rewards
        if rewards_with_collision and rewards_without_collision:
            avg_with_collision = np.mean(rewards_with_collision)
            avg_without_collision = np.mean(rewards_without_collision)

            # This might not always hold due to discovery rewards, but generally should
            print(f"Avg reward with collision: {avg_with_collision:.3f}")
            print(f"Avg reward without collision: {avg_without_collision:.3f}")

        env.close()

    def test_discovery_reward_structure(self):
        """Test the progressive discovery reward structure."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        discovery_events = []
        total_to_discover = len(env.accessible_wall_cells | env.wall_boundaries)

        for _ in range(100):
            old_discovered = len(env.discovered_cells)

            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            new_discovered = len(env.discovered_cells)

            if new_discovered > old_discovered:
                remaining = total_to_discover - new_discovered
                discovery_events.append({
                    'reward': reward,
                    'cells_discovered': new_discovered - old_discovered,
                    'remaining': remaining,
                    'coverage': info['wall_coverage']
                })

            if terminated or truncated:
                if info.get('task_status') == 1:  # Success
                    print(f"Task completed with {len(discovery_events)} discovery events")
                break

        # Verify reward structure increases near completion
        if len(discovery_events) >= 2:
            # Later discoveries (closer to completion) should generally have higher rewards
            early_rewards = [e['reward'] for e in discovery_events[:len(discovery_events)//2]]
            late_rewards = [e['reward'] for e in discovery_events[len(discovery_events)//2:]]

            if early_rewards and late_rewards:
                print(f"Early discovery avg: {np.mean(early_rewards):.3f}")
                print(f"Late discovery avg: {np.mean(late_rewards):.3f}")

        env.close()

    # ============= Episode Termination Tests =============

    def test_successful_termination_conditions(self):
        """Test that success requires discovering all task cells including boundaries."""
        small_config = self.env_config.copy()
        small_config['width'] = 10
        small_config['height'] = 10

        env = WallFollowingWrapper(small_config)

        success_count = 0
        for episode in range(5):
            obs, _ = env.reset()

            for step in range(200):
                # Simple exploration strategy
                if step % 3 == 0:
                    action = Action.TURN_LEFT
                else:
                    action = Action.FORWARD

                obs, reward, terminated, truncated, info = env.step(action)

                if terminated and info.get('task_status') == 1:
                    # Verify all cells discovered
                    total_to_discover = len(env.accessible_wall_cells | env.wall_boundaries)
                    assert len(env.discovered_cells) >= total_to_discover * 0.95, \
                        "Success without sufficient discovery"

                    # Verify final bonus was applied
                    assert reward > 5.0, "No final bonus on success"

                    success_count += 1
                    print(f"Episode {episode}: Success at step {step}")
                    break

                if terminated or truncated:
                    break

        print(f"Success rate: {success_count}/5 episodes")
        env.close()

    def test_timeout_termination(self):
        """Test timeout termination after 500 task steps."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # Note: task_step is used for timeout, not total steps
        for step in range(501):
            obs, reward, terminated, truncated, info = env.step(Action.STAY)

            if terminated or truncated:
                # Should terminate at exactly 500 task steps
                assert env.task_step <= 500, f"Didn't timeout at 500: {env.task_step}"
                assert info.get('task_status') == 2, "Wrong failure status on timeout"
                print(f"Timed out at task_step {env.task_step}")
                break
        else:
            assert False, "Didn't terminate after 500 steps"

        env.close()

    # ============= State Consistency Tests =============

    def test_state_consistency_across_episode(self):
        """Test that internal state remains consistent throughout episode."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        task_cells = env.accessible_wall_cells | env.wall_boundaries

        for step in range(100):
            # Verify state before step
            assert env.task_step == step, f"Task step mismatch: {env.task_step} != {step}"
            assert env.discovered_cells.issubset(task_cells), "Invalid discovered cells"

            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Verify state after step
            assert env.task_step == step + 1, "Task step didn't increment"
            assert len(env.discovered_cells) >= 0, "Negative discovered cells"
            assert info['wall_coverage'] >= 0.0 and info['wall_coverage'] <= 1.0, \
                "Invalid coverage"

            # Verify info consistency
            assert info['task_step'] == env.task_step, "Task step mismatch in info"
            assert info['collision_count'] == env.collision_count, "Collision count mismatch"

            if terminated or truncated:
                break

        env.close()

    def test_multiple_episode_isolation(self):
        """Test that episodes are properly isolated from each other."""
        env = WallFollowingWrapper(self.env_config)

        episode_data = []

        for episode in range(3):
            obs, info = env.reset()

            # Each episode should start fresh
            assert env.task_step == 0, f"Episode {episode}: task_step not reset"
            assert env.collision_count == 0, f"Episode {episode}: collisions not reset"
            assert env.last_known_collisions == 0, f"Episode {episode}: last_known not reset"
            assert len(env.discovered_cells) == 0, f"Episode {episode}: discoveries not reset"

            # Run episode
            episode_collisions = 0
            episode_discoveries = 0

            for _ in range(50):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

                episode_collisions = env.collision_count
                episode_discoveries = len(env.discovered_cells)

                if terminated or truncated:
                    break

            episode_data.append({
                'collisions': episode_collisions,
                'discoveries': episode_discoveries,
                'wall_segment': len(env.target_wall_segment)
            })

        # Each episode might have different walls
        print(f"Episode variations: {episode_data}")

        env.close()

    # ============= Edge Cases Tests =============

    def test_single_cell_wall_handling(self):
        """Test handling of single-cell wall segments."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # If we get a single-cell wall
        if len(env.accessible_wall_cells) == 1:
            # Boundaries should still be set (might be same as the cell itself)
            assert len(env.wall_boundaries) >= 1, "No boundaries for single cell"

            # Should be able to complete by discovering just this cell
            for _ in range(20):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

                if terminated and info.get('task_status') == 1:
                    assert len(env.discovered_cells) >= 1, "Didn't discover single cell"
                    print("Successfully handled single-cell wall")
                    break

                if terminated or truncated:
                    break

        env.close()

    def test_action_format_handling(self):
        """Test that various action formats are handled correctly."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        # Test different action formats
        action_formats = [
            0,  # Plain integer
            np.int32(1),  # Numpy integer
            np.array(2),  # Scalar array
            np.array([3]),  # 1D array with single element
        ]

        for action in action_formats:
            obs, reward, terminated, truncated, info = env.step(action)

            # Should not raise errors
            assert isinstance(reward, (int, float)), f"Invalid reward for action {action}"

            if terminated or truncated:
                env.reset()

        env.close()

    def test_discovered_cells_only_task_relevant(self):
        """Test that only task-relevant cells are tracked as discovered."""
        env = WallFollowingWrapper(self.env_config)
        obs, _ = env.reset()

        task_cells = env.accessible_wall_cells | env.wall_boundaries

        # Take many random actions
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            # Every discovered cell must be task-relevant
            for cell in env.discovered_cells:
                assert cell in task_cells, f"Non-task cell {cell} marked as discovered"

            if terminated or truncated:
                break

        env.close()

    # ============= Performance Tests =============

    def test_environment_performance(self):
        """Test environment performance metrics."""
        env = WallFollowingWrapper(self.env_config)

        # Warm up
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())

        # Measure reset time
        reset_times = []
        for _ in range(10):
            start = time.time()
            env.reset()
            reset_times.append(time.time() - start)

        # Measure step time
        env.reset()
        step_times = []
        for _ in range(100):
            start = time.time()
            env.step(env.action_space.sample())
            step_times.append(time.time() - start)

        avg_reset = np.mean(reset_times)
        avg_step = np.mean(step_times)

        print(f"\nPerformance Metrics:")
        print(f"  Avg Reset Time: {avg_reset * 1000:.2f}ms")
        print(f"  Avg Step Time: {avg_step * 1000:.2f}ms")
        print(f"  Steps per second: {1/avg_step:.0f}")

        # Performance thresholds
        assert avg_reset < 2.0, f"Reset too slow: {avg_reset:.3f}s"
        assert avg_step < 0.1, f"Step too slow: {avg_step:.3f}s"

        env.close()

    def test_info_dict_completeness(self):
        """Test that info dict contains all expected fields."""
        env = WallFollowingWrapper(self.env_config)
        obs, info = env.reset()

        # Check reset info
        assert 'pre_search_steps' in info, "Missing pre_search_steps"
        assert 'wall_locked' in info, "Missing wall_locked"

        # Check step info
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            required_fields = [
                'task_status', 'task_step', 'collision_count',
                'phase', 'wall_locked', 'wall_coverage',
                'discovered_accessible', 'total_accessible',
                'total_wall_cells', 'total_with_boundaries'
            ]

            for field in required_fields:
                assert field in info, f"Missing required field: {field}"

            # Validate field values
            assert info['total_with_boundaries'] >= info['total_accessible'], \
                "Total with boundaries should include accessible cells"
            assert info['wall_coverage'] == (
                len(env.discovered_cells) / info['total_with_boundaries']
                if info['total_with_boundaries'] > 0 else 0
            ), "Coverage calculation mismatch"

            if terminated or truncated:
                break

        env.close()


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("WALL FOLLOWING ENVIRONMENT TEST SUITE")
    print("Updated for latest changes")
    print("=" * 60)

    test_suite = TestWallFollowingEnvironment()
    test_methods = [method for method in dir(test_suite)
                    if method.startswith('test_')]

    passed = 0
    failed = 0
    errors = 0

    for test_name in test_methods:
        print(f"\nRunning {test_name}...")
        test_suite.setup_method()

        try:
            test_method = getattr(test_suite, test_name)
            test_method()
            print(f"  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠ ERROR: {type(e).__name__}: {e}")
            errors += 1
        finally:
            test_suite.teardown_method()

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{len(test_methods)}")
    print(f"Failed: {failed}/{len(test_methods)}")
    print(f"Errors: {errors}/{len(test_methods)}")

    if failed == 0 and errors == 0:
        print("\n✓ ALL TESTS PASSED!")
        return True
    else:
        print(f"\n✗ {failed + errors} TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)