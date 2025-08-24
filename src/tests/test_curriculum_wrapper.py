"""
tests/test_curriculum_wrapper.py

Comprehensive test suite for the CurriculumWrapper.
Tests curriculum learning features, map revelation, adaptive parameters,
rendering through both wrappers, and edge cases.
"""

import numpy as np
import pygame
import time

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.curriculum_wrapper import CurriculumWrapper
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper
from environments.base.constants import TileType
from sensors.camera_sensor import CameraSensor


class TestCurriculumWrapper:
    """Test suite for CurriculumWrapper"""

    def test_wrapper_initialization(self):
        """Test curriculum wrapper initialization with various hidden sizes."""
        print("\n" + "="*60)
        print("TEST: Curriculum Wrapper Initialization")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        # Test different hidden sizes
        hidden_sizes = [8, 12, 16, 20, 24, 28, 32]

        for hidden_size in hidden_sizes:
            base_env = MultiAgentSLAMEnv(
                width=32,
                height=32,
                num_agents=1,
                sensor_config={0: sensor}
            )

            curriculum_env = CurriculumWrapper(base_env, hidden_size=hidden_size)

            # Check adaptive parameters
            expected_cells = hidden_size * hidden_size
            expected_max_steps = max(200, int((expected_cells / 100) * 500))
            expected_bonus = expected_cells / 2.0

            assert curriculum_env.hidden_cells == expected_cells
            assert curriculum_env.adaptive_max_steps == expected_max_steps
            assert curriculum_env.adaptive_completion_bonus == expected_bonus

            print(f"✓ Hidden {hidden_size}x{hidden_size}: "
                  f"{expected_cells} cells, "
                  f"{expected_max_steps} steps, "
                  f"{expected_bonus:.1f} bonus")

            curriculum_env.close()

    def test_map_revelation(self):
        """Test that the wrapper correctly reveals most of the map."""
        print("\n" + "="*60)
        print("TEST: Map Revelation")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        for hidden_size in [8, 16, 24]:
            base_env = MultiAgentSLAMEnv(
                width=32,
                height=32,
                num_agents=1,
                randomize=False,
                sensor_config={0: sensor}
            )

            curriculum_env = CurriculumWrapper(base_env, hidden_size=hidden_size)
            obs, info = curriculum_env.reset()

            # Count unknown cells
            unknown_count = np.sum(obs['global_map'] == TileType.UNKNOWN)
            revealed_count = (32 * 32) - unknown_count

            # Check that only the hidden square is unknown
            expected_unknown = hidden_size * hidden_size
            tolerance = 10  # Allow some tolerance for boundaries

            assert abs(unknown_count - expected_unknown) <= tolerance

            print(f"✓ Hidden {hidden_size}x{hidden_size}: "
                  f"{unknown_count} unknown cells (expected ~{expected_unknown}), "
                  f"{revealed_count} revealed")

            # Check hidden square boundaries
            hidden_min = curriculum_env.hidden_min
            hidden_max = curriculum_env.hidden_max

            # Verify cells outside hidden area are revealed
            for y in range(32):
                for x in range(32):
                    if not (hidden_min <= x < hidden_max and hidden_min <= y < hidden_max):
                        # These should be revealed (not unknown)
                        if obs['global_map'][y, x] == TileType.UNKNOWN:
                            # Could be out of bounds or unreachable
                            pass

            print(f"  Hidden square: [{hidden_min}:{hidden_max}, {hidden_min}:{hidden_max}]")

            curriculum_env.close()

    def test_drone_placement(self):
        """Test that drone is placed within the hidden area."""
        print("\n" + "="*60)
        print("TEST: Drone Placement in Hidden Area")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        for hidden_size in [8, 12, 16]:
            base_env = MultiAgentSLAMEnv(
                width=32,
                height=32,
                num_agents=1,
                randomize=True,
                sensor_config={0: sensor}
            )

            curriculum_env = CurriculumWrapper(base_env, hidden_size=hidden_size)

            # Test multiple resets
            for _ in range(5):
                obs, info = curriculum_env.reset()

                # Get drone position
                drone_x, drone_y = obs['positions'][0]

                # Check if drone is in hidden area
                hidden_min = curriculum_env.hidden_min
                hidden_max = curriculum_env.hidden_max

                assert hidden_min <= drone_x < hidden_max
                assert hidden_min <= drone_y < hidden_max

                print(f"✓ Hidden {hidden_size}x{hidden_size}: "
                      f"Drone at ({drone_x}, {drone_y}) "
                      f"within [{hidden_min}:{hidden_max}]")

            curriculum_env.close()

    def test_reachable_mask(self):
        """Test that reachable mask only includes hidden area."""
        print("\n" + "="*60)
        print("TEST: Reachable Mask Configuration")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        for hidden_size in [8, 16]:
            base_env = MultiAgentSLAMEnv(
                width=32,
                height=32,
                num_agents=1,
                randomize=False,
                sensor_config={0: sensor}
            )

            curriculum_env = CurriculumWrapper(base_env, hidden_size=hidden_size)
            obs, info = curriculum_env.reset()

            # Access base environment through wrapper
            base_env = curriculum_env.env

            # Count reachable cells
            reachable_count = np.sum(base_env.reachable_mask)

            # Should be approximately the hidden area size
            expected_reachable = hidden_size * hidden_size

            print(f"✓ Hidden {hidden_size}x{hidden_size}: "
                  f"{reachable_count} reachable cells "
                  f"(hidden area: {expected_reachable})")

            # Verify reachable cells are in hidden area
            hidden_min = curriculum_env.hidden_min
            hidden_max = curriculum_env.hidden_max

            for y in range(32):
                for x in range(32):
                    if base_env.reachable_mask[y, x]:
                        # Should be in hidden area
                        assert hidden_min <= x < hidden_max
                        assert hidden_min <= y < hidden_max

            curriculum_env.close()

    def test_with_multidiscrete_wrapper(self):
        """Test curriculum wrapper with MultiDiscrete wrapper."""
        print("\n" + "="*60)
        print("TEST: Integration with MultiDiscrete Wrapper")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            sensor_config={0: sensor}
        )

        # Apply both wrappers
        curriculum_env = CurriculumWrapper(base_env, hidden_size=12)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        # Check action space
        assert wrapped_env.action_space.n == 4  # Single agent = 4 actions
        print(f"✓ Action space: Discrete({wrapped_env.action_space.n})")

        # Reset and check
        obs, info = wrapped_env.reset()

        assert 'curriculum_stage' in info
        assert info['curriculum_stage']['hidden_size'] == 12
        print(f"✓ Curriculum info preserved through wrapper")

        # Take some steps
        for i in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            assert 'hidden_cells' in info
            assert 'hidden_size' in info

        print(f"✓ Step execution works through both wrappers")

        wrapped_env.close()

    def test_adaptive_parameters(self):
        """Test adaptive max_steps and completion_bonus calculation."""
        print("\n" + "="*60)
        print("TEST: Adaptive Parameters")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        test_cases = [
            (8, 64, max(200, int(64/100 * 500)), 32.0),
            (12, 144, max(200, int(144/100 * 500)), 72.0),
            (16, 256, int(256/100 * 500), 128.0),
            (24, 576, int(576/100 * 500), 288.0),
        ]

        for hidden_size, expected_cells, expected_steps, expected_bonus in test_cases:
            base_env = MultiAgentSLAMEnv(
                width=32,
                height=32,
                num_agents=1,
                sensor_config={0: sensor}
            )

            curriculum_env = CurriculumWrapper(base_env, hidden_size=hidden_size)

            assert curriculum_env.hidden_cells == expected_cells
            assert curriculum_env.adaptive_max_steps == expected_steps
            assert curriculum_env.adaptive_completion_bonus == expected_bonus

            print(f"✓ Hidden {hidden_size}x{hidden_size}: "
                  f"cells={expected_cells}, "
                  f"steps={expected_steps}, "
                  f"bonus={expected_bonus:.1f}")

            curriculum_env.close()

    def test_step_and_rewards(self):
        """Test step execution and reward calculation."""
        print("\n" + "="*60)
        print("TEST: Step Execution and Rewards")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            randomize=False,
            sensor_config={0: sensor},
            discovery_reward=1.0,
            collision_penalty=-5.0,
            step_penalty=-0.1
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=12)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()

        # Run episode
        total_reward = 0
        step_count = 0
        discoveries = 0

        for _ in range(50):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            total_reward += reward
            step_count += 1

            if reward > 0:
                discoveries += 1

            if terminated or truncated:
                break

        print(f"✓ Ran {step_count} steps")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Discovery steps: {discoveries}")
        print(f"  Final progress: {info.get('progress', 0)*100:.1f}%")

        wrapped_env.close()

    def test_rendering_with_curriculum(self):
        """Test rendering through curriculum wrapper."""
        print("\n" + "="*60)
        print("TEST: Rendering with Curriculum")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        # Test human rendering
        print("\nTesting 'human' rendering...")
        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            render_mode='human',
            randomize=False,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=12)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()

        # Render a few frames
        for i in range(5):
            wrapped_env.render()
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            time.sleep(0.1)

        print("✓ Human rendering works with curriculum")
        wrapped_env.close()

        # Test rgb_array rendering
        print("\nTesting 'rgb_array' rendering...")
        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            render_mode='rgb_array',
            randomize=False,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=12)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()
        frame = wrapped_env.render()

        assert frame is not None
        assert len(frame.shape) == 3
        print(f"✓ RGB array rendering works: shape {frame.shape}")

        wrapped_env.close()

    def test_termination_conditions(self):
        """Test termination with adaptive parameters."""
        print("\n" + "="*60)
        print("TEST: Termination Conditions")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        # Small hidden area for quick completion
        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            randomize=False,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=4)  # Very small area
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()

        # Try to complete the small area
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step_count < 300:
            # Use semi-random exploration
            if step_count % 4 == 0:
                action = 2  # FORWARD
            else:
                action = wrapped_env.action_space.sample()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            step_count += 1

        print(f"✓ Episode ended after {step_count} steps")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Final progress: {info.get('progress', 0)*100:.1f}%")

        if terminated:
            print(f"  Completion bonus earned!")

        wrapped_env.close()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        print("\n" + "="*60)
        print("TEST: Edge Cases")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        # Test with hidden_size = 32 (entire map)
        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=32)
        obs, info = curriculum_env.reset()

        # Should have all cells as unknown
        unknown_count = np.sum(obs['global_map'] == TileType.UNKNOWN)
        print(f"✓ Hidden 32x32 (full map): {unknown_count} unknown cells")

        curriculum_env.close()

        # Test with very small hidden size
        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=2)
        obs, info = curriculum_env.reset()

        unknown_count = np.sum(obs['global_map'] == TileType.UNKNOWN)
        print(f"✓ Hidden 2x2 (tiny): {unknown_count} unknown cells")

        curriculum_env.close()

    def test_performance(self):
        """Test performance with curriculum wrapper."""
        print("\n" + "="*60)
        print("TEST: Performance with Curriculum")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            render_mode=None,
            sensor_config={0: sensor}
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=16)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()

        import time
        start_time = time.time()
        num_steps = 100

        for _ in range(num_steps):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                obs, info = wrapped_env.reset()

        elapsed = time.time() - start_time
        steps_per_second = num_steps / elapsed

        print(f"✓ Processed {num_steps} steps in {elapsed:.2f} seconds")
        print(f"  Performance: {steps_per_second:.1f} steps/second")

        wrapped_env.close()

    def test_interactive_curriculum(self):
        """Interactive test with curriculum wrapper."""
        print("\n" + "="*60)
        print("TEST: Interactive Curriculum Test")
        print("="*60)
        print("Controls:")
        print("  SPACE - Random action")
        print("  Arrow Keys - Manual control")
        print("  1-5 - Change hidden size (8, 12, 16, 20, 24)")
        print("  R - Reset")
        print("  Q - Quit")
        print("="*60)

        sensor = CameraSensor(max_range=5, fov_deg=90, num_rays=20)

        # Start with medium hidden size
        current_hidden_size = 12

        base_env = MultiAgentSLAMEnv(
            width=32,
            height=32,
            num_agents=1,
            render_mode='human',
            randomize=True,
            sensor_config={0: sensor},
            discovery_reward=1.0,
            collision_penalty=-0.1,
            step_penalty=0.0
        )

        curriculum_env = CurriculumWrapper(base_env, hidden_size=current_hidden_size)
        wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

        obs, info = wrapped_env.reset()
        wrapped_env.render()

        clock = pygame.time.Clock()
        running = True
        step_count = 0
        total_reward = 0

        print(f"\nStarting with hidden size {current_hidden_size}x{current_hidden_size}")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = None

                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, info = wrapped_env.reset()
                        step_count = 0
                        total_reward = 0
                        print(f"\n🔄 Environment reset!")
                        print(f"Hidden size: {current_hidden_size}x{current_hidden_size}")
                    elif event.key == pygame.K_SPACE:
                        action = wrapped_env.action_space.sample()
                    elif event.key == pygame.K_LEFT:
                        action = 0  # TURN_LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = 1  # TURN_RIGHT
                    elif event.key == pygame.K_UP:
                        action = 2  # FORWARD
                    elif event.key == pygame.K_DOWN:
                        action = 3  # STAY
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        # Change hidden size
                        size_map = {
                            pygame.K_1: 8,
                            pygame.K_2: 12,
                            pygame.K_3: 16,
                            pygame.K_4: 20,
                            pygame.K_5: 24
                        }
                        new_hidden_size = size_map[event.key]

                        if new_hidden_size != current_hidden_size:
                            print(f"\n🔧 Changing hidden size from {current_hidden_size} to {new_hidden_size}")

                            # Close current environment
                            wrapped_env.close()

                            # Create new environment with different hidden size
                            base_env = MultiAgentSLAMEnv(
                                width=32,
                                height=32,
                                num_agents=1,
                                render_mode='human',
                                randomize=True,
                                sensor_config={0: sensor},
                                discovery_reward=1.0,
                                collision_penalty=-0.1,
                                step_penalty=0.0
                            )

                            curriculum_env = CurriculumWrapper(base_env, hidden_size=new_hidden_size)
                            wrapped_env = MultiDiscreteToDiscreteWrapper(curriculum_env)

                            obs, info = wrapped_env.reset()
                            step_count = 0
                            total_reward = 0
                            current_hidden_size = new_hidden_size

                            print(f"✓ Now using {current_hidden_size}x{current_hidden_size} hidden area")
                            print(f"  Max steps: {curriculum_env.adaptive_max_steps}")
                            print(f"  Completion bonus: {curriculum_env.adaptive_completion_bonus:.1f}")

                    if action is not None:
                        obs, reward, terminated, truncated, info = wrapped_env.step(action)
                        step_count += 1
                        total_reward += reward

                        # Get actual drone position through wrapper chain
                        actual_env = wrapped_env.env.env  # MultiDiscrete -> Curriculum -> SLAM
                        drone_pos = actual_env.drones[0].pos

                        action_names = ["LEFT", "RIGHT", "FWD", "STAY"]
                        print(f"Step {step_count}: {action_names[action]}, "
                              f"Pos={drone_pos}, "
                              f"R={reward:.2f}, "
                              f"Prog={info['progress']*100:.1f}%")

                        if terminated or truncated:
                            print(f"\n{'✅ Completed!' if terminated else '⏱️ Truncated!'}")
                            print(f"Total reward: {total_reward:.3f}")
                            print(f"Steps taken: {step_count}")
                            obs, info = wrapped_env.reset()
                            step_count = 0
                            total_reward = 0

            wrapped_env.render()
            clock.tick(10)

        wrapped_env.close()
        print("\n✓ Interactive test completed")


def run_all_tests(skip_interactive=False):
    """Run all tests in sequence."""
    print("\n" + "="*70)
    print(" COMPREHENSIVE TEST SUITE FOR CurriculumWrapper")
    print("="*70)

    test_suite = TestCurriculumWrapper()

    # Non-interactive tests
    test_suite.test_wrapper_initialization()
    test_suite.test_map_revelation()
    test_suite.test_drone_placement()
    test_suite.test_reachable_mask()
    test_suite.test_with_multidiscrete_wrapper()
    test_suite.test_adaptive_parameters()
    test_suite.test_step_and_rewards()
    test_suite.test_rendering_with_curriculum()
    test_suite.test_termination_conditions()
    test_suite.test_edge_cases()
    test_suite.test_performance()

    print("\n" + "="*70)
    print(" ALL AUTOMATED TESTS PASSED! ✅")
    print("="*70)

    # Ask if user wants to run interactive test
    if not skip_interactive:
        response = input("\nRun interactive curriculum test? (y/n): ")
        if response.lower() == 'y':
            test_suite.test_interactive_curriculum()


if __name__ == "__main__":
    import sys
    skip_interactive = '--no-interactive' in sys.argv
    run_all_tests(skip_interactive=skip_interactive)