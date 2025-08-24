"""
tests/test_multidiscrete_wrapper.py

Comprehensive test suite for the MultiDiscreteToDiscreteWrapper.
Tests action space conversion, bidirectional mapping, rendering through wrapper,
and compatibility with the base environment.
"""

import numpy as np
import pygame
import time

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.base.slam_env import MultiAgentSLAMEnv
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper


class TestMultiDiscreteWrapper:
    """Test suite for MultiDiscreteToDiscreteWrapper"""

    def test_wrapper_initialization(self):
        """Test wrapper initialization with various configurations."""
        print("\n" + "="*60)
        print("TEST: Wrapper Initialization")
        print("="*60)

        # Test with different numbers of agents
        for num_agents in [1, 2, 3, 5]:
            base_env = MultiAgentSLAMEnv(num_agents=num_agents)
            wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

            # Check action space conversion
            expected_actions = 4 ** num_agents  # 4 actions per agent
            assert wrapped_env.action_space.n == expected_actions
            print(f"✓ {num_agents} agents: Discrete({expected_actions}) action space")

            wrapped_env.close()

        # Test error handling for non-MultiDiscrete space
        try:
            import gymnasium as gym
            dummy_env = gym.make('CartPole-v1')
            wrapped = MultiDiscreteToDiscreteWrapper(dummy_env)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✓ Correctly rejects non-MultiDiscrete environments")
        finally:
            dummy_env.close()

    def test_action_conversion(self):
        """Test bidirectional action conversion."""
        print("\n" + "="*60)
        print("TEST: Action Conversion")
        print("="*60)

        base_env = MultiAgentSLAMEnv(num_agents=3)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        # Test decode action
        test_cases = [
            (0, [0, 0, 0]),   # All agents TURN_LEFT
            (1, [0, 0, 1]),   # Last agent TURN_RIGHT
            (21, [1, 1, 1]),  # All agents TURN_RIGHT
            (63, [3, 3, 3]),  # All agents STAY
        ]

        for discrete_action, expected_multi in test_cases:
            decoded = wrapped_env._decode_action(discrete_action)
            assert np.array_equal(decoded, expected_multi)
            print(f"✓ Action {discrete_action:2d} -> {decoded}")

        # Test encode action
        for discrete_action, multi_action in test_cases:
            encoded = wrapped_env._encode_action(np.array(multi_action))
            assert encoded == discrete_action
            print(f"✓ Multi {multi_action} -> {encoded}")

        # Test round-trip conversion
        for i in range(wrapped_env.total_actions):
            multi = wrapped_env._decode_action(i)
            back = wrapped_env._encode_action(multi)
            assert back == i

        print(f"✓ Round-trip conversion works for all {wrapped_env.total_actions} actions")

        wrapped_env.close()

    def test_action_meanings(self):
        """Test human-readable action meanings."""
        print("\n" + "="*60)
        print("TEST: Action Meanings")
        print("="*60)

        base_env = MultiAgentSLAMEnv(num_agents=2)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        meanings = wrapped_env.get_action_meanings()

        # Check all actions have meanings
        assert len(meanings) == wrapped_env.total_actions
        print(f"✓ All {wrapped_env.total_actions} actions have meanings")

        # Display sample meanings
        print("\nSample action meanings:")
        for i in [0, 1, 5, 10, 15]:
            if i < len(meanings):
                print(f"  Action {i:2d}: {meanings[i]}")

        # Verify meaning format
        assert "Agent0" in meanings[0]
        assert "Agent1" in meanings[0]
        assert "|" in meanings[0]
        print("✓ Action meanings properly formatted")

        wrapped_env.close()

    def test_step_execution(self):
        """Test step execution through wrapper."""
        print("\n" + "="*60)
        print("TEST: Step Execution Through Wrapper")
        print("="*60)

        base_env = MultiAgentSLAMEnv(num_agents=2, randomize=False)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()

        # Test various discrete actions
        test_actions = [0, 5, 10, 15]

        for action in test_actions:
            # Get initial positions
            initial_positions = obs['positions'].copy()

            # Take action
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # Decode what action was taken
            multi_action = wrapped_env._action_combinations[action]

            print(f"✓ Action {action} executed as {multi_action}")

            # Verify observation structure
            assert 'global_map' in obs
            assert 'positions' in obs
            assert 'facings' in obs
            assert 'active' in obs

        wrapped_env.close()

    def test_reset_through_wrapper(self):
        """Test reset functionality through wrapper."""
        print("\n" + "=" * 60)
        print("TEST: Reset Through Wrapper")
        print("=" * 60)

        base_env = MultiAgentSLAMEnv(num_agents=2)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        # Test multiple resets
        for i in range(3):
            obs, info = wrapped_env.reset(seed=42 + i)

            # Check observation structure
            assert isinstance(obs, dict)
            assert 'global_map' in obs
            assert obs['global_map'].shape == (32, 32)

            print(f"✓ Reset {i + 1} successful")

        # Test deterministic reset with non-randomized environment
        # Create a new environment with randomize=False for deterministic behavior
        base_env_det = MultiAgentSLAMEnv(num_agents=2, randomize=False)
        wrapped_env_det = MultiDiscreteToDiscreteWrapper(base_env_det)

        obs1, _ = wrapped_env_det.reset(seed=100)
        obs2, _ = wrapped_env_det.reset(seed=100)
        np.testing.assert_array_equal(obs1['positions'], obs2['positions'])
        print("✓ Deterministic reset works through wrapper")

        wrapped_env_det.close()
        wrapped_env.close()

    def test_reward_and_termination(self):
        """Test reward calculation and termination through wrapper."""
        print("\n" + "="*60)
        print("TEST: Rewards and Termination")
        print("="*60)

        base_env = MultiAgentSLAMEnv(
            num_agents=1,
            max_steps=50,
            discovery_reward=1.0,
            collision_penalty=-5.0,
            step_penalty=-0.1
        )
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()

        total_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        # Run until termination
        while not (terminated or truncated) and step_count < 60:
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward
            step_count += 1

        print(f"✓ Episode ended after {step_count} steps")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        assert step_count <= 50 or terminated  # Should truncate at max_steps
        print("✓ Termination conditions work correctly")

        wrapped_env.close()

    def test_rendering_through_wrapper(self):
        """Test rendering functionality through wrapper."""
        print("\n" + "="*60)
        print("TEST: Rendering Through Wrapper")
        print("="*60)

        # Test human rendering
        print("\nTesting 'human' rendering...")
        base_env = MultiAgentSLAMEnv(
            num_agents=2,
            render_mode='human',
            randomize=False
        )
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()

        # Render a few frames
        for i in range(5):
            wrapped_env.render()
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            time.sleep(0.1)

        print("✓ Human rendering works through wrapper")
        wrapped_env.close()

        # Test rgb_array rendering
        print("\nTesting 'rgb_array' rendering...")
        base_env = MultiAgentSLAMEnv(
            num_agents=2,
            render_mode='rgb_array',
            randomize=False
        )
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()
        frame = wrapped_env.render()

        assert frame is not None
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        print(f"✓ RGB array rendering works: shape {frame.shape}")

        wrapped_env.close()

    def test_single_vs_multi_agent(self):
        """Test wrapper with single agent (edge case)."""
        print("\n" + "="*60)
        print("TEST: Single vs Multi Agent")
        print("="*60)

        # Single agent
        base_env = MultiAgentSLAMEnv(num_agents=1)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        assert wrapped_env.action_space.n == 4  # Only 4 actions for single agent
        print(f"✓ Single agent: Discrete(4) action space")

        obs, info = wrapped_env.reset()

        # Test all 4 actions
        action_names = ["TURN_LEFT", "TURN_RIGHT", "FORWARD", "STAY"]
        for action in range(4):
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            print(f"  Action {action} ({action_names[action]}) executed")

        wrapped_env.close()

        # Multi agent comparison
        base_env = MultiAgentSLAMEnv(num_agents=3)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        assert wrapped_env.action_space.n == 64  # 4^3 = 64 actions
        print(f"✓ Three agents: Discrete(64) action space")

        wrapped_env.close()

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "="*60)
        print("TEST: Edge Cases")
        print("="*60)

        base_env = MultiAgentSLAMEnv(num_agents=2)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()

        # Test invalid action
        try:
            invalid_action = wrapped_env.action_space.n + 1
            obs, reward, terminated, truncated, info = wrapped_env.step(invalid_action)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Invalid action rejected correctly")

        # Test action space boundaries
        min_action = 0
        max_action = wrapped_env.action_space.n - 1

        obs, reward, terminated, truncated, info = wrapped_env.step(min_action)
        print(f"✓ Minimum action {min_action} works")

        obs, reward, terminated, truncated, info = wrapped_env.step(max_action)
        print(f"✓ Maximum action {max_action} works")

        # Test random sampling
        for _ in range(10):
            action = wrapped_env.sample_random_action()
            assert 0 <= action < wrapped_env.action_space.n

        print("✓ Random action sampling works")

        wrapped_env.close()

    def test_performance(self):
        """Test wrapper performance with many steps."""
        print("\n" + "="*60)
        print("TEST: Performance")
        print("="*60)

        base_env = MultiAgentSLAMEnv(num_agents=3, render_mode=None)
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

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

    def test_interactive_control(self):
        """Interactive test with manual control through wrapper."""
        print("\n" + "="*60)
        print("TEST: Interactive Control Through Wrapper")
        print("="*60)
        print("Controls:")
        print("  SPACE - Random action")
        print("  1-9 - Specific discrete actions")
        print("  Arrow Keys - Control first agent")
        print("  R - Reset")
        print("  Q - Quit")
        print("="*60)

        base_env = MultiAgentSLAMEnv(
            width=20,
            height=20,
            num_agents=2,
            render_mode='human',
            randomize=True
        )
        wrapped_env = MultiDiscreteToDiscreteWrapper(base_env)

        obs, info = wrapped_env.reset()
        wrapped_env.render()

        clock = pygame.time.Clock()
        running = True
        step_count = 0
        total_reward = 0

        # For 2 agents: map arrow keys to discrete actions
        # Action mapping for 2 agents (16 total actions)
        # Agent0: action_id // 4, Agent1: action_id % 4
        # We want Agent0 to respond to arrows, Agent1 to STAY

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
                    elif event.key == pygame.K_SPACE:
                        action = wrapped_env.action_space.sample()
                        print(f"Random action: {action}")
                    elif event.key == pygame.K_LEFT:
                        # Agent0: TURN_LEFT (0), Agent1: STAY (3)
                        action = 0 * 4 + 3  # = 3
                    elif event.key == pygame.K_RIGHT:
                        # Agent0: TURN_RIGHT (1), Agent1: STAY (3)
                        action = 1 * 4 + 3  # = 7
                    elif event.key == pygame.K_UP:
                        # Agent0: FORWARD (2), Agent1: STAY (3)
                        action = 2 * 4 + 3  # = 11
                    elif event.key == pygame.K_DOWN:
                        # Agent0: STAY (3), Agent1: STAY (3)
                        action = 3 * 4 + 3  # = 15
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        # Direct action selection
                        action = event.key - pygame.K_1
                        if action >= wrapped_env.action_space.n:
                            action = None
                            print(f"Action {event.key - pygame.K_1} out of range")

                    if action is not None:
                        obs, reward, terminated, truncated, info = wrapped_env.step(action)
                        step_count += 1
                        total_reward += reward

                        # Decode action for display
                        multi_action = wrapped_env._action_combinations[action]
                        action_names = ["LEFT", "RIGHT", "FWD", "STAY"]
                        decoded = f"[{action_names[multi_action[0]]}, {action_names[multi_action[1]]}]"

                        print(f"Step {step_count}: Action {action} = {decoded}, "
                              f"Reward={reward:.3f}, Progress={info['progress']*100:.1f}%")

                        if terminated or truncated:
                            print(f"\n{'✅ Completed!' if terminated else '⏱️ Truncated!'}")
                            print(f"Total reward: {total_reward:.3f}")
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
    print(" COMPREHENSIVE TEST SUITE FOR MultiDiscreteToDiscreteWrapper")
    print("="*70)

    test_suite = TestMultiDiscreteWrapper()

    # Non-interactive tests
    test_suite.test_wrapper_initialization()
    test_suite.test_action_conversion()
    test_suite.test_action_meanings()
    test_suite.test_step_execution()
    test_suite.test_reset_through_wrapper()
    test_suite.test_reward_and_termination()
    test_suite.test_rendering_through_wrapper()
    test_suite.test_single_vs_multi_agent()
    test_suite.test_edge_cases()
    test_suite.test_performance()

    print("\n" + "="*70)
    print(" ALL AUTOMATED TESTS PASSED! ✅")
    print("="*70)

    # Ask if user wants to run interactive test
    if not skip_interactive:
        response = input("\nRun interactive control test? (y/n): ")
        if response.lower() == 'y':
            test_suite.test_interactive_control()


if __name__ == "__main__":
    import sys
    skip_interactive = '--no-interactive' in sys.argv
    run_all_tests(skip_interactive=skip_interactive)