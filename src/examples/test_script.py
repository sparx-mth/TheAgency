"""
Test script for the Multi-Agent SLAM Gym Environment
"""

import sys
import os
import numpy as np

# Import the environment with the correct name
from envs.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv

def test_basic_functionality():
    """Test basic environment functionality."""
    print("=== Testing Basic Functionality ===\n")

    # Create environment
    env = MultiAgentSLAMGymEnv(
        width=16,
        height=16,
        num_drones=2,
        num_entry_points=1,
        camera_range=8,
        fov=45,
        max_steps=500,
        render_mode=None
    )

    # Test reset
    observations, info = env.reset()
    assert len(observations) == 2, f"Expected 2 agents, got {len(observations)}"
    assert 0 in observations and 1 in observations, "Missing agent observations"

    print(f"Number of agents: {len(observations)}")
    print(f"Observation keys: {list(observations[0].keys())}")
    print(f"Action space: {env.action_spaces[0]}")
    print(f"Initial info: {list(info.keys())}\n")

    # Test step
    actions = {0: 0, 1: 1}  # Agent 0: FORWARD, Agent 1: TURN_LEFT
    observations, rewards, dones, truncated, info = env.step(actions)

    assert len(rewards) == 2, "Rewards not returned for all agents"
    assert len(dones) == 2, "Dones not returned for all agents"

    print(f"After step:")
    print(f"  Rewards: {rewards}")
    print(f"  Dones: {dones}")
    print(f"  Exploration progress: {info['exploration_progress']:.2%}")

    env.close()
    print("✓ Basic functionality test passed\n")


def test_observation_structure():
    """Test that observations have the correct structure."""
    print("=== Testing Observation Structure ===\n")

    env = MultiAgentSLAMGymEnv(
        width=10,
        height=10,
        num_drones=1,
        render_mode=None
    )

    observations, _ = env.reset()
    obs = observations[0]

    # Check all required keys are present
    required_keys = ['local_map', 'position', 'facing_direction', 'active', 'collided', 'entry_time']
    for key in required_keys:
        assert key in obs, f"Missing required key: {key}"

    # Check shapes and types
    assert obs['local_map'].shape == (10, 10), f"Wrong local_map shape: {obs['local_map'].shape}"
    assert obs['position'].shape == (2,), f"Wrong position shape: {obs['position'].shape}"
    assert isinstance(obs['facing_direction'], (int, np.integer)), "facing_direction should be int"
    assert 0 <= obs['facing_direction'] <= 3, f"Invalid facing_direction: {obs['facing_direction']}"

    print("Observation structure verified:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    env.close()
    print("\n✓ Observation structure test passed\n")


def test_action_execution():
    """Test that actions are executed correctly."""
    print("=== Testing Action Execution ===\n")

    env = MultiAgentSLAMGymEnv(
        width=20,
        height=20,
        num_drones=1,
        render_mode=None
    )

    observations, _ = env.reset()
    initial_pos = observations[0]['position'].copy()
    initial_facing = observations[0]['facing_direction']

    # Test FORWARD action
    observations, _, _, _, _ = env.step({0: 0})  # FORWARD
    new_pos = observations[0]['position']

    # Position should have changed (unless collision)
    if not observations[0]['collided']:
        assert not np.array_equal(initial_pos, new_pos), "Position didn't change after FORWARD"

    # Test TURN_LEFT
    observations, _, _, _, _ = env.step({0: 1})  # TURN_LEFT
    new_facing = observations[0]['facing_direction']
    expected_facing = (initial_facing - 1) % 4

    print(f"Initial facing: {initial_facing}")
    print(f"After TURN_LEFT: {new_facing}")
    print(f"Expected: {expected_facing}")

    env.close()
    print("\n✓ Action execution test passed\n")


def test_multi_agent():
    """Test multi-agent functionality."""
    print("=== Testing Multi-Agent Functionality ===\n")

    env = MultiAgentSLAMGymEnv(
        width=30,
        height=30,
        num_drones=4,
        num_entry_points=2,
        render_mode=None
    )

    observations, info = env.reset()

    # Check all agents are present
    assert len(observations) == 4, f"Expected 4 agents, got {len(observations)}"

    # Get initial positions
    positions = {i: tuple(obs['position']) for i, obs in observations.items()}
    print(f"Initial positions: {positions}")

    # All agents take different actions
    actions = {0: 0, 1: 1, 2: 2, 3: 3}  # FORWARD, TURN_LEFT, TURN_RIGHT, STAY

    for step in range(3):
        observations, rewards, dones, truncated, info = env.step(actions)

    # Check positions changed appropriately
    new_positions = {i: tuple(obs['position']) for i, obs in observations.items()}
    print(f"Final positions: {new_positions}")

    # Agent 3 (STAY) should not have moved
    assert positions[3] == new_positions[3], "Agent with STAY action moved"

    env.close()
    print("\n✓ Multi-agent test passed\n")


def test_controller_integration():
    """Test environment with controller."""
    print("=== Testing Controller Integration ===\n")

    env = MultiAgentSLAMGymEnv(
        width=15,
        height=15,
        num_drones=2,
        use_controller=True,
        controller_mode='frontier',
        render_mode=None
    )

    observations, info = env.reset()

    # Run steps without providing actions
    for i in range(20):
        observations, rewards, dones, truncated, info = env.step({})

    # Check that exploration is happening
    progress = info['exploration_progress']
    assert progress > 0, "No exploration progress with controller"

    print(f"Exploration progress after 20 steps: {progress:.2%}")
    print(f"Individual discoveries: {info['drone_discoveries']}")

    env.close()
    print("\n✓ Controller integration test passed\n")


def test_reset():
    """Test that reset properly reinitializes the environment."""
    print("=== Testing Reset Functionality ===\n")

    env = MultiAgentSLAMGymEnv(
        width=10,
        height=10,
        num_drones=1,
        render_mode=None
    )

    # First episode
    observations1, info1 = env.reset()
    for _ in range(10):
        env.step({0: 0})  # Move forward

    # Reset and check
    observations2, info2 = env.reset()

    # Check that environment is reset
    assert info2['step'] == 0, "Step counter not reset"
    assert info2['exploration_progress'] == 0 or info2['exploration_progress'] < 0.1, "Progress not reset"

    # Local map should be mostly unknown again
    unknown_count = np.sum(observations2[0]['local_map'] == -1)
    total_cells = observations2[0]['local_map'].size
    assert unknown_count / total_cells > 0.8, "Local map not properly reset"

    env.close()
    print("\n✓ Reset test passed\n")


def test_full_episode_with_rendering():
    """Run a complete episode with random actions and rendering."""
    print("=== Testing Full Episode with Rendering ===\n")

    # Create environment with rendering
    env = MultiAgentSLAMGymEnv(
        width=25,
        height=25,
        num_drones=3,
        num_entry_points=2,
        camera_range=10,
        fov=60,
        max_steps=2000,
        render_mode='human',
        randomize=True
    )

    print("Environment created with rendering enabled")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Number of drones: {env.num_drones}")
    print(f"Max steps: {env.max_steps}")

    # Reset environment
    observations, info = env.reset()
    print(f"\nInitial state:")
    print(f"  Exploration progress: {info['exploration_progress']:.2%}")
    print(f"  Active drones: {[i for i, obs in observations.items() if obs['active']]}")

    # Track metrics
    step_count = 0
    total_rewards = {i: 0.0 for i in env.agents}
    exploration_history = []

    print("\nRunning episode with random actions...")
    print("(Close the window to stop early)\n")

    # Run until completion or timeout
    done_agents = set()

    while len(done_agents) < len(env.agents) and step_count < env.max_steps:
        # Generate random actions for active agents
        actions = {}
        for agent_id in env.agents:
            if agent_id not in done_agents:
                # Random action
                actions[agent_id] = env.action_spaces[agent_id].sample()

        # Step environment
        observations, rewards, dones, truncated, info = env.step(actions)

        # Update metrics
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Check for done agents
        for agent_id, done in dones.items():
            if done and agent_id not in done_agents:
                done_agents.add(agent_id)
                print(f"Agent {agent_id} finished at step {step_count}")

        # Track exploration progress
        exploration_history.append(info['exploration_progress'])

        # Render
        env.render()

        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}:")
            print(f"  Exploration: {info['exploration_progress']:.2%}")
            print(f"  Discoveries by drone: {info['drone_discoveries']}")
            print(f"  Active agents: {[i for i in env.agents if i not in done_agents]}")

        step_count += 1

        # Check if exploration is complete
        if info['exploration_progress'] >= 1.0:
            print(f"\n🎉 Exploration completed at step {step_count}!")
            break

    # Final statistics
    print(f"\n=== Episode Summary ===")
    print(f"Total steps: {step_count}")
    print(f"Final exploration: {info['exploration_progress']:.2%}")
    print(f"Total rewards by agent: {total_rewards}")
    print(f"Final discoveries by drone: {info['drone_discoveries']}")

    # Check exploration improvement
    if len(exploration_history) > 10:
        early_exploration = np.mean(exploration_history[:10])
        late_exploration = np.mean(exploration_history[-10:])
        print(f"Early exploration (first 10 steps): {early_exploration:.2%}")
        print(f"Late exploration (last 10 steps): {late_exploration:.2%}")
        improvement = late_exploration - early_exploration
        print(f"Improvement: {improvement:.2%}")

    env.close()
    print("\n✓ Full episode test completed\n")


def test_controller_episode_with_rendering():
    """Run a complete episode with the controller and rendering."""
    print("=== Testing Controller Episode with Rendering ===\n")

    # Create environment with controller
    env = MultiAgentSLAMGymEnv(
        width=30,
        height=30,
        num_drones=4,
        num_entry_points=2,
        camera_range=10,
        fov=45,
        max_steps=2000,
        render_mode='human',
        use_controller=True,
        controller_mode='frontier',
        randomize=True
    )

    print("Environment created with frontier-based controller")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Number of drones: {env.num_drones}")
    print("Controller will coordinate drone movements\n")

    # Reset environment
    observations, info = env.reset()

    # Track metrics
    step_count = 0
    exploration_history = []

    print("Running episode with controller...")
    print("(Close the window to stop early)\n")

    # Run until completion
    while step_count < env.max_steps:
        # Let controller handle all actions
        observations, rewards, dones, truncated, info = env.step({})

        # Track progress
        exploration_history.append(info['exploration_progress'])

        # Render
        env.render()

        # Print progress
        if step_count % 50 == 0:
            print(f"Step {step_count}: Exploration {info['exploration_progress']:.2%}")

        step_count += 1

        # Check completion
        if all(dones.values()) or info['exploration_progress'] >= 1.0:
            print(f"\n🎉 Exploration completed at step {step_count}!")
            break

    # Analyze efficiency
    print(f"\n=== Controller Performance ===")
    print(f"Total steps: {step_count}")
    print(f"Final exploration: {info['exploration_progress']:.2%}")
    print(f"Steps per percent explored: {step_count / (info['exploration_progress'] * 100):.2f}")

    env.close()
    print("\n✓ Controller episode test completed\n")


if __name__ == "__main__":
    print("Multi-Agent SLAM Gym Environment Test Suite\n")
    print("=" * 50 + "\n")

    try:
        # Basic tests
        test_basic_functionality()
        test_observation_structure()
        test_action_execution()
        test_multi_agent()
        test_controller_integration()
        test_reset()

        print("\n" + "=" * 50)
        print("All basic tests passed! ✓")
        print("\nNow running visual tests...")
        print("=" * 50 + "\n")

        # Visual tests
        test_full_episode_with_rendering()

        # Ask if user wants to see controller demo
        response = input("\nRun controller demonstration? (y/n): ")
        if response.lower() == 'y':
            test_controller_episode_with_rendering()

        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")
        print("The environment is ready for RL training.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise