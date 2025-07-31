"""
Demo script to run and visualize the Multi-Agent SLAM Environment
"""

import numpy as np
import time
from planner.simulation.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv

def run_random_exploration_demo():
    """Run a demo with random agent actions."""
    print("=== Random Exploration Demo ===\n")

    # Create environment
    env = MultiAgentSLAMGymEnv(
        width=25,
        height=25,
        num_drones=3,
        num_entry_points=2,
        camera_range=10,
        fov=60,
        max_steps=3000,
        render_mode='human',
        randomize=True,
        use_controller=False  # Manual control
    )

    print(f"Environment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Sensor range: {env.camera_range}")
    print(f"  FOV: {env.fov}°")
    print("\nRunning random exploration...")

    # Reset
    observations, info = env.reset()

    # Initialize metrics
    step = 0
    total_rewards = {i: 0.0 for i in env.agents}
    start_time = time.time()

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Random actions for each drone
        actions = {}
        for agent_id in env.agents:
            if observations[agent_id]['active']:
                # Bias towards forward movement for better exploration
                if np.random.random() < 0.6:
                    actions[agent_id] = 0  # FORWARD
                else:
                    actions[agent_id] = np.random.choice([1, 2, 3])  # TURN_LEFT, TURN_RIGHT, STAY
            else:
                actions[agent_id] = 3  # STAY if not active

        # Step
        observations, rewards, dones, truncated, info = env.step(actions)

        # Update rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:6.1f}s | Progress: {info['exploration_progress']:6.1%} | "
                  f"Discoveries: {list(info['drone_discoveries'].values())}")

        step += 1

        # Check completion
        if info['exploration_progress'] >= 1.0:
            print(f"\n✅ Exploration completed in {step} steps!")
            running = False

        if all(dones.values()):
            print(f"\nAll drones finished.")
            running = False

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final exploration: {info['exploration_progress']:.1%}")
    print(f"Rewards by drone: {[f'{r:.2f}' for r in total_rewards.values()]}")
    print(f"Discoveries by drone: {list(info['drone_discoveries'].values())}")

    # Keep window open for a moment
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_controller_demo():
    """Run a demo with the intelligent controller."""
    print("\n=== Frontier-Based Controller Demo ===\n")

    # Create environment with controller
    env = MultiAgentSLAMGymEnv(
        width=32,
        height=32,
        num_drones=4,
        num_entry_points=2,
        camera_range=10,
        fov=45,
        max_steps=3000,
        render_mode='human',
        randomize=True,
        use_controller=True,
        controller_mode='frontier'
    )

    print(f"Environment created with intelligent controller:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Controller: Frontier-based exploration")
    print("\nController will coordinate all drone movements...")

    # Reset
    observations, info = env.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Let controller handle everything
        observations, rewards, dones, truncated, info = env.step({})

        # Update total rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:6.1f}s | Progress: {info['exploration_progress']:6.1%} | "
                  f"Discoveries: {list(info['drone_discoveries'].values())}")

        step += 1

        # Check completion
        if info['exploration_progress'] >= 1.0:
            print(f"\n✅ Exploration completed in {step} steps!")
            running = False

        if all(dones.values()):
            print(f"\nAll drones finished.")
            running = False

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final exploration: {info['exploration_progress']:.1%}")
    print(f"Total rewards by drone: {[f'{r:.2f}' for r in total_rewards.values()]}")
    print(f"Final discoveries by drone: {list(info['drone_discoveries'].values())}")
    print(f"Efficiency: {step / (info['exploration_progress'] * env.width * env.height):.2f} steps per cell explored")

    # Keep window open for a moment
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_custom_scenario():
    """Run a custom scenario with specific parameters."""
    print("\n=== Custom Scenario Demo ===\n")

    # Get user preferences
    print("Configure your scenario:")
    width = int(input("Map width (10-50): ") or "20")
    height = int(input("Map height (10-50): ") or "20")
    num_drones = int(input("Number of drones (1-6): ") or "2")

    # Ask for control mode
    print("\nControl mode:")
    print("1. Semi-random (drone 0 explores systematically, others random)")
    print("2. All random")
    print("3. Mixed strategies")
    control_mode = input("Select mode (1-3): ") or "1"

    # Create environment
    env = MultiAgentSLAMGymEnv(
        width=width,
        height=height,
        num_drones=num_drones,
        num_entry_points=max(1, num_drones // 2),
        camera_range=8,
        fov=60,
        max_steps=5000,
        render_mode='human',
        randomize=True,
        use_controller=False
    )

    print(f"\nCustom environment created!")
    print(f"Control mode: {['Semi-random', 'All random', 'Mixed'][int(control_mode)-1]}")
    print("Close window to quit\n")

    # Reset
    observations, info = env.reset()

    # Initialize metrics
    step = 0
    running = True
    start_time = time.time()
    last_positions = {i: observations[i]['position'].copy() for i in range(num_drones)}
    stuck_counters = {i: 0 for i in range(num_drones)}

    while running and step < env.max_steps:
        actions = {}

        if control_mode == "1":  # Semi-random
            # Drone 0: Systematic exploration
            if observations[0]['active']:
                # If stuck, turn
                current_pos = observations[0]['position']
                if np.array_equal(current_pos, last_positions[0]):
                    stuck_counters[0] += 1
                    if stuck_counters[0] > 3:
                        actions[0] = np.random.choice([1, 2])  # Turn
                        stuck_counters[0] = 0
                    else:
                        actions[0] = 0  # Try forward again
                else:
                    stuck_counters[0] = 0
                    # Explore systematically
                    if np.random.random() < 0.8:
                        actions[0] = 0  # Forward
                    else:
                        actions[0] = np.random.choice([1, 2])  # Turn
                last_positions[0] = current_pos.copy()

            # Other drones: Random
            for agent_id in range(1, num_drones):
                if observations[agent_id]['active']:
                    actions[agent_id] = np.random.choice([0, 0, 0, 1, 2, 3])

        elif control_mode == "2":  # All random
            for agent_id in range(num_drones):
                if observations[agent_id]['active']:
                    actions[agent_id] = np.random.choice([0, 0, 0, 1, 2, 3])

        else:  # Mixed strategies
            for agent_id in range(num_drones):
                if observations[agent_id]['active']:
                    if agent_id % 2 == 0:
                        # Even drones: Systematic
                        if np.random.random() < 0.7:
                            actions[agent_id] = 0
                        else:
                            actions[agent_id] = np.random.choice([1, 2])
                    else:
                        # Odd drones: More turning
                        actions[agent_id] = np.random.choice([0, 1, 2, 3], p=[0.4, 0.25, 0.25, 0.1])

        # Step
        observations, rewards, dones, truncated, info = env.step(actions)

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:5.1f}s | Progress: {info['exploration_progress']:6.1%}")

        step += 1

        # Check completion
        if info['exploration_progress'] >= 1.0:
            print(f"\n✅ Exploration completed in {step} steps!")
            running = False

        # Allow window close
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final exploration: {info['exploration_progress']:.1%}")
    print(f"Discoveries by drone: {list(info['drone_discoveries'].values())}")

    env.close()


def run_with_loaded_map():
    """Run a demo with a map loaded from file."""
    print("\n=== Demo with Loaded Map ===\n")

    import os
    import glob

    # Map directory
    map_dir = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps"

    # Check if directory exists
    if not os.path.exists(map_dir):
        print(f"Map directory not found: {map_dir}")
        print("Please ensure the directory exists and contains map files.")
        return

    # Find available maps
    map_files = glob.glob(os.path.join(map_dir, "*.txt"))

    if not map_files:
        print(f"No .txt map files found in {map_dir}")
        return

    # Display available maps
    print("Available maps:")
    for i, map_file in enumerate(map_files):
        map_name = os.path.basename(map_file)
        print(f"{i+1}. {map_name}")

    # Let user select
    choice = input(f"\nSelect map (1-{len(map_files)}): ")
    try:
        map_index = int(choice) - 1
        if 0 <= map_index < len(map_files):
            selected_map = map_files[map_index]
        else:
            print("Invalid selection")
            return
    except ValueError:
        print("Invalid selection")
        return

    print(f"\nLoading map: {os.path.basename(selected_map)}")

    # Load the map to get its dimensions
    try:
        import numpy as np
        loaded_grid = np.loadtxt(selected_map, dtype=np.int8)
        height, width = loaded_grid.shape
        print(f"Map dimensions: {width}x{height}")
    except Exception as e:
        print(f"Error loading map: {e}")
        return

    # Ask for configuration
    print("\nConfiguration:")
    num_drones = int(input("Number of drones (1-6): ") or "3")
    use_controller = input("Use intelligent controller? (y/n): ").lower() == 'y'

    # Create environment with loaded map
    env = MultiAgentSLAMGymEnv(
        width=width,
        height=height,
        num_drones=num_drones,
        num_entry_points=2,  # Will use existing entry points from map
        camera_range=10,
        fov=60,
        max_steps=5000,
        render_mode='human',
        randomize=False,  # Don't randomize when loading a map
        map_path=selected_map,
        use_controller=use_controller,
        controller_mode='frontier' if use_controller else None
    )

    print(f"\nEnvironment created with loaded map!")
    print(f"Drones: {num_drones}")
    print(f"Controller: {'Enabled (Frontier-based)' if use_controller else 'Disabled (Random actions)'}")
    print("\nStarting simulation...")

    # Reset
    observations, info = env.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    # Main loop
    running = True
    while running and step < env.max_steps:
        if use_controller:
            # Let controller handle actions
            observations, rewards, dones, truncated, info = env.step({})
        else:
            # Random actions
            actions = {}
            for agent_id in env.agents:
                if observations[agent_id]['active']:
                    # Bias towards forward movement
                    if np.random.random() < 0.6:
                        actions[agent_id] = 0  # FORWARD
                    else:
                        actions[agent_id] = np.random.choice([1, 2, 3])

            observations, rewards, dones, truncated, info = env.step(actions)

        # Update rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:6.1f}s | Progress: {info['exploration_progress']:6.1%}")

        step += 1

        # Check completion
        if info['exploration_progress'] >= 1.0:
            print(f"\n✅ Exploration completed in {step} steps!")
            running = False

        if all(dones.values()):
            print(f"\nAll drones finished.")
            running = False

        # Allow window close
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Map: {os.path.basename(selected_map)}")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final exploration: {info['exploration_progress']:.1%}")
    print(f"Total rewards by drone: {[f'{r:.2f}' for r in total_rewards.values()]}")
    print(f"Discoveries by drone: {list(info['drone_discoveries'].values())}")

    # Keep window open briefly
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    print("Multi-Agent SLAM Environment Demo")
    print("=" * 50)

    while True:
        print("\nSelect demo:")
        print("1. Random exploration (multiple drones)")
        print("2. Intelligent controller (frontier-based)")
        print("3. Custom scenario (manual control)")
        print("4. Load map from file")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ")

        if choice == '1':
            run_random_exploration_demo()
        elif choice == '2':
            run_controller_demo()
        elif choice == '3':
            run_custom_scenario()
        elif choice == '4':
            run_with_loaded_map()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

        if choice in ['1', '2', '3', '4']:
            input("\nPress Enter to continue...")