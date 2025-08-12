"""
Demo script to run and visualize the Multi-Agent SLAM Environment
with complete separation between environment and agents.
"""

import numpy as np
import time
from planner.simulation.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv
from planner.agents import RandomAgent, FrontierAgent, HybridAgent


def run_random_agent_demo():
    """Run a demo with random agent."""
    print("=== Random Agent Demo ===\n")

    # Create environment (no controller parameters!)
    env = MultiAgentSLAMGymEnv(
        width=32, height=32, num_drones=4, num_entry_points=2,
        camera_range=1, fov=45, max_steps=3000,
        render_mode='human', randomize=True,
        save_interval=50,                
        save_dir="runs/frontier_002",     
        save_format="png",              
        save_true_map=True,
        save_global_map=True,
        save_per_drone=False             
    )


    # Create agent separately
    agent = RandomAgent(num_agents=env.num_drones)

    print(f"Environment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Sensor range: {env.camera_range}")
    print(f"  FOV: {env.fov}°")
    print(f"  Agent: Random exploration")
    print("\nRunning random exploration...")

    # Reset
    observations, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    total_rewards = {i: 0.0 for i in env.agents}
    start_time = time.time()

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Get actions from agent
        actions = agent.get_actions(observations, info)

        # Step environment
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
    print(f"\nRewards by drone:")
    for i, total in enumerate(total_rewards.values()):
        print(f"  Drone {i}: {total:.2f}")
    print(f"\nDiscoveries by drone: {list(info['drone_discoveries'].values())}")

    # Keep window open for a moment
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_frontier_agent_demo():
    """Run a demo with the intelligent frontier-based agent."""
    print("\n=== Frontier-Based Agent Demo ===\n")

    # Create environment (no controller parameters!)
    env = MultiAgentSLAMGymEnv(
        width=32, height=32, num_drones=4, num_entry_points=2,
        camera_range=1, fov=45, max_steps=3000,
        render_mode='human', randomize=True,
        save_interval=50,                
        save_dir="runs/frontier_002",     
        save_format="png",              
        save_true_map=True,
        save_global_map=True,
        save_per_drone=False             
    )



    # Create frontier agent separately
    agent = FrontierAgent(num_agents=env.num_drones, camera_range=env.camera_range)

    print(f"Environment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Agent: Frontier-based exploration")
    print("\nAgent will coordinate all drone movements...")

    # Reset
    observations, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Get actions from agent
        actions = agent.get_actions(observations, info)

        # Step environment
        observations, rewards, dones, truncated, info = env.step(actions)

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
    print(f"\nRewards by drone:")
    for i, total in enumerate(total_rewards.values()):
        print(f"  Drone {i}: {total:.2f}")
    print(f"\nDiscoveries by drone: {list(info['drone_discoveries'].values())}")
    print(f"Efficiency: {step / (info['exploration_progress'] * env.width * env.height):.2f} steps per cell explored")

    # Keep window open for a moment
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_hybrid_agent_demo():
    """Run a demo with hybrid agent (mix of frontier and random)."""
    print("\n=== Hybrid Agent Demo ===\n")

    # Create environment (no controller parameters!)
    env = MultiAgentSLAMGymEnv(
        width=32, height=32, num_drones=4, num_entry_points=2,
        camera_range=1, fov=45, max_steps=3000,
        render_mode='human', randomize=True,
        save_interval=50,                
        save_dir="runs/frontier_002",     
        save_format="png",              
        save_true_map=True,
        save_global_map=True,
        save_per_drone=False             
    )



    # Create hybrid agent (50% frontier, 50% random)
    agent = HybridAgent(
        num_agents=env.num_drones,
        camera_range=env.camera_range,
        frontier_ratio=0.5
    )

    print(f"Environment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Drones: {env.num_drones}")
    print(f"  Agent: Hybrid (50% frontier, 50% random)")
    print(f"  Frontier drones: {sorted(agent.frontier_agents)}")
    print("\nRunning hybrid exploration...")

    # Reset
    observations, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Get actions from agent
        actions = agent.get_actions(observations, info)

        # Step environment
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

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final exploration: {info['exploration_progress']:.1%}")
    print(f"\nRewards by drone:")
    for i, total in enumerate(total_rewards.values()):
        print(f"  Drone {i}: {total:.2f}")
    print(f"\nDiscoveries by drone: {list(info['drone_discoveries'].values())}")

    # Keep window open
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_custom_scenario():
    """Run a custom scenario with user-selected agent type."""
    print("\n=== Custom Scenario Demo ===\n")

    # Get user preferences
    print("Configure your scenario:")
    width = int(input("Map width (10-50): ") or "20")
    height = int(input("Map height (10-50): ") or "20")
    num_drones = int(input("Number of drones (1-6): ") or "2")

    # Ask for agent type
    print("\nAgent type:")
    print("1. Random agent")
    print("2. Frontier-based agent")
    print("3. Hybrid agent")
    agent_type = input("Select agent (1-3): ") or "2"

    # Create environment (no controller parameters!)
    env = MultiAgentSLAMGymEnv(
        width=32, height=32, num_drones=4, num_entry_points=2,
        camera_range=1, fov=45, max_steps=3000,
        render_mode='human', randomize=True,
        save_interval=50,                
        save_dir="runs/frontier_002",     
        save_format="png",              
        save_true_map=True,
        save_global_map=True,
        save_per_drone=False             
    )



    # Create agent based on selection
    if agent_type == "1":
        agent = RandomAgent(num_agents=num_drones)
        agent_name = "Random"
    elif agent_type == "2":
        agent = FrontierAgent(num_agents=num_drones, camera_range=env.camera_range)
        agent_name = "Frontier-based"
    else:
        agent = HybridAgent(num_agents=num_drones, camera_range=env.camera_range)
        agent_name = "Hybrid"

    print(f"\nCustom environment created!")
    print(f"Agent: {agent_name}")
    print("Close window to quit\n")

    # Reset
    observations, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    running = True
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    while running and step < env.max_steps:
        # Get actions from agent
        actions = agent.get_actions(observations, info)

        # Step environment
        observations, rewards, dones, truncated, info = env.step(actions)

        # Update rewards
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

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
    print(f"\nRewards by drone:")
    for i, total in enumerate(total_rewards.values()):
        print(f"  Drone {i}: {total:.2f}")
    print(f"\nDiscoveries by drone: {list(info['drone_discoveries'].values())}")

    env.close()


def run_with_loaded_map():
    """Run a demo with a map loaded from file."""
    print("\n=== Demo with Loaded Map ===\n")

    import os
    import glob

    # Map directory
    map_dir = "/home/user1/TheAgency/resources/planner/maps"

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

    print("\nAgent type:")
    print("1. Random agent")
    print("2. Frontier-based agent")
    print("3. Hybrid agent")
    agent_type = input("Select agent (1-3): ") or "2"

    # Create environment (no controller parameters!)
    env = MultiAgentSLAMGymEnv(
        width=width,
        height=height,
        num_drones=num_drones,
        num_entry_points=2,
        camera_range=10,
        fov=60,
        max_steps=5000,
        render_mode='human',
        randomize=False,
        map_path=selected_map,
        save_interval=20,                
        save_dir="runs/frontier_002",     
        save_format="png",              
        save_true_map=True,
        save_global_map=True,
        save_per_drone=False             
    )



    # Create agent
    if agent_type == "1":
        agent = RandomAgent(num_agents=num_drones)
        agent_name = "Random"
    elif agent_type == "2":
        agent = FrontierAgent(num_agents=num_drones, camera_range=env.camera_range)
        agent_name = "Frontier-based"
    else:
        agent = HybridAgent(num_agents=num_drones, camera_range=env.camera_range)
        agent_name = "Hybrid"

    print(f"\nEnvironment created with loaded map!")
    print(f"Drones: {num_drones}")
    print(f"Agent: {agent_name}")
    print("\nStarting simulation...")

    # Reset
    observations, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_rewards = {i: 0.0 for i in env.agents}

    # Main loop
    running = True
    while running and step < env.max_steps:
        # Get actions from agent
        actions = agent.get_actions(observations, info)

        # Step environment
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
    print(f"\nRewards by drone:")
    for i, total in enumerate(total_rewards.values()):
        print(f"  Drone {i}: {total:.2f}")
    print(f"\nDiscoveries by drone: {list(info['drone_discoveries'].values())}")

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
        print("1. Random agent exploration")
        print("2. Frontier-based agent (intelligent)")
        print("3. Hybrid agent (mixed strategies)")
        print("4. Custom scenario")
        print("5. Load map from file")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ")

        if choice == '1':
            run_random_agent_demo()
        elif choice == '2':
            run_frontier_agent_demo()
        elif choice == '3':
            run_hybrid_agent_demo()
        elif choice == '4':
            run_custom_scenario()
        elif choice == '5':
            run_with_loaded_map()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")