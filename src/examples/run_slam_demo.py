"""
run_slam_demo.py - Updated for new unified state/action space
"""

import os
import sys
import glob
import time
import numpy as np
import warnings

# Suppress Gym deprecation warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Add src directory to path if needed
src_path = os.path.join(os.path.dirname(__file__), 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

# Import environment - NO LONGER NEED SingleAgentWrapper
from environments.slam_env import MultiAgentSLAMEnv

# Import agents
from agents.random_agent import RandomAgent
from agents.frontier_agent import FrontierAgent

# Import sensors for custom configurations
from sensors.camera_sensor import CameraSensor
from sensors.lidar_sensor import LidarSensor


def get_camera_parameters():
    """
    Get camera range and FOV from user input.

    Returns:
        tuple: (camera_range, fov_deg)
    """
    print("\nCamera Configuration:")

    # Get camera range
    try:
        camera_range = int(input("Camera range (3-20) [default=10]: ") or "10")
        camera_range = max(3, min(20, camera_range))
    except ValueError:
        camera_range = 10

    # Get FOV
    try:
        fov_deg = int(input("Field of View in degrees (15-120) [default=60]: ") or "60")
        fov_deg = max(15, min(120, fov_deg))
    except ValueError:
        fov_deg = 60

    return camera_range, fov_deg


def run_single_agent_demo():
    """Run a demo with a single agent."""
    print("=== Single Agent Demo ===\n")

    # Get camera parameters
    camera_range, fov_deg = get_camera_parameters()

    # Create environment with num_agents=1
    env = MultiAgentSLAMEnv(
        width=20,
        height=20,
        num_agents=1,  # Single agent
        max_steps=500,
        sensor_config={0: CameraSensor(max_range=camera_range, fov_deg=fov_deg)},
        randomize=True,
        render_mode='human',
        discovery_reward=0.1,
        collision_penalty=-1.0,
    )

    # Create agent with the same camera range for sensor-aware exploration
    agent = FrontierAgent(num_agents=1, camera_range=camera_range)

    print(f"\nEnvironment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Sensor: Camera (range={camera_range}, fov={fov_deg}°)")
    print(f"  Agent: Frontier-based exploration")
    print("\nRunning single agent exploration...")

    # Reset
    obs, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    total_reward = 0.0
    start_time = time.time()

    # Main loop
    done = False
    truncated = False
    while not done and not truncated and step < env.max_steps:
        # Get action from agent (now expects array)
        action = agent.get_actions(obs, info)

        # Step environment (now returns single reward)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:6.1f}s | Progress: {info['progress']:6.1%} | "
                  f"Reward: {total_reward:7.2f}")

        step += 1

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Camera settings: range={camera_range}, FOV={fov_deg}°")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final progress: {info['progress']:.1%}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Collisions: {info['collision_counts'][0]}")

    # Keep window open briefly
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_multi_agent_demo():
    """Run a demo with multiple agents using frontier strategy."""
    print("\n=== Multi-Agent Frontier Demo ===\n")

    # Get camera parameters
    camera_range, fov_deg = get_camera_parameters()

    # Configure sensors for all agents with the same parameters
    sensor_config = {
        i: CameraSensor(max_range=camera_range, fov_deg=fov_deg)
        for i in range(3)
    }

    # Create multi-agent environment
    env = MultiAgentSLAMEnv(
        width=30,
        height=30,
        num_agents=3,
        max_steps=800,
        sensor_config=sensor_config,
        randomize=True,
        render_mode='human',
    )

    # Create frontier agent to control all drones with the same camera range
    agent = FrontierAgent(num_agents=3, camera_range=camera_range)

    print(f"\nEnvironment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Number of agents: {env.num_agents}")
    print(f"  Sensor: Camera (range={camera_range}, fov={fov_deg}°) for all agents")
    print(f"  Agent strategy: Frontier-based exploration")
    print("\nAgent will coordinate all drone movements...")

    # Reset
    obs, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_reward = 0.0

    # Main loop
    done = False
    while not done and step < env.max_steps:
        # Get actions from agent (array of actions)
        actions = agent.get_actions(obs, info)

        # Step environment (now returns single reward)
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:6.1f}s | Progress: {info['progress']:6.1%} | Reward: {total_reward:7.2f}")

        step += 1

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Camera settings: range={camera_range}, FOV={fov_deg}°")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final progress: {info['progress']:.1%}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"\nCollisions by agent:")
    for i in range(env.num_agents):
        print(f"  Agent {i}: {info['collision_counts'][i]} collisions")

    # Keep window open
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_mixed_sensors_demo():
    """Run a demo with different sensor types on different drones."""
    print("\n=== Mixed Sensors Demo ===\n")

    print("Configure sensors for each agent:")

    # Get parameters for Agent 0 (narrow camera)
    print("\nAgent 0 - Narrow Camera:")
    try:
        range_0 = int(input("  Range (3-20) [default=10]: ") or "10")
        range_0 = max(3, min(20, range_0))
        fov_0 = int(input("  FOV (15-60) [default=45]: ") or "45")
        fov_0 = max(15, min(60, fov_0))
    except ValueError:
        range_0, fov_0 = 10, 45

    # Get parameters for Agent 1 (LIDAR)
    print("\nAgent 1 - LIDAR:")
    try:
        range_1 = int(input("  Range (3-20) [default=12]: ") or "12")
        range_1 = max(3, min(20, range_1))
        rays_1 = int(input("  Number of rays (60-360) [default=180]: ") or "180")
        rays_1 = max(60, min(360, rays_1))
    except ValueError:
        range_1, rays_1 = 12, 180

    # Get parameters for Agent 2 (wide camera)
    print("\nAgent 2 - Wide Camera:")
    try:
        range_2 = int(input("  Range (3-20) [default=8]: ") or "8")
        range_2 = max(3, min(20, range_2))
        fov_2 = int(input("  FOV (60-120) [default=90]: ") or "90")
        fov_2 = max(60, min(120, fov_2))
    except ValueError:
        range_2, fov_2 = 8, 90

    # Configure different sensors for each drone
    sensor_config = {
        0: CameraSensor(max_range=range_0, fov_deg=fov_0),      # Narrow camera
        1: LidarSensor(max_range=range_1, num_rays=rays_1),     # LIDAR
        2: CameraSensor(max_range=range_2, fov_deg=fov_2),      # Wide-angle camera
    }

    # Create environment with mixed sensors
    env = MultiAgentSLAMEnv(
        width=35,
        height=35,
        num_agents=3,
        max_steps=1000,
        sensor_config=sensor_config,
        randomize=True,
        render_mode='human',
    )

    # Create frontier agent (use max range for sensor-aware exploration)
    max_range = max(range_0, range_1, range_2)
    agent = FrontierAgent(num_agents=3, camera_range=max_range)

    print(f"\nEnvironment created:")
    print(f"  Map size: {env.width}x{env.height}")
    print(f"  Agents with different sensors:")
    print(f"    Agent 0: Camera (narrow, range={range_0}, fov={fov_0}°)")
    print(f"    Agent 1: LIDAR (range={range_1}, {rays_1} rays)")
    print(f"    Agent 2: Camera (wide, range={range_2}, fov={fov_2}°)")
    print("\nRunning coordinated exploration with mixed sensors...")

    # Reset
    obs, info = env.reset()
    agent.reset()

    print(f"Sensor types: {info['sensor_types']}")

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_reward = 0.0

    # Main loop
    done = False
    while not done and step < env.max_steps:
        # Get actions (array)
        actions = agent.get_actions(obs, info)

        # Step environment (single reward)
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        # Render
        env.render()

        # Progress update
        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Progress: {info['progress']:6.1%} | Reward: {total_reward:7.2f}")
            if step % 200 == 0:  # Detailed update less frequently
                for i in range(env.num_agents):
                    print(f"    Agent {i} ({info['sensor_types'][i]}): "
                          f"Collisions={info['collision_counts'][i]}")

        step += 1

    # Summary
    elapsed = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final progress: {info['progress']:.1%}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"\nPerformance by sensor type:")
    for i in range(env.num_agents):
        print(f"  Agent {i} ({info['sensor_types'][i]}): "
              f"Collisions={info['collision_counts'][i]}")

    # Keep window open
    for _ in range(30):
        env.render()
        time.sleep(0.1)

    env.close()


def run_benchmark():
    """Benchmark different agent strategies."""
    print("\n=== Agent Strategy Benchmark ===\n")

    # Get camera parameters for testing
    print("Configure camera for benchmark:")
    camera_range, fov_deg = get_camera_parameters()

    # Environment configuration
    env_config = {
        'width': 20,
        'height': 20,
        'num_agents': 1,  # Single agent for benchmark
        'max_steps': 400,
        'sensor_config': {0: CameraSensor(max_range=camera_range, fov_deg=fov_deg)},
        'randomize': True,
        'render_mode': None,  # No rendering for speed
    }

    # Agents to test
    agents_to_test = [
        ("Random (low bias)", RandomAgent(1, forward_bias=0.4)),
        ("Random (high bias)", RandomAgent(1, forward_bias=0.8)),
        ("Frontier", FrontierAgent(1, camera_range=camera_range)),
    ]

    # Number of episodes per agent
    num_episodes = 5

    print(f"\nRunning {num_episodes} episodes per agent...")
    print(f"Environment: {env_config['width']}x{env_config['height']}")
    print(f"Camera: range={camera_range}, FOV={fov_deg}°")
    print()

    results = {}

    for agent_name, agent in agents_to_test:
        print(f"Testing {agent_name}...")

        episode_metrics = []

        for episode in range(num_episodes):
            env = MultiAgentSLAMEnv(**env_config)
            obs, info = env.reset()
            agent.reset()

            total_reward = 0
            done = False
            truncated = False
            steps = 0

            while not done and not truncated and steps < env.max_steps:
                action = agent.get_actions(obs, info)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            episode_metrics.append({
                'reward': total_reward,
                'progress': info['progress'],
                'steps': steps,
                'collisions': info['collision_counts'][0],
            })

            env.close()
            print(f"  Episode {episode+1}: Reward={total_reward:.1f}, Progress={info['progress']*100:.1f}%")

        # Calculate averages
        results[agent_name] = {
            'avg_reward': np.mean([m['reward'] for m in episode_metrics]),
            'std_reward': np.std([m['reward'] for m in episode_metrics]),
            'avg_progress': np.mean([m['progress'] for m in episode_metrics]),
            'avg_steps': np.mean([m['steps'] for m in episode_metrics]),
            'avg_collisions': np.mean([m['collisions'] for m in episode_metrics]),
        }

    # Print results
    print("\n" + "=" * 60)
    print(f"Benchmark Results (averaged over {num_episodes} episodes)")
    print(f"Camera settings: range={camera_range}, FOV={fov_deg}°")
    print("=" * 60)

    for agent_name, metrics in results.items():
        print(f"\n{agent_name}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Avg Progress: {metrics['avg_progress']*100:.1f}%")
        print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
        print(f"  Avg Collisions: {metrics['avg_collisions']:.1f}")

    # Determine winner
    best_agent = max(results.items(), key=lambda x: x[1]['avg_reward'])
    print(f"\n🏆 Best performing agent: {best_agent[0]}")


def run_with_loaded_map():
    """Run a demo with a map loaded from file."""
    print("\n=== Load Custom Map ===\n")

    # Ask for map directory or use default
    default_dir = os.path.join(os.getcwd(), "maps")
    alt_dir = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps"

    print("Map directories to search:")
    print(f"1. Current directory/maps: {default_dir}")
    print(f"2. Project maps: {alt_dir}")
    print("3. Custom path")

    dir_choice = input("\nSelect directory (1-3) [default=1]: ") or "1"

    if dir_choice == "1":
        map_dir = default_dir
    elif dir_choice == "2":
        map_dir = alt_dir
    elif dir_choice == "3":
        map_dir = input("Enter map directory path: ").strip()
    else:
        map_dir = default_dir

    # Check if directory exists
    if not os.path.exists(map_dir):
        print(f"\n⚠️  Directory not found: {map_dir}")

        # Try to find maps in common locations
        search_paths = [
            os.path.join(os.getcwd(), "maps"),
            os.path.join(os.getcwd(), "resources", "maps"),
            os.path.join(os.path.dirname(__file__), "maps"),
            os.path.join(os.path.dirname(__file__), "..", "maps"),
        ]

        found_maps = []
        for path in search_paths:
            if os.path.exists(path):
                txt_files = glob.glob(os.path.join(path, "*.txt"))
                if txt_files:
                    found_maps.extend(txt_files)

        if found_maps:
            print(f"\nFound {len(found_maps)} map(s) in other locations:")
            for i, map_file in enumerate(found_maps[:10]):  # Show max 10
                print(f"{i+1}. {os.path.basename(map_file)} ({os.path.dirname(map_file)})")

            choice = input(f"\nSelect map (1-{min(10, len(found_maps))}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return

            try:
                map_index = int(choice) - 1
                if 0 <= map_index < len(found_maps):
                    selected_map = found_maps[map_index]
                else:
                    print("Invalid selection")
                    return
            except ValueError:
                print("Invalid selection")
                return
        else:
            print("No map files found. Please create .txt map files.")
            return
    else:
        # Find available maps in selected directory
        map_files = glob.glob(os.path.join(map_dir, "*.txt"))

        if not map_files:
            print(f"No .txt map files found in {map_dir}")
            return

        # Display available maps
        print(f"\nFound {len(map_files)} map(s):")
        for i, map_file in enumerate(map_files):
            map_name = os.path.basename(map_file)
            # Try to get map size
            try:
                test_map = np.loadtxt(map_file, dtype=np.int8)
                h, w = test_map.shape
                print(f"{i+1}. {map_name} ({w}x{h})")
            except:
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

    print(f"\n📁 Loading map: {os.path.basename(selected_map)}")

    # Load the map to get its dimensions
    try:
        loaded_grid = np.loadtxt(selected_map, dtype=np.int8)
        height, width = loaded_grid.shape
        print(f"✓ Map loaded successfully: {width}x{height}")

        # Analyze map
        unique_tiles = np.unique(loaded_grid)
        tile_names = {0: "Free", 1: "Wall", 2: "Entry", 3: "Door(C)", 4: "Door(O)", 5: "Window", 6: "OOB"}
        print("\nMap contains:")
        for tile in unique_tiles:
            count = np.sum(loaded_grid == tile)
            name = tile_names.get(tile, f"Unknown({tile})")
            print(f"  {name}: {count} tiles")

    except Exception as e:
        print(f"❌ Error loading map: {e}")
        return

    # Configuration
    print("\n=== Configuration ===")

    # Number of agents
    num_agents = int(input(f"Number of agents (1-6) [default=2]: ") or "2")
    num_agents = max(1, min(6, num_agents))

    # Agent type
    print("\nAgent strategy:")
    print("1. Random exploration")
    print("2. Frontier-based (intelligent)")
    agent_type = input("Select strategy (1-2) [default=2]: ") or "2"

    # Get camera parameters
    camera_range, fov_deg = get_camera_parameters()

    # Configure sensors for all agents
    sensor_config = {
        i: CameraSensor(max_range=camera_range, fov_deg=fov_deg)
        for i in range(num_agents)
    }

    # Create environment with loaded map
    print(f"\n🚁 Creating environment with loaded map...")

    env = MultiAgentSLAMEnv(
        width=width,  # Will be overridden by map file
        height=height,  # Will be overridden by map file
        num_agents=num_agents,
        max_steps=width * height * 3,
        map_path=selected_map,
        sensor_config=sensor_config,
        randomize=False,  # Don't randomize when loading a map
        render_mode='human',
    )

    # Create agent
    if agent_type == "1":
        agent = RandomAgent(num_agents=num_agents, forward_bias=0.7)
        agent_name = "Random"
    else:
        agent = FrontierAgent(num_agents=num_agents, camera_range=camera_range)
        agent_name = "Frontier"

    print(f"\n=== Simulation Ready ===")
    print(f"Map: {os.path.basename(selected_map)}")
    print(f"Size: {width}x{height}")
    print(f"Agents: {num_agents}")
    print(f"Strategy: {agent_name}")
    print(f"Sensors: Camera (range={camera_range}, FOV={fov_deg}°)")
    print("\nStarting in 2 seconds...")
    time.sleep(2)

    # Reset
    obs, info = env.reset()
    agent.reset()

    # Initialize metrics
    step = 0
    start_time = time.time()
    total_reward = 0.0

    # Main loop
    done = False
    truncated = False
    while not done and not truncated and step < env.max_steps:
        # Get actions
        actions = agent.get_actions(obs, info)

        # Step environment
        obs, reward, done, truncated, info = env.step(actions)
        total_reward += reward

        # Render
        env.render()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Time: {elapsed:5.1f}s | Progress: {info['progress']:6.1%} | "
                  f"Reward: {total_reward:7.2f}")

        step += 1

        # Allow window close
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Simulation Complete!")
    print(f"{'='*50}")
    print(f"Map: {os.path.basename(selected_map)}")
    print(f"Total steps: {step}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final progress: {info['progress']:.1%}")
    print(f"Discovered cells: {info['discovered_cells']}/{info['total_reachable']}")
    print(f"Total reward: {total_reward:.2f}")

    print(f"\nPerformance by agent:")
    for i in range(num_agents):
        sensor_info = info['sensor_types'][i] if 'sensor_types' in info else "camera"
        print(f"  Agent {i} ({sensor_info}): Collisions: {info['collision_counts'][i]}")

    # Keep window open
    print("\nPress Enter to close...")
    for _ in range(30):
        env.render()
        time.sleep(0.1)

        import pygame
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                env.close()
                return

    env.close()


def main_menu():
    """Main menu for the demo program."""
    print("\n" + "=" * 50)
    print("Multi-Agent SLAM Environment Demo")
    print("=" * 50)

    while True:
        print("\nSelect demo:")
        print("1. Single agent exploration")
        print("2. Multi-agent coordination")
        print("3. Mixed sensors demonstration")
        print("4. Load custom map")
        print("5. Benchmark agents")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            run_single_agent_demo()
        elif choice == '2':
            run_multi_agent_demo()
        elif choice == '3':
            run_mixed_sensors_demo()
        elif choice == '4':
            run_with_loaded_map()
        elif choice == '5':
            run_benchmark()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()