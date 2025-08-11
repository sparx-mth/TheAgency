"""
benchmark_runner.py

This script runs benchmarks for the SLAM environment with various configurations.
It tests different numbers of agents across multiple maps and saves results for analysis.
"""

import time
import os
import logging
import gc
import csv
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np

# Import your new environment and agents
from environments.slam_env import MultiAgentSLAMEnv
from agents.random_agent import RandomAgent
from agents.frontier_agent import FrontierAgent
from sensors.camera_sensor import CameraSensor


def parse_args():
    parser = argparse.ArgumentParser(description='SLAM Benchmark Runner')
    parser.add_argument('--map_count', type=int, default=10,
                        help='Number of maps to run')
    parser.add_argument('--drone_counts', nargs='+', type=int, default=[1, 2, 3],
                        help='Number of drones to test')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of iterations per configuration')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--maps_dir', type=str, default='maps',
                        help='Directory containing map files')
    parser.add_argument('--csv_name', type=str, default='slam_benchmark_results.csv',
                        help='Name of CSV results file')
    parser.add_argument('--log_name', type=str, default='slam_benchmark.log',
                        help='Name of log file')
    parser.add_argument('--write_header', action='store_true',
                        help='Whether to write header to CSV')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the simulation')
    parser.add_argument('--render_delay', type=float, default=0.0,
                        help='Delay between frames when rendering (seconds)')
    parser.add_argument('--agent_type', type=str, default='frontier',
                        choices=['random', 'frontier'],
                        help='Agent type: random or frontier')
    parser.add_argument('--camera_range', type=int, default=10,
                        help='Camera sensing range')
    parser.add_argument('--camera_fov', type=int, default=60,
                        help='Camera field of view in degrees')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum steps per episode')
    parser.add_argument('--max_time', type=int, default=90,
                        help='Maximum time in seconds before timeout')
    parser.add_argument('--map_width', type=int, default=32,
                        help='Width of the map')
    parser.add_argument('--map_height', type=int, default=32,
                        help='Height of the map')

    return parser.parse_args()


def run_single_episode(env, agent, max_time=90, render=False, render_delay=0.0):
    """
    Run a single episode with the given environment and agent.

    Args:
        env: The SLAM environment
        agent: The agent (RandomAgent or FrontierAgent)
        max_time: Maximum time in seconds before timeout
        render: Whether to render the environment
        render_delay: Delay between frames when rendering

    Returns:
        dict: Results including completion time, final progress, total reward, etc.
    """
    obs, info = env.reset()
    agent.reset()

    done = False
    truncated = False
    start_time = time.time()
    total_rewards = {i: 0.0 for i in range(env.num_agents)}
    step_count = 0

    while not done and not truncated:
        # Get actions from the agent
        actions = agent.get_actions(obs, info)

        # Step the environment
        if env.num_agents == 1:
            obs, reward, done, truncated, info = env.step(actions)
            total_rewards[0] += reward
        else:
            obs, rewards, dones, truncateds, info = env.step(actions)
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            done = any(dones.values())
            truncated = any(truncateds.values())

        if render:
            env.render()
            if render_delay > 0:
                time.sleep(render_delay)

        step_count += 1

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_time:
            return {
                'completed': False,
                'time': None,
                'progress': info['progress'],
                'steps': step_count,
                'total_reward': sum(total_rewards.values()),
                'avg_reward': sum(total_rewards.values()) / len(total_rewards),
                'collisions': sum(info['collision_counts']),
                'timeout': True
            }

    # Episode finished
    completion_time = time.time() - start_time
    completed = info['progress'] >= 0.99  # 99% completion threshold

    return {
        'completed': completed,
        'time': completion_time if completed else None,
        'progress': info['progress'],
        'steps': step_count,
        'total_reward': sum(total_rewards.values()),
        'avg_reward': sum(total_rewards.values()) / len(total_rewards),
        'collisions': sum(info['collision_counts']),
        'timeout': False
    }


def generate_random_map(width, height, num_agents, complexity=0.15):
    """
    Generate a random map for testing.

    Args:
        width: Map width
        height: Map height
        num_agents: Number of agents (determines entry points)
        complexity: Wall density (0-1)

    Returns:
        np.ndarray: Generated map
    """
    from core.constants import TileType

    grid = np.zeros((height, width), dtype=np.int8)

    # Add walls on borders
    grid[0, :] = TileType.WALL
    grid[-1, :] = TileType.WALL
    grid[:, 0] = TileType.WALL
    grid[:, -1] = TileType.WALL

    # Add random internal walls
    num_walls = int(width * height * complexity)
    for _ in range(num_walls):
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)
        grid[y, x] = TileType.WALL

    # Add some doors
    num_doors = max(2, int(width * height * 0.02))
    for _ in range(num_doors):
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)
        if grid[y, x] == TileType.FREE_SPACE:
            grid[y, x] = np.random.choice([TileType.DOOR_CLOSED, TileType.DOOR_OPEN])

    # Add entry points for agents
    entry_points_added = 0
    attempts = 0
    while entry_points_added < num_agents and attempts < 1000:
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)
        if grid[y, x] == TileType.FREE_SPACE:
            grid[y, x] = TileType.ENTRY_POINT
            entry_points_added += 1
        attempts += 1

    return grid


def main():
    """
    Main benchmark runner function.
    """
    args = parse_args()

    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / args.log_name
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup CSV output
    csv_path = log_dir / args.csv_name
    write_header = not csv_path.exists() or args.write_header
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow([
            "map", "iteration", "drones", "agent_type",
            "camera_range", "camera_fov",
            "completed", "time", "progress", "steps",
            "total_reward", "avg_reward", "collisions"
        ])

    # Check if using pre-existing maps or generating random ones
    maps_dir = Path(args.maps_dir)
    use_saved_maps = maps_dir.exists()

    if use_saved_maps:
        # Look for existing map files
        map_files = list(maps_dir.glob("house_map_*.txt"))
        if not map_files:
            map_files = list(maps_dir.glob("map_*.txt"))

        if not map_files:
            logging.warning(f"No map files found in {maps_dir}, will generate random maps")
            use_saved_maps = False
        else:
            map_files = sorted(map_files)[:args.map_count]
            logging.info(f"Found {len(map_files)} map files")

    # Calculate total runs for progress bar
    total_runs = args.map_count * len(args.drone_counts) * args.iterations

    with tqdm(total=total_runs, desc="Running Benchmarks", ncols=100) as pbar:
        for map_idx in range(args.map_count):
            # Get or generate map
            if use_saved_maps and map_idx < len(map_files):
                map_path = str(map_files[map_idx])
                logging.info(f"Using map file: {map_path}")
            else:
                map_path = None  # Will generate random map
                logging.info(f"Generating random map {map_idx}")

            for num_drones in args.drone_counts:
                # Configure sensors for all drones
                sensor_config = {
                    i: CameraSensor(
                        max_range=args.camera_range,
                        fov_deg=args.camera_fov
                    )
                    for i in range(num_drones)
                }

                for iteration in range(1, args.iterations + 1):
                    try:
                        # Create environment
                        env = MultiAgentSLAMEnv(
                            width=args.map_width,
                            height=args.map_height,
                            num_agents=num_drones,
                            max_steps=args.max_steps,
                            map_path=map_path,
                            randomize=(map_path is None),  # Randomize if no map file
                            render_mode='human' if args.render else None,
                            sensor_config=sensor_config,
                            discovery_reward=0.1,
                            collision_penalty=-1.0,
                            step_penalty=-0.001,
                            completion_bonus=10.0
                        )

                        # Create agent
                        if args.agent_type == 'frontier':
                            agent = FrontierAgent(
                                num_agents=num_drones,
                                camera_range=args.camera_range
                            )
                        else:
                            agent = RandomAgent(
                                num_agents=num_drones,
                                forward_bias=0.7
                            )

                        # Run episode
                        results = run_single_episode(
                            env, agent,
                            max_time=args.max_time,
                            render=args.render,
                            render_delay=args.render_delay
                        )

                        # Log results
                        if results['completed']:
                            log_msg = (f"Map: {map_idx} | Iteration: {iteration} | "
                                      f"Drones: {num_drones} | Agent: {args.agent_type} | "
                                      f"Time: {results['time']:.2f}s | "
                                      f"Progress: {results['progress']:.1%}")
                        else:
                            log_msg = (f"Map: {map_idx} | Iteration: {iteration} | "
                                      f"Drones: {num_drones} | Agent: {args.agent_type} | "
                                      f"FAILED | Progress: {results['progress']:.1%}")

                        logging.info(log_msg)

                        # Write to CSV
                        csv_writer.writerow([
                            map_idx, iteration, num_drones, args.agent_type,
                            args.camera_range, args.camera_fov,
                            results['completed'],
                            results['time'] if results['time'] else None,
                            round(results['progress'], 3),
                            results['steps'],
                            round(results['total_reward'], 2),
                            round(results['avg_reward'], 2),
                            results['collisions']
                        ])
                        csv_file.flush()  # Ensure data is written

                        # Clean up
                        env.close()

                    except Exception as e:
                        error_msg = (f"Error - Map: {map_idx} | Iteration: {iteration} | "
                                    f"Drones: {num_drones} | Agent: {args.agent_type} | "
                                    f"Error: {str(e)}")
                        logging.error(error_msg)

                        # Write error to CSV
                        csv_writer.writerow([
                            map_idx, iteration, num_drones, args.agent_type,
                            args.camera_range, args.camera_fov,
                            False, None, 0, 0, 0, 0, 0
                        ])
                        csv_file.flush()

                    # Free memory
                    gc.collect()
                    pbar.update(1)

    csv_file.close()
    print(f"\nBenchmark complete! Results saved to:")
    print(f"  Log file: {log_file}")
    print(f"  CSV file: {csv_path}")


if __name__ == "__main__":
    main()