import time
import os
import logging
import gc
import csv
from tqdm import tqdm
from planner.simulation.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv
from planner.simulation.simulation_constants import CAMERA_RANGE, MAX_TIME
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='SLAM Benchmark Runner')
    parser.add_argument('--map_count', type=int, default=10, help='Number of maps to run')
    parser.add_argument('--drone_counts', nargs='+', type=int, default=[1, 2, 3], help='Number of drones to run')
    parser.add_argument('--iterations', type=int, default=30, help='Number of iterations to run')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--maps_dir', type=str, default='../../../resources/planner/maps', help='Directory to save logs')
    parser.add_argument('--csv_name', type=str, default='agency_planner_slam_results11.csv', help='Name of CSV results file')
    parser.add_argument('--log_name', type=str, default='agency_planner_slam_run11.log', help='Name of log file')
    parser.add_argument('--write_header', action='store_true', help='Whether to write header to CSV')
    parser.add_argument('--render', type=bool, default=True, help='Whether to render the simulation')
    parser.add_argument('--use_controller', type=bool, default=True, help='Whether to use MasterController')
    parser.add_argument('--controller_mode', type=str, default='frontier', help='Controller mode: random or frontier')

    args = parser.parse_args()
    return args

def run_gym_simulation(env, render=True):
    """
    Run a single simulation episode using the Gym environment.

    Returns:
        float: Completion time in seconds, or None if failed/timeout
    """
    obs, info = env.reset()
    done = False
    truncated = False
    start_time = time.time()

    while not done and not truncated:
        # Since we're using the controller, we can pass empty actions
        # The controller will handle the movement decisions
        actions = {}

        obs, rewards, dones, truncateds, info = env.step(actions)

        # Check if any agent is done (they should all finish together)
        done = any(dones.values())
        truncated = any(truncateds.values())

        if render:
            env.render()

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            return None

    # Check if completed successfully
    if info['exploration_progress'] >= 1.0:
        completion_time = time.time() - start_time
        return completion_time
    else:
        return None


def main():
    """
    This script runs a series of SLAM simulations across different map instances
    and drone counts. The results (completion time or failure) are logged to a file
    and also saved to a CSV file for further analysis.

    Each simulation is executed with the following parameters:
    - Varying number of drones: [1, 2, 3]
    - Multiple maps: house_map_0.txt to house_map_9.txt
    - Each map and drone setup is run for 30 iterations

    Logs are written to: ../logs/slam_run1.log
    CSV is written to:  ../logs/slam_results1.csv

    Expected log format:
    Map: <index> | Iteration: <i> | Drones: <count> | Time: <seconds or 'not solved'>

    Requirements:
    - The `src/` folder should be in the Python path
    - Maps must be located at: ../resources/maps/house_map_{i}.txt
    - Dependencies: pygame, tqdm, logging, csv
    """


    args = parse_args()

    # === Configuration ===
    max_iterations = args.iterations
    map_count = args.map_count
    drone_counts = args.drone_counts

    # === Logging setup ===
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_dir = Path(log_dir)

    log_file = log_dir / args.log_name
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # === CSV setup ===
    csv_path = log_dir/ args.csv_name
    write_header = not csv_path.exists() or args.write_header
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["map", "iteration", "drones", "time"])

    # === Simulation ===
    total_runs = map_count * len(drone_counts) * max_iterations
    maps_dir = Path(args.maps_dir)
    assert maps_dir.exists(), "Maps directory does not exist"

    with tqdm(total=total_runs, desc="Running Simulations", ncols=100) as pbar:
        for map_idx in range(map_count):
            map_path = maps_dir / f"house_map_{map_idx}.txt"
            assert map_path.exists(), f"Map {map_idx} does not exist"

            for num_drones in drone_counts:
                for iteration in range(1, max_iterations + 1):
                    try:
                        # Create Gym environment
                        env = MultiAgentSLAMGymEnv(
                            width=32,
                            height=32,
                            num_drones=num_drones,
                            num_entry_points=1,
                            camera_range=CAMERA_RANGE,
                            fov=45,  # Using default FOV from the Gym env
                            sensor_type='camera',
                            max_steps=int(MAX_TIME * 10),  # Convert time to steps
                            map_path=map_path.as_posix(),
                            randomize=False,  # We're using pre-defined maps
                            render_mode='human' if args.render else None,
                            use_controller=args.use_controller,
                            controller_mode=args.controller_mode
                        )

                        # Run simulation
                        result = run_gym_simulation(env, render=args.render)

                        if result is None:
                            logging.info(
                                f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved")
                            csv_writer.writerow([map_idx, iteration, num_drones, None])
                        else:
                            logging.info(
                                f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: {result:.2f} seconds")
                            csv_writer.writerow([map_idx, iteration, num_drones, round(result, 2)])

                        # Clean up
                        env.close()

                    except Exception as e:
                        logging.error(
                            f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved | Error: {e}")
                        csv_writer.writerow([map_idx, iteration, num_drones, None])

                    # Free memory
                    gc.collect()
                    pbar.update(1)

    csv_file.close()


if __name__ == "__main__":
    main()