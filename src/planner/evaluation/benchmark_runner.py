"""
This script runs a series of SLAM simulations across different map instances
and drone counts. The results (completion time or failure) are logged to a file
and also saved to a CSV file for further analysis.

Each simulation is executed with the following parameters:
- Varying number of drones: [1, 2, 3]
- Multiple maps: house_map_0.txt to house_map_9.txt
- Each map and drone setup is run for 30 iterations

Logs are written to: ../logs/slam_run1.log
CSV is written to:  ../logs/slam_results.csv

Expected log format:
Map: <index> | Iteration: <i> | Drones: <count> | Time: <seconds or 'not solved'>

Usage:
> python run_all_simulations.py

Requirements:
- The `src/` folder should be in the Python path
- Maps must be located at: ../resources/maps/house_map_{i}.txt
- Dependencies: pygame, tqdm, logging, csv
"""

import time
import os
import logging
import gc
import csv
from tqdm import tqdm
from src.planner.simulation.sim_runner import run_simulation
from src.planner.communication.local_bus import LocalCommBus

# === Configuration ===
MAX_ITERATIONS = 30
MAX_TIME = 50  # seconds
MAP_COUNT = 10
DRONE_COUNTS = [1, 2, 3]

# === Logging setup ===
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "slam_run1.log")
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === CSV setup ===
csv_path = os.path.join(log_dir, "slam_results1.csv")
write_header = not os.path.exists(csv_path)
csv_file = open(csv_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["map", "iteration", "drones", "time"])

# === Simulation ===
total_runs = MAP_COUNT * len(DRONE_COUNTS) * MAX_ITERATIONS
comm = LocalCommBus()

with tqdm(total=total_runs, desc="Running Simulations", ncols=100) as pbar:
    for map_idx in range(MAP_COUNT):
        map_path = f"../resources/maps/house_map_{map_idx}.txt"
        for num_drones in DRONE_COUNTS:
            for iteration in range(1, MAX_ITERATIONS + 1):
                try:
                    start = time.time()
                    result = run_simulation(
                        comm=comm,
                        map_path=map_path,
                        width=32,
                        height=32,
                        num_drones=num_drones,
                        num_entry_points=1,
                        fov=1,
                        render=True
                    )
                    elapsed = time.time() - start

                    if result is None or elapsed > MAX_TIME:
                        logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved")
                        csv_writer.writerow([map_idx, iteration, num_drones, None])
                    else:
                        logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: {result:.2f} seconds")
                        csv_writer.writerow([map_idx, iteration, num_drones, round(result, 2)])

                except Exception as e:
                    logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved | Error: {e}")
                    csv_writer.writerow([map_idx, iteration, num_drones, None])

                # Free memory
                import pygame
                pygame.quit()
                gc.collect()
                pbar.update(1)

csv_file.close()
