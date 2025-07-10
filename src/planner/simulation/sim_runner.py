"""
This module defines the main simulation loop for a multi-agent SLAM task in a grid environment. It manages environment
initialization, drone activation, motion, sensing, rendering (optional), and success criteria evaluation.

Functionality Overview:
-----------------------
- Initializes the grid map environment with specified dimensions, number of drones, and field of view (FOV).
- Optionally loads a predefined map or generates a random map with walls, doors, windows, and out-of-bounds areas.
- Launches and manages drone behavior through a MasterController using either random or frontier-based planning.
- Tracks exploration progress using a reachable mask and terminates upon completion or timeout.
- Provides both graphical rendering (with Pygame) and headless (non-GUI) execution for fast benchmarking.

Key Functions:
--------------
- run_simulation(comm, map_path, width, height, num_drones, num_entry_points, fov, render):
    The core function that runs the entire SLAM simulation loop. Handles drone control, progress tracking,
    visualization, and return of the final completion time.

- compute_reachable_mask(env):
    Computes a boolean mask of all tiles that are discoverable from the entry points. Used to measure simulation
    progress and determine completion.

Visualization (if enabled):
---------------------------
- Left side of the screen: True environment map
- Right side of the screen: Aggregated observed map from all drones
- Drones visualized with IDs and positions
- Progress bar, elapsed time display, and completion message
- Legend at bottom describing tile types and colors

Usage:
------
This file is intended to be imported and called via the `run_simulation()` function from benchmarking
scripts or interactive experiments.

Example:
--------
```python
from src.planner.simulation.sim_runner import run_simulation
run_simulation(comm, width=32, height=32, num_drones=3, render=True)
"""

import pygame
import time
import numpy as np

from planner.simulation.grid_map_env import GridMapEnv
from planner.simulation.master_controller import MasterController
from planner.simulation.simulation_constants import *
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from planner.communication.comm_interface import CommunicationInterface
    from planner.simulation.grid_map_env import GridMapEnv


def compute_reachable_mask(env: "GridMapEnv") -> np.ndarray:
    """
    Computes a boolean mask indicating which tiles in the map are discoverable.

    The mask includes:
    - All walkable tiles reachable from the drone entry points (via BFS)
    - Adjacent non-walkable tiles such as walls, doors, and out-of-bounds that
      border reachable walkable space (for rendering or completion checks)

    Args:
        env (GridMapEnv): The simulation environment containing the map and entry points.

    Returns:
        np.ndarray: A 2D boolean array (height x width) where True indicates a discoverable tile.
    """
    height, width = env.grid.shape
    walkable_reachable = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue = list(env.entry_points)

    # Phase 1: BFS over walkable tiles only
    while queue:
        y, x = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        walkable_reachable[y, x] = True

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if not visited[ny, nx] and env.grid[ny, nx] not in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                    queue.append((ny, nx))

    # Phase 2: Build final reachable mask:
    # - All walkable_reachable cells
    # - Plus walls/doors that are adjacent to walkable_reachable cells
    final_reachable = walkable_reachable.copy()
    for y in range(height):
        for x in range(width):
            if env.grid[y, x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if walkable_reachable[ny, nx]:
                            final_reachable[y, x] = True
                            break

    return final_reachable


def run_simulation(
    comm: "CommunicationInterface",
    map_path: Optional[str] = None,
    width: int = 32,
    height: int = 32,
    num_drones: int = 3,
    num_entry_points: int = 1,
    fov: int = 1,
    render: bool = True
) -> Optional[float]:
    if map_path is None:
        env = GridMapEnv(comm_interface=comm, width=width, height=height, randomize=True,
                         num_entry_points=num_entry_points, num_drones=num_drones, fov=fov)
    else:
        env = GridMapEnv(comm_interface=comm, map_path=map_path, width=width, height=height, randomize=False,
                         num_entry_points=num_entry_points, num_drones=num_drones, fov=fov)

    MAP_WIDTH = env.grid.shape[1]
    MAP_HEIGHT = env.grid.shape[0]

    if render:
        pygame.init()
        font = pygame.font.SysFont("Arial", 16)
        screen_width = TILE_SIZE * MAP_WIDTH * 2 + 50
        screen_height = TILE_SIZE * MAP_HEIGHT + 160
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Multi-Agent SLAM Simulation")

    clock = pygame.time.Clock()
    reachable_mask = compute_reachable_mask(env)
    master = MasterController(env, reachable_mask, comm)
    start_time = time.time()
    tick = 0
    running = True
    completed = False
    completion_time = None

    while running:
        if render:
            screen.fill((20, 20, 20))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if not completed:
            master.step(tick)

            for drone in env.drones:
                # 1. Activate if it's time
                drone.activate(tick)

                # 2. Move if active
                if drone.active:
                    instruction = comm.receive_instructions(drone.id)
                    if instruction:
                        drone.move(instruction, env)

        # Progress check
        observed_map = master.global_map
        known_cells = np.count_nonzero((observed_map != -1) & reachable_mask)
        total_cells = np.count_nonzero(reachable_mask)
        progress_ratio = min(known_cells / total_cells, 1.0)

        if not completed and progress_ratio >= 1.0:
            completed = True
            completion_time = time.time() - start_time
            if not render:
                break  # End simulation early in headless mode

        if render:
            # True map (left)
            for y in range(env.height):
                for x in range(env.width):
                    tile = env.grid[y, x]
                    color = {
                        WALL: (100, 100, 100),
                        FREE_SPACE: (60, 60, 60),
                        ENTRY_POINT: (0, 255, 255),
                        DOOR_CLOSED: (255, 0, 0),
                        DOOR_OPEN: (0, 200, 0),
                        WINDOW: (0, 0, 255),
                        OUT_OF_BOUNDS: (0, 0, 0)
                    }.get(tile, (120, 120, 120))
                    pygame.draw.rect(screen, color, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1))

            # Observed map (right)
            for y in range(env.height):
                for x in range(env.width):
                    tile = observed_map[y, x]
                    color = {
                        WALL: (100, 100, 100),
                        FREE_SPACE: (200, 200, 200),
                        ENTRY_POINT: (0, 255, 255),
                        DOOR_CLOSED: (255, 0, 0),
                        DOOR_OPEN: (0, 200, 0),
                        WINDOW: (0, 0, 255),
                        OUT_OF_BOUNDS: (0, 0, 0),
                        -1: (20, 20, 20)
                    }.get(tile, (150, 150, 150))
                    pygame.draw.rect(screen, color, (MAP_WIDTH * TILE_SIZE + 50 + x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1))

            # Drones
            for drone in env.drones:
                if drone.active:
                    dx, dy = drone.get_position()
                    pygame.draw.circle(screen, (255, 255, 0),
                                       (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5)
                    # Draw arrow showing drone's facing direction
                    fx, fy = drone.get_facing_arrow_vector()
                    arrow_start = (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2)
                    arrow_end = (arrow_start[0] + fx * TILE_SIZE // 2, arrow_start[1] + fy * TILE_SIZE // 2)
                    pygame.draw.line(screen, (255, 0, 0), arrow_start, arrow_end, 2)

                    drone_id_text = font.render(str(drone.id), True, (0, 0, 0))
                    screen.blit(drone_id_text, (MAP_WIDTH * TILE_SIZE + 50 + dx * TILE_SIZE + 5, dy * TILE_SIZE))

            # Progress bar
            bar_top = screen_height - 100
            bar_height = 24
            pygame.draw.rect(screen, (80, 80, 80), (50, bar_top, screen_width - 100, bar_height))
            pygame.draw.rect(screen, (0, 255, 0), (50, bar_top, int((screen_width - 100) * progress_ratio), bar_height))
            screen.blit(font.render(f"Progress: {int(progress_ratio * 100)}%", True, (255, 255, 255)), (50, bar_top - 20))

            # Timer
            elapsed = time.time() - start_time
            if elapsed > MAX_TIME:
                if render:
                    pygame.quit()
                return None  # Timeout
            screen.blit(font.render(f"Time: {elapsed:.2f}s", True, (255, 255, 255)), (screen_width - 140, bar_top - 20))

            # Completion message
            if completed:
                msg = f"Objective Achieved in {completion_time:.2f} seconds"
                rendered_msg = font.render(msg, True, (0, 255, 255))
                screen.blit(rendered_msg, ((screen_width - rendered_msg.get_width()) // 2, bar_top - 50))
                pygame.display.flip()
                time.sleep(5)
                running = False

            # Legend
            legend_items = [
                ("Free", (200, 200, 200)),
                ("Wall", (100, 100, 100)),
                ("Entry", (0, 255, 255)),
                ("Door (Closed)", (255, 0, 0)),
                ("Door (Open)", (0, 200, 0)),
                ("Window", (0, 0, 255)),
                ("Out of Bounds", (0, 0, 0)),
                ("Drone", (255, 255, 0)),
                ("Facing Dir", (255, 0, 0)),
            ]

            legend_y = screen_height - 36
            box_size = 14
            spacing_x = 140
            total_width = len(legend_items) * spacing_x
            start_x = (screen_width - total_width) // 2

            for i, (label, color) in enumerate(legend_items):
                x = start_x + i * spacing_x
                pygame.draw.rect(screen, color, (x, legend_y, box_size, box_size))
                screen.blit(font.render(label, True, (255, 255, 255)), (x + box_size + 6, legend_y - 2))

            pygame.display.flip()
            tick += 1
            clock.tick(FPS)
        else:
            elapsed = time.time() - start_time
            if elapsed > MAX_TIME:
                return None  # Timeout
            tick += 1  # Headless progression
            clock.tick(FPS)

    if render:
        pygame.quit()

    return completion_time

if __name__ == "__main__":
    from planner.communication.local_bus import LocalCommBus
    run_simulation(comm=LocalCommBus())

