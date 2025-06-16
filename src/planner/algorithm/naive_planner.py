"""
This module implements drone motion planning strategies for the SLAM simulation.

It provides two main planners:
- `plan_random_walk`: A naive planner that randomly samples directions until it finds a valid move.
- `plan_frontier`: A sensor-aware planner that first checks if unexplored cells can be discovered by rotating in place
  using the drone’s directional field-of-view, and only then assigns the drone to explore the closest unexplored frontier
  using A* pathfinding.

Utility functions:
- `a_star`: A standard A* pathfinding algorithm over a 2D grid map.
- `_follow_path`: Internal helper for following a path while handling potential collisions with other drones.

Constants like `DIRECTIONS` and `DIRECTION_LIST` are imported from `simulation_constants`.
"""

import random
import numpy as np
import heapq
from src.planner.simulation.simulation_constants import *
from src.planner.simulation.drone import turn
from typing import Optional, Tuple, List, Dict, Set, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv


def plan_random_walk(
    pos: tuple[int, int],
    facing: FACING_DIRECTION,
    env: "GridMapEnv"
) -> DIRECTIONS:
    """
    Selects a random action (forward or turn) for naive exploration.

    Args:
        pos (tuple[int, int]): Current (x, y) position of the drone.
        facing (str): Current facing direction (e.g., 'NORTH').
        env: The simulation environment with collision info.

    Returns:
        str: A direction string ('FORWARD', 'TURN_LEFT', etc.).
    """
    if random.random() < 0.25:
        return random.choice(['TURN_LEFT', 'TURN_RIGHT'])

    # Attempt to move forward if no rotation is chosen
    dx, dy = FACING_TO_DELTA[facing]
    new_x, new_y = pos[0] + dx, pos[1] + dy

    if 0 <= new_x < env.width and 0 <= new_y < env.height:
        if not env.is_collision(new_x, new_y):
            return 'FORWARD'

    # If forward is blocked, rotate instead
    return random.choice(['TURN_LEFT', 'TURN_RIGHT', 'STAY'])


def a_star(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray
) -> list[tuple[int, int]]:
    """
    A* pathfinding algorithm over a 2D grid.

    Args:
        start (tuple[int, int]): Start cell.
        goal (tuple[int, int]): Target cell.
        grid (np.ndarray): Map grid with occupancy values.

    Returns:
        list[tuple[int, int]]: Path as list of (x, y) positions, or empty list if unreachable.
    """
    height, width = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if grid[neighbor[1], neighbor[0]] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS, -1}:  # -1 = unexplored area
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_set, (f_score, neighbor))

    path = []
    while goal in came_from:
        path.append(goal)
        goal = came_from[goal]
    path.reverse()

    return path


def plan_frontier(
    drone_id: int,
    current_pos: Tuple[int, int],
    facing: FACING_DIRECTION,
    goal: Optional[Tuple[int, int]],
    path: List[Tuple[int, int]],
    frontiers: Set[Tuple[int, int]],
    assigned_goals: Set[Tuple[int, int]],
    all_states: Dict[int, Dict[str, Any]],
    global_map: np.ndarray,
    wait_counter: int,
    max_wait: int,
    env: "GridMapEnv"
) -> Tuple[DIRECTIONS, Optional[Tuple[int, int]], List[Tuple[int, int]], int]:
    """
    Sensor-aware planner for SLAM exploration.

    First checks if the drone's directional camera can reveal any unexplored (-1) cells
    nearby by rotating in place. If so, the drone either turns toward them or stays in
    place to sense. If nothing can be discovered locally, the drone is assigned a goal
    from the frontier set using A* pathfinding while avoiding overlap with other drones.

    Args:
        drone_id (int): Unique identifier for the drone.
        current_pos (Tuple[int, int]): Drone's current (x, y) position.
        facing (str): Current direction the drone is facing (e.g., 'NORTH').
        goal (Optional[Tuple[int, int]]): Current target frontier cell.
        path (List[Tuple[int, int]]): Current planned path to the goal.
        frontiers (Set[Tuple[int, int]]): Set of frontier (unknown-bordering) cells.
        assigned_goals (Set[Tuple[int, int]]): Frontier points already taken by other drones.
        all_states (Dict[int, Dict[str, Any]]): All drones' states including position, direction, etc.
        global_map (np.ndarray): Shared map across all drones.
        wait_counter (int): How many ticks the drone has been waiting.
        max_wait (int): Max allowed wait time before switching strategy.
        env (GridMapEnv): The simulation environment (for bounds and collisions).

    Returns:
        Tuple[str, Optional[Tuple[int, int]], List[Tuple[int, int]], int]:
            - Action to perform (e.g., 'FORWARD', 'TURN_LEFT', etc.)
            - New or current goal
            - Updated path to goal
            - Updated wait counter
    """
    for direction in FACING_DIRECTIONS:
        ddx, ddy = FACING_TO_DELTA[direction]
        for step in range(1, CAMERA_RANGE + 1):
            x = current_pos[0] + ddx * step
            y = current_pos[1] + ddy * step

            if not (0 <= x < global_map.shape[1] and 0 <= y < global_map.shape[0]):
                break

            val = global_map[y, x]
            if val in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                break

            if val == -1:  # unexplored
                if direction != facing:
                    # Turn toward that direction
                    if turn(facing, 'TURN_LEFT') == direction:
                        return 'TURN_LEFT', goal, path, wait_counter
                    elif turn(facing, 'TURN_RIGHT') == direction:
                        return 'TURN_RIGHT', goal, path, wait_counter
                    else:
                        return 'TURN_RIGHT', goal, path, wait_counter
                else:
                    return 'STAY', goal, path, wait_counter

    if not goal or global_map[goal[1], goal[0]] != -1 or not path:
        available_frontiers = [f for f in frontiers if f not in assigned_goals]

        if not available_frontiers:
            return plan_random_walk(current_pos, facing, env), None, [], wait_counter

        min_dist = float('inf')
        closest_frontiers = []
        for f in available_frontiers:
            dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
            if dist < min_dist:
                closest_frontiers = [f]
                min_dist = dist
            elif dist == min_dist:
                closest_frontiers.append(f)

        best_goal, best_path = None, []
        max_spacing = -1
        for f in closest_frontiers:
            temp_path = a_star(current_pos, f, global_map)
            if not temp_path:
                continue
            spacing = sum(np.linalg.norm(np.array(f) - np.array(other["pos"]))
                          for other_id, other in all_states.items()
                          if other_id != drone_id and "pos" in other)
            if spacing > max_spacing:
                best_goal, best_path = f, temp_path
                max_spacing = spacing

        if best_goal:
            assigned_goals.add(best_goal)
            return _follow_path(drone_id, current_pos, facing, best_goal, best_path, all_states, wait_counter, max_wait, env)
        else:
            return plan_random_walk(current_pos, facing, env), None, [], wait_counter

    return _follow_path(drone_id, current_pos, facing, goal, path, all_states, wait_counter, max_wait, env)


def _follow_path(
        drone_id: int,
        current_pos: Tuple[int, int],
        facing: FACING_DIRECTION,
        goal: Optional[Tuple[int, int]],
        path: List[Tuple[int, int]],
        all_states: Dict[int, Dict[str, Any]],
        wait_counter: int,
        max_wait: int,
        env: "GridMapEnv"
) -> Tuple[DIRECTIONS, Optional[Tuple[int, int]], List[Tuple[int, int]], int]:
    """
    Follows the given path toward a goal using directional steps.

    If the next step in the path is blocked by another drone, the drone waits in place
    and increments a wait counter. If the wait exceeds `max_wait`, the drone gives up
    and switches to random walk. If unblocked, the drone either moves forward or rotates
    to align with the desired path direction.

    Args:
        drone_id (int): ID of the drone.
        current_pos (Tuple[int, int]): Current position of the drone.
        facing (str): Current facing direction ('NORTH', 'EAST', etc.).
        goal (Optional[Tuple[int, int]]): Current target goal coordinate.
        path (List[Tuple[int, int]]): Path the drone should follow.
        all_states (Dict[int, Dict[str, Any]]): All drones' states for collision checking.
        wait_counter (int): Number of consecutive steps drone has been blocked.
        max_wait (int): Max wait before giving up and rerouting.
        env (GridMapEnv): Simulation environment.

    Returns:
        Tuple[str, Optional[Tuple[int, int]], List[Tuple[int, int]], int]:
            - Action to take
            - Current or updated goal
            - Remaining path
            - Updated wait counter
    """
    if not path:
        return 'STAY', goal, path, wait_counter

    next_pos = path[0]
    blocked = any(other_id != drone_id and other.get("pos") == next_pos
                  for other_id, other in all_states.items())

    if blocked:
        wait_counter += 1
        if wait_counter >= max_wait:
            direction = plan_random_walk(current_pos, facing, env)
            return direction, None, [], 0
        else:
            return 'STAY', goal, path, wait_counter

    dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
    fdx, fdy = FACING_TO_DELTA[facing]

    if (dx, dy) == (fdx, fdy):
        path.pop(0)
        return 'FORWARD', goal, path, 0

    # Try to rotate toward the desired direction
    left_facing = turn(facing, 'TURN_LEFT')
    right_facing = turn(facing, 'TURN_RIGHT')

    if FACING_TO_DELTA[left_facing] == (dx, dy):
        return 'TURN_LEFT', goal, path, wait_counter
    elif FACING_TO_DELTA[right_facing] == (dx, dy):
        return 'TURN_RIGHT', goal, path, wait_counter
    else:
        # If neither single turn gets us there, turn right (arbitrary choice)
        return 'TURN_RIGHT', goal, path, wait_counter
