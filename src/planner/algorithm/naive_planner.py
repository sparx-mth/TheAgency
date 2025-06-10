"""
This module implements drone motion planning strategies for the SLAM simulation.

It provides two main planners:
- `plan_random_walk`: A naive planner that randomly samples directions until it finds a valid move.
- `plan_frontier`: A more advanced frontier-based planner that assigns each drone to explore the closest unexplored
    frontier, maximizing distance from other drones to avoid overlap. It uses A* pathfinding and handles goal
    reassignment and waiting logic.

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


def plan_random_walk(pos, facing, env):
    print("plan_random_walk")
    # 25% chance to rotate instead of moving forward
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


def a_star(start, goal, grid):
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


def plan_frontier(drone_id, current_pos, facing, goal, path, frontiers, assigned_goals, all_states,
                  global_map, wait_counter, max_wait, env):

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


def _follow_path(drone_id, current_pos, facing, goal, path, all_states, wait_counter, max_wait, env):
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
