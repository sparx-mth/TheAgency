"""
This module defines the `Drone` class, which models an autonomous agent in a
2D grid environment performing SLAM (Simultaneous Localization and Mapping).

Each drone maintains:
- A position and movement history
- A local map of the environment (initialized as unknown)
- A field-of-view (FOV) radius for sensing
- A communication interface to interact with a central controller (e.g., master)

Core Methods:
-------------
- `activate(current_time)`: Activates the drone if its entry time has arrived.
- `move(direction, env)`: Moves the drone in a specified direction and updates its state.
- `sense(env)`: Uses Bresenham's algorithm to perform FOV-limited sensing and updates the local map.
- `initialize_map(map_shape)`: Initializes the local map with all unknown tiles.
- `get_observed_map()`: Returns the drone's local map.
- `get_position()`: Returns the current position.
- `get_history()`: Returns the movement path history.

Usage:
------
This class is used as part of a multi-agent SLAM simulation. The drone interacts
with the environment and reports its findings to a communication bus, which is used
by a centralized planner or controller.
"""

import numpy as np
import random
from src.planner.simulation.simulation_constants import *


class Drone:
    def __init__(self, drone_id, start_pos, comm_interface, fov_radius=5, entry_time=0):
        self.id = drone_id
        self.pos = start_pos  # (x, y)
        self.fov_radius = fov_radius
        self.entry_time = entry_time
        self.active = False
        self.local_map = None   # Will be initialized once we get map dimensions
        self.path_history = [start_pos]
        self.collided = False
        self.comm = comm_interface


    def initialize_map(self, map_shape):
        self.local_map = np.full(map_shape, -1, dtype=np.int8)  # -1 = unknown

    def activate(self, current_time):
        if not self.active and current_time >= self.entry_time:
            self.active = True
            self.comm.broadcast_state(self.id, {
                'pos': self.pos,
                'entry_time': self.entry_time,
                'active': self.active,
                'new_discoveries': []
            })

    def move(self, direction, env):
        if not self.active:
            # print("drone", self.id, "is not active")
            return []

        dx, dy = DIRECTIONS[direction]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy

        if env.is_collision(new_x, new_y):
            # print("drone", self.id, "collided")
            self.collided = True
            return []  # No move or discoveries due to collision

        self.pos = (new_x, new_y)
        self.path_history.append(self.pos)
        self.collided = False

        # Sense environment and broadcast new state
        new_discoveries = self.sense(env)


        self.comm.broadcast_state(self.id, {
            'pos': self.pos,
            'entry_time': self.entry_time,
            'active': self.active,
            'new_discoveries': new_discoveries
        })
        # print(f"[Drone {self.id}] Moved to {self.pos}, discovered {len(new_discoveries)} cells")

        return new_discoveries

    def sense(self, env):
        if not self.active:
            return []

        def bresenham(x0, y0, x1, y1):
            """Yield integer coordinates on the line from (x0, y0) to (x1, y1)."""
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        cx, cy = self.pos
        new_discoveries = []

        for offset_y in range(-self.fov_radius, self.fov_radius + 1):
            for offset_x in range(-self.fov_radius, self.fov_radius + 1):
                x = cx + offset_x
                y = cy + offset_y
                if not (0 <= x < env.width and 0 <= y < env.height):
                    continue
                if offset_x ** 2 + offset_y ** 2 > self.fov_radius ** 2:
                    continue

                for lx, ly in bresenham(cx, cy, x, y):
                    if not (0 <= lx < env.width and 0 <= ly < env.height):
                        break
                    val = env.get_tile(lx, ly)
                    if self.local_map[ly, lx] != val:
                        self.local_map[ly, lx] = val
                        new_discoveries.append((lx, ly, val))
                    if val in {WALL, DOOR_CLOSED}:
                        break

        return new_discoveries

    def get_observed_map(self):
        return self.local_map

    def get_position(self):
        return self.pos

    def get_history(self):
        return self.path_history
