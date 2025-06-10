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
from src.planner.simulation.sensors.sensor_manager import SensorManager



def turn(facing, action):
    idx = FACING_DIRECTIONS.index(facing)
    if action == 'TURN_LEFT':
        return FACING_DIRECTIONS[(idx - 1) % 4]
    elif action == 'TURN_RIGHT':
        return FACING_DIRECTIONS[(idx + 1) % 4]
    return facing


class Drone:
    def __init__(self, drone_id, start_pos, comm_interface, fov_radius=5, entry_time=0, facing_direction='NORTH', sensors=None):
        self.id = drone_id
        self.pos = start_pos  # (x, y)
        self.fov_radius = fov_radius
        self.entry_time = entry_time
        self.active = False
        self.local_map = None   # Will be initialized once we get map dimensions
        self.path_history = [start_pos]
        self.collided = False
        self.comm = comm_interface
        self.facing_direction = facing_direction
        self.sensor_manager = SensorManager()
        if sensors:
            for sensor in sensors:
                self.sensor_manager.add_sensor(sensor)

    def initialize_map(self, map_shape):
        self.local_map = np.full(map_shape, -1, dtype=np.int8)  # -1 = unknown

    def activate(self, current_time):
        if not self.active and current_time >= self.entry_time:
            self.active = True
            self.comm.broadcast_state(self.id, self._make_state([]))

    def move(self, action, env):
        if not self.active:
            return []

        if action in ['TURN_LEFT', 'TURN_RIGHT']:
            self.facing_direction = turn(self.facing_direction, action)
            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        elif action == 'FORWARD':
            dx, dy = FACING_TO_DELTA[self.facing_direction]
            new_x, new_y = self.pos[0] + dx, self.pos[1] + dy

            if env.is_collision(new_x, new_y):
                self.collided = True
                return []

            self.pos = (new_x, new_y)
            self.path_history.append(self.pos)
            self.collided = False

            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        elif action == 'STAY':
            new_discoveries = self.sense(env)
            self.comm.broadcast_state(self.id, self._make_state(new_discoveries))
            return new_discoveries

        else:
            raise ValueError(f"Invalid action for constrained drone: {action}")

    def sense(self, env):
        if not self.active:
            return []

        observations = self.sensor_manager.sense_all(self.pos, self.facing_direction, env)
        new_discoveries = []

        for x, y, val in observations:
            if self.local_map[y, x] != val:
                self.local_map[y, x] = val
                new_discoveries.append((x, y, val))

        return new_discoveries

    def _make_state(self, new_discoveries):
        return {
            'pos': self.pos,
            'facing_direction': self.facing_direction,
            'entry_time': self.entry_time,
            'active': self.active,
            'new_discoveries': new_discoveries
        }

    def get_observed_map(self):
        return self.local_map

    def get_position(self):
        return self.pos

    def get_history(self):
        return self.path_history

    def get_facing_arrow_vector(self):
        """Returns the direction vector the drone is facing, for rendering."""
        dx, dy = FACING_TO_DELTA[self.facing_direction]
        return dx, dy
