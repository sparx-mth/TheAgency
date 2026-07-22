import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from sparx_agency.tasks.planning.slam_simulator.constants import (
    TileType, Action, DIRECTIONS, DEFAULT_REWARDS, DEFAULT_SENSOR_CONFIG
)
from sparx_agency.tasks.planning.slam_simulator.drone import Drone
from sparx_agency.tasks.planning.slam_simulator.sensors import CameraSensor, BaseSensor
from sparx_agency.tasks.planning.slam_simulator.map_generator import (
    generate_random_map, load_map, find_entry_points, compute_reachable_mask
)
from sparx_agency.tasks.planning.slam_simulator.renderer import Renderer


class SLAMEnv(gym.Env):
    """Multi-Agent SLAM Environment compatible with Gymnasium."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        num_agents: int = 3,
        max_steps: int = 1000,
        map_path: Optional[str] = None,
        randomize: bool = True,
        render_mode: Optional[str] = None,
        sensor_config: Optional[Dict[int, BaseSensor]] = None,
        rewards: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.map_path = map_path
        self.randomize = randomize
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Load map to get dimensions if path provided
        if map_path:
            temp = load_map(map_path)
            self.height, self.width = temp.shape
        else:
            self.width = width
            self.height = height

        self.rewards = rewards if rewards else DEFAULT_REWARDS.copy()
        self.sensor_config = sensor_config if sensor_config else {}

        # State (initialized in reset)
        self.true_map: Optional[np.ndarray] = None
        self.global_map: Optional[np.ndarray] = None
        self.drones: List[Drone] = []
        self.reachable_mask: Optional[np.ndarray] = None
        self.total_reachable: int = 0
        self.current_step: int = 0
        self.renderer: Optional[Renderer] = None

        # Spaces
        self.action_space = spaces.MultiDiscrete([len(Action)] * num_agents)
        self.observation_space = spaces.Dict({
            'global_map': spaces.Box(-1, 6, (self.height, self.width), dtype=np.int8),
            'positions': spaces.Box(0, max(width, height), (num_agents, 2), dtype=np.int32),
            'facings': spaces.Box(0, 3, (num_agents,), dtype=np.int32),
            'active': spaces.Box(0, 1, (num_agents,), dtype=np.int8),
        })

    def _create_sensor(self, drone_id: int) -> BaseSensor:
        if drone_id in self.sensor_config:
            return self.sensor_config[drone_id]
        return CameraSensor(**DEFAULT_SENSOR_CONFIG)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        # Generate/load map
        if self.map_path:
            self.true_map = load_map(self.map_path)
            self.height, self.width = self.true_map.shape
        elif self.randomize:
            self.true_map = generate_random_map(self.width, self.height, num_entry_points=self.num_agents)
        else:
            self.true_map = self._default_map()

        self.global_map = np.full((self.height, self.width), TileType.UNKNOWN, dtype=np.int8)

        # Find/create entry points
        entry_points = find_entry_points(self.true_map)
        while len(entry_points) < self.num_agents:
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.true_map[y, x] == TileType.FREE_SPACE and len(entry_points) < self.num_agents:
                        self.true_map[y, x] = TileType.ENTRY_POINT
                        entry_points.append((x, y))

        # Initialize drones
        self.drones = []
        for i in range(self.num_agents):
            pos = entry_points[i] if i < len(entry_points) else (self.width // 2, self.height // 2)
            self.drones.append(Drone(drone_id=i, pos=pos, facing=DIRECTIONS[i % 4]))

        # Compute reachable cells
        self.reachable_mask = compute_reachable_mask(self.true_map, entry_points)
        self.total_reachable = int(np.sum(self.reachable_mask))

        self.current_step = 0

        # Initial sensing
        for drone in self.drones:
            sensor = self._create_sensor(drone.drone_id)
            obs = sensor.sense(drone.pos, drone.facing, self.true_map)
            for x, y, val in obs:
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.global_map[y, x] == TileType.UNKNOWN:
                        self.global_map[y, x] = val

        return self._get_obs(), self._get_info()

    def _default_map(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = TileType.WALL
        for i in range(self.num_agents):
            x = min(1 + i * 2, self.width - 2)
            grid[self.height // 2, x] = TileType.ENTRY_POINT
        return grid

    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self.current_step += 1
        total_reward = 0.0
        total_discoveries = 0

        for i, drone in enumerate(self.drones):
            if not drone.active:
                continue

            action = Action(actions[i])
            total_reward += self.rewards['step']
            collision = False

            if action == Action.TURN_LEFT:
                drone.turn('LEFT')
            elif action == Action.TURN_RIGHT:
                drone.turn('RIGHT')
            elif action == Action.FORWARD:
                new_pos = drone.get_forward_pos()
                nx, ny = new_pos

                # Check collisions
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    collision = True
                elif self.true_map[ny, nx] in (TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS):
                    collision = True
                else:
                    for other in self.drones:
                        if other.drone_id != drone.drone_id and other.active and other.pos == new_pos:
                            collision = True
                            break

                if collision:
                    total_reward += self.rewards['collision']
                    drone.add_collision()
                else:
                    drone.move_to(new_pos)

            # Sensing
            sensor = self._create_sensor(drone.drone_id)
            obs = sensor.sense(drone.pos, drone.facing, self.true_map)
            discoveries = 0
            for x, y, val in obs:
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.global_map[y, x] == TileType.UNKNOWN:
                        self.global_map[y, x] = val
                        if not collision:
                            discoveries += 1

            drone.add_discoveries(discoveries)
            total_discoveries += discoveries

        total_reward += total_discoveries * self.rewards['discovery']

        # Check termination
        discovered = np.sum((self.global_map != TileType.UNKNOWN) & self.reachable_mask)
        terminated = discovered >= self.total_reachable
        truncated = self.current_step >= self.max_steps

        if terminated:
            total_reward += self.rewards['completion']

        return self._get_obs(), total_reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> Dict:
        positions = np.array([d.pos for d in self.drones], dtype=np.int32)
        facings = np.array([d.get_facing_idx() for d in self.drones], dtype=np.int32)
        active = np.array([int(d.active) for d in self.drones], dtype=np.int8)
        return {
            'global_map': self.global_map.copy(),
            'positions': positions,
            'facings': facings,
            'active': active,
        }

    def _get_info(self) -> Dict[str, Any]:
        discovered = int(np.sum((self.global_map != TileType.UNKNOWN) & self.reachable_mask))
        return {
            'step': self.current_step,
            'progress': discovered / self.total_reachable if self.total_reachable > 0 else 0.0,
            'discovered_cells': discovered,
            'total_reachable': self.total_reachable,
            'collisions': [d.collision_count for d in self.drones],
        }

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = Renderer(self.width, self.height)
        info = self._get_info()
        return self.renderer.render(
            self.true_map, self.global_map, self.drones,
            info['progress'], self.current_step, self.max_steps, self.render_mode
        )

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None