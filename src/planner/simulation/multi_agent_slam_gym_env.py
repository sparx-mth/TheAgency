"""
Multi-Agent SLAM Gym Environment - Refactored for Clean Separation

This module provides a Gym-compatible multi-agent environment for SLAM simulation
with complete separation between environment and agent logic.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import pygame
import time

# Import original modules - try different import methods for compatibility
try:
    # Absolute imports (when installed as package)
    from planner.simulation.grid_map_env import GridMapEnv
    from planner.simulation.drone import Drone
    from planner.simulation.simulation_constants import (
        DIRECTIONS, DIRECTION_COMMANDS, FACING_DIRECTIONS, FACING_TO_DELTA,
        WALL, DOOR_CLOSED, OUT_OF_BOUNDS, FREE_SPACE, ENTRY_POINT,
        TILE_SIZE, FPS, MAX_TIME, TILE_NAME
    )
    from planner.simulation.sensors.camera_sensor import CameraSensor
    from planner.simulation.sensors.bresenham_fov import BresenhamFOVSensor
    from planner.communication.local_bus import LocalCommBus
except ImportError:
    # Relative imports (when running from within the package)
    from .grid_map_env import GridMapEnv
    from .drone import Drone
    from .simulation_constants import (
        DIRECTIONS, DIRECTION_COMMANDS, FACING_DIRECTIONS, FACING_TO_DELTA,
        WALL, DOOR_CLOSED, OUT_OF_BOUNDS, FREE_SPACE, ENTRY_POINT,
        TILE_SIZE, FPS, MAX_TIME, TILE_NAME
    )
    from .sensors.camera_sensor import CameraSensor
    from .sensors.bresenham_fov import BresenhamFOVSensor
    from ..communication.local_bus import LocalCommBus


class MultiAgentSLAMGymEnv(gym.Env):
    """
    Multi-Agent SLAM Environment following Gym conventions.

    This environment provides a pure simulation without any agent logic.
    Agents should be implemented separately and interact with the environment
    through the standard Gym interface.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        num_drones: int = 3,
        num_entry_points: int = 1,
        camera_range: int = 10,
        fov: int = 45,
        sensor_type: str = 'camera',
        max_steps: int = 1000,
        map_path: Optional[str] = None,
        randomize: bool = True,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the multi-agent SLAM environment.

        Args:
            width: Width of the grid map
            height: Height of the grid map
            num_drones: Number of drone agents
            num_entry_points: Number of entry points on the map
            camera_range: Maximum sensing range
            fov: Field of view for camera sensor (degrees)
            sensor_type: Type of sensor ('camera' or 'bresenham')
            max_steps: Maximum steps per episode
            map_path: Path to load a pre-defined map
            randomize: Whether to generate random maps
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()

        # Environment parameters
        self.width = width
        self.height = height
        self.num_drones = num_drones
        self.num_entry_points = num_entry_points
        self.camera_range = camera_range
        self.fov = fov
        self.sensor_type = sensor_type
        self.max_steps = max_steps
        self.map_path = map_path
        self.randomize = randomize
        self.render_mode = render_mode

        # Communication bus
        self.comm = LocalCommBus()

        # Initialize environment components
        self._init_environment()

        # Define action and observation spaces
        self._init_spaces()

        # Rendering components
        self.screen = None
        self.clock = None
        self.font = None

        # Episode tracking
        self.current_step = 0
        self.start_time = None

    def _init_environment(self):
        """Initialize the grid environment and drones."""
        # Create grid environment with drones
        self.env = GridMapEnv(
            comm_interface=self.comm,
            width=self.width,
            height=self.height,
            randomize=self.randomize,
            map_path=self.map_path,
            num_entry_points=self.num_entry_points,
            num_drones=self.num_drones,
            camera_range=self.camera_range,
            fov=self.fov
        )

        # Override sensor type if needed
        if self.sensor_type == 'bresenham':
            for drone in self.env.drones:
                drone.sensor_manager.sensors.clear()
                drone.sensor_manager.add_sensor(BresenhamFOVSensor(self.camera_range))

        # Compute reachable mask for progress tracking
        self.reachable_mask = self._compute_reachable_mask()
        self.total_reachable = np.sum(self.reachable_mask)

    def _init_spaces(self):
        """Define action and observation spaces for all agents."""
        # Agent IDs
        self.agents = list(range(self.num_drones))
        self.possible_agents = self.agents.copy()

        # Action space: Discrete actions matching DIRECTION_COMMANDS
        self.action_spaces = {
            agent_id: spaces.Discrete(len(DIRECTION_COMMANDS))
            for agent_id in self.agents
        }

        # Observation space
        self.observation_spaces = {}
        for agent_id in self.agents:
            self.observation_spaces[agent_id] = spaces.Dict({
                # Drone's local map
                'local_map': spaces.Box(
                    low=-1, high=6,
                    shape=(self.height, self.width),
                    dtype=np.int8
                ),
                # Current position
                'position': spaces.Box(
                    low=0, high=max(self.width, self.height),
                    shape=(2,), dtype=np.int32
                ),
                # Facing direction (0-3)
                'facing_direction': spaces.Discrete(4),
                # Whether drone is active
                'active': spaces.Discrete(2),
                # Last collision status
                'collided': spaces.Discrete(2),
                # Entry time
                'entry_time': spaces.Box(
                    low=0, high=self.max_steps,
                    shape=(1,), dtype=np.int32
                ),
                # New discoveries from last action
                'new_discoveries': spaces.Box(
                    low=-1, high=max(self.width * self.height, 100),
                    shape=(1,), dtype=np.int32
                )
            })

    def _compute_reachable_mask(self) -> np.ndarray:
        """Compute reachable/discoverable tiles."""
        height, width = self.env.grid.shape
        walkable_reachable = np.zeros((height, width), dtype=bool)
        visited = np.zeros((height, width), dtype=bool)
        queue = list(self.env.entry_points)

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
                    if not visited[ny, nx] and self.env.grid[ny, nx] not in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                        queue.append((ny, nx))

        # Phase 2: Build final reachable mask
        final_reachable = walkable_reachable.copy()
        for y in range(height):
            for x in range(width):
                if self.env.grid[y, x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if walkable_reachable[ny, nx]:
                                final_reachable[y, x] = True
                                break

        return final_reachable

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[int, Any], Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observations: Dict mapping agent_id to observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reinitialize environment
        self._init_environment()

        # Reset episode tracking
        self.current_step = 0
        self.start_time = time.time()

        # Clear communication bus
        self.comm.clear()

        # Store new discoveries for each drone
        self.last_discoveries = {agent_id: [] for agent_id in self.agents}

        # Store reward components for debugging
        self.last_reward_components = {agent_id: {} for agent_id in self.agents}

        # Get initial observations
        observations = {}
        for agent_id, drone in enumerate(self.env.drones):
            observations[agent_id] = self._get_observation(drone, agent_id)

        info = self._get_info()

        return observations, info

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Any], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: Dict mapping agent_id to action index (required for all active agents)

        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            dones: Whether each agent is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1

        observations = {}
        rewards = {}
        dones = {}
        truncated = {}

        # Process each drone
        for agent_id, drone in enumerate(self.env.drones):
            # 1. Activate drone if it's time
            drone.activate(self.current_step)

            # 2. Move if active
            if drone.active:
                if agent_id not in actions:
                    raise ValueError(f"No action provided for active agent {agent_id}")

                # Use provided action
                action = DIRECTION_COMMANDS[actions[agent_id]]

                # Execute movement
                new_discoveries = drone.move(action, self.env)
                self.last_discoveries[agent_id] = new_discoveries

                # Calculate reward (exploration-based)
                discovery_reward = len(new_discoveries) * 0.1  # Small reward per discovery
                time_penalty = -0.001  # Small time penalty
                collision_penalty = -0.5 if drone.collided else 0.0  # Collision penalty

                reward = discovery_reward + time_penalty + collision_penalty

                # Store reward components
                self.last_reward_components[agent_id] = {
                    'discovery_reward': discovery_reward,
                    'time_penalty': time_penalty,
                    'collision_penalty': collision_penalty,
                    'total': reward
                }

            else:
                reward = 0.0
                self.last_discoveries[agent_id] = []
                self.last_reward_components[agent_id] = {
                    'discovery_reward': 0.0,
                    'time_penalty': 0.0,
                    'collision_penalty': 0.0,
                    'total': 0.0
                }

            # Get observation
            observations[agent_id] = self._get_observation(drone, agent_id)
            rewards[agent_id] = reward

            # Check termination
            done = False
            trunc = False

            # Time limit
            elapsed = time.time() - self.start_time
            if elapsed > MAX_TIME or self.current_step >= self.max_steps:
                done = True
                trunc = True

            # Check if exploration complete
            progress = self._get_exploration_progress()
            if progress >= 1.0:
                done = True
                completion_bonus = 10.0  # Completion bonus
                reward += completion_bonus
                rewards[agent_id] = reward
                self.last_reward_components[agent_id]['completion_bonus'] = completion_bonus
                self.last_reward_components[agent_id]['total'] = reward

            dones[agent_id] = done
            truncated[agent_id] = trunc

        info = self._get_info()

        return observations, rewards, dones, truncated, info

    def _get_observation(self, drone: Drone, agent_id: int) -> Dict[str, Any]:
        """Get observation for a specific drone."""
        facing_idx = FACING_DIRECTIONS.index(drone.facing_direction)

        return {
            'local_map': drone.local_map.copy() if drone.local_map is not None else np.full((self.height, self.width), -1, dtype=np.int8),
            'position': np.array(drone.pos, dtype=np.int32),
            'facing_direction': facing_idx,
            'active': int(drone.active),
            'collided': int(drone.collided),
            'entry_time': np.array([drone.entry_time], dtype=np.int32),
            'new_discoveries': np.array([len(self.last_discoveries.get(agent_id, []))], dtype=np.int32)
        }

    def _get_exploration_progress(self) -> float:
        """Calculate exploration progress across all drones."""
        # Merge all drone observations
        global_map = np.full(self.env.grid.shape, -1, dtype=np.int8)
        for drone in self.env.drones:
            if drone.local_map is not None:
                mask = drone.local_map != -1
                global_map[mask] = drone.local_map[mask]

        # Calculate progress
        known_cells = np.count_nonzero((global_map != -1) & self.reachable_mask)
        return known_cells / self.total_reachable if self.total_reachable > 0 else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        # Get global observed map
        global_map = np.full(self.env.grid.shape, -1, dtype=np.int8)
        for drone in self.env.drones:
            if drone.local_map is not None:
                mask = drone.local_map != -1
                global_map[mask] = drone.local_map[mask]

        # Individual drone discoveries
        drone_discoveries = {}
        for i, drone in enumerate(self.env.drones):
            if drone.local_map is not None:
                drone_discoveries[i] = np.count_nonzero(drone.local_map != -1)
            else:
                drone_discoveries[i] = 0

        # Get all drone states from communication bus
        all_drone_states = self.comm.get_all_drones_state()

        return {
            'step': self.current_step,
            'elapsed_time': time.time() - self.start_time,
            'exploration_progress': self._get_exploration_progress(),
            'global_map': global_map,
            'drone_discoveries': drone_discoveries,
            'all_drone_states': all_drone_states,
            'reachable_mask': self.reachable_mask,
            'entry_points': self.env.entry_points,
            'true_map': self.env.grid.copy(),
            'reward_components': self.last_reward_components.copy()
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            self.font = pygame.font.SysFont("Arial", 16)
            screen_width = TILE_SIZE * self.width * 2 + 50
            screen_height = TILE_SIZE * self.height + 160
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Multi-Agent SLAM Simulation")
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((20, 20, 20))

        # Get global observed map
        info = self._get_info()
        observed_map = info['global_map']

        # Draw true map (left side)
        for y in range(self.env.height):
            for x in range(self.env.width):
                tile = self.env.grid[y, x]
                color = self._get_tile_color(tile)
                pygame.draw.rect(
                    self.screen, color,
                    (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                )

        # Draw observed map (right side)
        for y in range(self.env.height):
            for x in range(self.env.width):
                tile = observed_map[y, x]
                color = self._get_tile_color(tile, unknown_color=(20, 20, 20))
                pygame.draw.rect(
                    self.screen, color,
                    (self.width * TILE_SIZE + 50 + x * TILE_SIZE, y * TILE_SIZE,
                     TILE_SIZE - 1, TILE_SIZE - 1)
                )

        # Draw drones
        for drone in self.env.drones:
            if drone.active:
                dx, dy = drone.get_position()

                # Draw on true map
                pygame.draw.circle(
                    self.screen, (255, 255, 0),
                    (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5
                )

                # Draw arrow showing facing direction
                fx, fy = drone.get_facing_arrow_vector()
                arrow_start = (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2)
                arrow_end = (arrow_start[0] + fx * TILE_SIZE // 2, arrow_start[1] + fy * TILE_SIZE // 2)
                pygame.draw.line(self.screen, (255, 0, 0), arrow_start, arrow_end, 2)

                # Draw ID on observed map
                drone_id_text = self.font.render(str(drone.id), True, (0, 0, 0))
                self.screen.blit(
                    drone_id_text,
                    (self.width * TILE_SIZE + 50 + dx * TILE_SIZE + 5, dy * TILE_SIZE)
                )

        # Progress bar
        progress_ratio = self._get_exploration_progress()
        bar_top = self.screen.get_height() - 100
        bar_height = 24
        bar_width = self.screen.get_width() - 100

        pygame.draw.rect(self.screen, (80, 80, 80), (50, bar_top, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (50, bar_top, int(bar_width * progress_ratio), bar_height))

        progress_text = self.font.render(f"Progress: {int(progress_ratio * 100)}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (50, bar_top - 20))

        # Timer
        elapsed = time.time() - self.start_time
        time_text = self.font.render(f"Time: {elapsed:.2f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (self.screen.get_width() - 140, bar_top - 20))

        # Legend
        self._draw_legend()

        pygame.display.flip()
        self.clock.tick(FPS)

        if self.render_mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _get_tile_color(self, tile: int, unknown_color: Tuple[int, int, int] = (120, 120, 120)) -> Tuple[int, int, int]:
        """Get color for a tile value."""
        color_map = {
            -1: unknown_color,
            0: (60, 60, 60),       # FREE_SPACE - darker in true map
            1: (100, 100, 100),    # WALL
            2: (0, 255, 255),      # ENTRY_POINT
            3: (255, 0, 0),        # DOOR_CLOSED
            4: (0, 200, 0),        # DOOR_OPEN
            5: (0, 0, 255),        # WINDOW
            6: (0, 0, 0)           # OUT_OF_BOUNDS
        }

        # Make free space brighter in observed map
        if tile == 0 and unknown_color == (20, 20, 20):
            return (200, 200, 200)

        return color_map.get(tile, (150, 150, 150))

    def _draw_legend(self):
        """Draw the legend at the bottom of the screen."""
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

        legend_y = self.screen.get_height() - 36
        box_size = 14
        spacing_x = 140
        total_width = len(legend_items) * spacing_x
        start_x = (self.screen.get_width() - total_width) // 2

        for i, (label, color) in enumerate(legend_items):
            x = start_x + i * spacing_x
            pygame.draw.rect(self.screen, color, (x, legend_y, box_size, box_size))
            label_text = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(label_text, (x + box_size + 6, legend_y - 2))

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

    def get_drone_states(self) -> Dict[int, Dict[str, Any]]:
        """Get current state of all drones (for debugging/analysis)."""
        states = {}
        for i, drone in enumerate(self.env.drones):
            states[i] = {
                'position': drone.pos,
                'facing': drone.facing_direction,
                'active': drone.active,
                'collided': drone.collided,
                'path_history': drone.path_history,
                'local_map_coverage': np.count_nonzero(drone.local_map != -1) if drone.local_map is not None else 0
            }
        return states