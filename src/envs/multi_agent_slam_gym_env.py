"""
Multi-Agent SLAM Gym Environment - Modified for Shared Global Map

This module provides a Gym-compatible multi-agent environment for SLAM simulation
with a single shared global map and simplified observations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import pygame
import time

from envs.grid_map_env import GridMapEnv
from simulation.world.drone import Drone
from simulation.world.simulation_constants import (
    DIRECTIONS, DIRECTION_COMMANDS, FACING_DIRECTIONS, FACING_TO_DELTA,
    WALL, DOOR_CLOSED, OUT_OF_BOUNDS, FREE_SPACE, ENTRY_POINT,
    TILE_SIZE, FPS, MAX_TIME, TILE_NAME
)
from simulation.sensors.camera_sensor import CameraSensor
from communication.local_bus import LocalCommBus


class MultiAgentSLAMGymEnv(gym.Env):
    """
    Multi-Agent SLAM Environment with shared global map.

    All agents share a single global map and the observation includes:
    - The global map (what has been discovered so far)
    - Positions of all drones
    - Facing directions of all drones
    - Active status of all drones
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        num_drones: int = 3,
        num_entry_points: int = 3,
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

        # Global map shared by all drones (initially unknown)
        self.global_map = None

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

        # Configure sensors for all drones
        for drone in self.env.drones:
            drone.sensor_manager.sensors.clear()
            drone.sensor_manager.add_sensor(CameraSensor(self.camera_range, self.fov))
            # Remove individual drone maps since we'll use a global one
            drone.local_map = None

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

        # Simplified observation space - same for all agents
        self.observation_spaces = {}
        for agent_id in self.agents:
            self.observation_spaces[agent_id] = spaces.Dict({
                # Shared global map
                'global_map': spaces.Box(
                    low=-1, high=6,
                    shape=(self.height, self.width),
                    dtype=np.int8
                ),
                # All drone positions (num_drones, 2)
                'drone_positions': spaces.Box(
                    low=0, high=max(self.width, self.height),
                    shape=(self.num_drones, 2), dtype=np.int32
                ),
                # All drone facing directions (0-3 for NORTH, EAST, SOUTH, WEST)
                'drone_directions': spaces.Box(
                    low=0, high=3,
                    shape=(self.num_drones,), dtype=np.int32
                ),
                # Whether each drone is active
                'drone_active': spaces.Box(
                    low=0, high=1,
                    shape=(self.num_drones,), dtype=np.int32
                ),
                # Current agent's ID for reference
                'agent_id': spaces.Box(
                    low=0, high=self.num_drones-1,
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

        # Phase 2: Build final reachable mask (including walls adjacent to walkable areas)
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

        # Initialize global map as unknown
        self.global_map = np.full((self.height, self.width), -1, dtype=np.int8)

        # Store collision flags for each drone
        self.collision_flags = {agent_id: False for agent_id in self.agents}

        # Get initial observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_observation(agent_id)

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

                # Get action
                action = DIRECTION_COMMANDS[actions[agent_id]]

                # Check for collision BEFORE moving
                collision_occurred = False
                if action == 'FORWARD':
                    dx, dy = FACING_TO_DELTA[drone.facing_direction]
                    new_x, new_y = drone.pos[0] + dx, drone.pos[1] + dy

                    # Check if the new position would cause a collision
                    if not (0 <= new_x < self.env.width and 0 <= new_y < self.env.height):
                        collision_occurred = True
                    elif self.env.grid[new_y, new_x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                        collision_occurred = True
                    else:
                        # Check collision with other drones
                        for other_id, other_drone in enumerate(self.env.drones):
                            if other_id != agent_id and other_drone.active:
                                if other_drone.pos == (new_x, new_y):
                                    collision_occurred = True
                                    break

                self.collision_flags[agent_id] = collision_occurred

                # Execute movement (even if collision - drone stays in place)
                if collision_occurred and action == 'FORWARD':
                    # Don't actually move, but still sense
                    sensed_tiles = drone.sense(self.env)
                else:
                    # Normal movement
                    sensed_tiles = drone.move(action, self.env)

                # Update global map with sensed tiles (only update unknown cells)
                new_discoveries = []
                for x, y, val in sensed_tiles:
                    if 0 <= y < self.height and 0 <= x < self.width:
                        if self.global_map[y, x] == -1:
                            self.global_map[y, x] = val
                            new_discoveries.append((x, y, val))

                # Calculate reward
                discovery_reward = len(new_discoveries) * 0.1  # Reward for discovery
                collision_penalty = -1.0 if collision_occurred else 0.0  # Penalty for collision
                time_penalty = -0.001  # Small time penalty

                reward = discovery_reward + collision_penalty + time_penalty

            else:
                reward = 0.0
                self.collision_flags[agent_id] = False

            # Get observation
            observations[agent_id] = self._get_observation(agent_id)
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

            dones[agent_id] = done
            truncated[agent_id] = trunc

        info = self._get_info()

        return observations, rewards, dones, truncated, info

    def _get_observation(self, agent_id: int) -> Dict[str, Any]:
        """Get observation for a specific agent."""
        # Collect all drone positions and directions
        drone_positions = np.zeros((self.num_drones, 2), dtype=np.int32)
        drone_directions = np.zeros(self.num_drones, dtype=np.int32)
        drone_active = np.zeros(self.num_drones, dtype=np.int32)

        for i, drone in enumerate(self.env.drones):
            drone_positions[i] = drone.pos
            drone_directions[i] = FACING_DIRECTIONS.index(drone.facing_direction)
            drone_active[i] = int(drone.active)

        return {
            'global_map': self.global_map.copy(),
            'drone_positions': drone_positions,
            'drone_directions': drone_directions,
            'drone_active': drone_active,
            'agent_id': np.array([agent_id], dtype=np.int32)
        }

    def _get_exploration_progress(self) -> float:
        """Calculate exploration progress based on the global map."""
        known_cells = np.count_nonzero((self.global_map != -1) & self.reachable_mask)
        return known_cells / self.total_reachable if self.total_reachable > 0 else 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        # Individual drone discoveries (cells each drone has personally discovered)
        drone_discoveries = {}
        for i in range(self.num_drones):
            # Count would need to be tracked separately if needed
            drone_discoveries[i] = 0  # Placeholder

        return {
            'step': self.current_step,
            'elapsed_time': time.time() - self.start_time,
            'exploration_progress': self._get_exploration_progress(),
            'global_map': self.global_map.copy(),
            'drone_discoveries': drone_discoveries,
            'reachable_mask': self.reachable_mask,
            'entry_points': self.env.entry_points,
            'true_map': self.env.grid.copy(),
            'collision_flags': self.collision_flags.copy()
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
                tile = self.global_map[y, x]
                color = self._get_tile_color(tile, unknown_color=(20, 20, 20))
                pygame.draw.rect(
                    self.screen, color,
                    (self.width * TILE_SIZE + 50 + x * TILE_SIZE, y * TILE_SIZE,
                     TILE_SIZE - 1, TILE_SIZE - 1)
                )

        # Draw drones
        for i, drone in enumerate(self.env.drones):
            if drone.active:
                dx, dy = drone.get_position()

                # Use different colors for each drone
                drone_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
                drone_color = drone_colors[i % len(drone_colors)]

                # Draw on true map
                pygame.draw.circle(
                    self.screen, drone_color,
                    (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5
                )

                # Draw arrow showing facing direction
                fx, fy = drone.get_facing_arrow_vector()
                arrow_start = (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2)
                arrow_end = (arrow_start[0] + fx * TILE_SIZE // 2, arrow_start[1] + fy * TILE_SIZE // 2)
                pygame.draw.line(self.screen, (255, 0, 0), arrow_start, arrow_end, 2)

                # Draw collision indicator if collided
                if self.collision_flags.get(i, False):
                    pygame.draw.circle(
                        self.screen, (255, 0, 0),
                        (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 8, 2
                    )

                # Draw on observed map
                pygame.draw.circle(
                    self.screen, drone_color,
                    (self.width * TILE_SIZE + 50 + dx * TILE_SIZE + TILE_SIZE // 2,
                     dy * TILE_SIZE + TILE_SIZE // 2), 5
                )

                # Draw ID
                drone_id_text = self.font.render(str(i), True, (255, 255, 255))
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
            ("Door (C)", (255, 0, 0)),
            ("Door (O)", (0, 200, 0)),
            ("Window", (0, 0, 255)),
            ("Out", (0, 0, 0)),
            ("Drone", (255, 255, 0)),
            ("Facing", (255, 0, 0)),
            ("Collision", (255, 0, 0))
        ]

        legend_y = self.screen.get_height() - 36
        box_size = 14
        spacing_x = 110
        total_width = len(legend_items) * spacing_x
        start_x = (self.screen.get_width() - total_width) // 2

        for i, (label, color) in enumerate(legend_items):
            x = start_x + i * spacing_x
            if label == "Collision":
                # Draw circle outline for collision indicator
                pygame.draw.circle(self.screen, color, (x + box_size // 2, legend_y + box_size // 2), box_size // 2, 2)
            else:
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
                'collided': self.collision_flags.get(i, False),
                'path_history': drone.path_history
            }
        return states