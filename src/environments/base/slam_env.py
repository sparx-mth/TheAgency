"""
environments/slam_env.py

This is the main SLAM environment file that implements a Gymnasium-compatible
multi-agent environment for simultaneous localization and mapping (SLAM).

The environment supports:
- Configurable number of agents (1 to N)
- Heterogeneous sensor configurations per drone
- Abstract communication interface for future ROS2 integration
- Collision penalties
- Shared global map discovery
- Full Gymnasium compatibility for RL training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional, Any

from environments.base.constants import (
    TileType, Action, DIRECTIONS, DIRECTION_DELTAS,
    TILE_SIZE, FPS, TILE_COLORS, DRONE_COLORS,
    DEFAULT_REWARD_PARAMS
)
from environments.base.drone_state import DroneState
from sensors.base_sensor import BaseSensor
from sensors.camera_sensor import CameraSensor
from communication.comm_interface import CommunicationInterface
from communication.local_comm import LocalCommunication


class MultiAgentSLAMEnv(gym.Env):
    """
    Multi-Agent SLAM Environment.

    A Gymnasium-compatible environment for multi-agent simultaneous localization
    and mapping. Supports heterogeneous sensor configurations and abstract
    communication interfaces while maintaining clean separation from agent logic.

    The environment manages:
    - Drone states and movements
    - Sensor-based environmental perception
    - Global map discovery and sharing
    - Collision detection with penalties
    - Episode termination conditions
    - Rendering for visualization
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        num_agents: int = 3,
        max_steps: int = 1000,
        map_path: Optional[str] = None,
        randomize: bool = True,
        render_mode: Optional[str] = None,
        # Sensor configuration
        sensor_config: Optional[Dict[int, BaseSensor]] = None,
        default_sensor_params: Optional[Dict[str, Any]] = None,
        # Communication
        communication: Optional[CommunicationInterface] = None,
        # Reward parameters
        discovery_reward: float = DEFAULT_REWARD_PARAMS['discovery_reward'],
        collision_penalty: float = DEFAULT_REWARD_PARAMS['collision_penalty'],
        step_penalty: float = DEFAULT_REWARD_PARAMS['step_penalty'],
        completion_bonus: float = DEFAULT_REWARD_PARAMS['completion_bonus'],
    ):
        """
        Initialize the multi-agent SLAM environment.

        Args:
            width: Width of the grid map (ignored if map_path is provided)
            height: Height of the grid map (ignored if map_path is provided)
            num_agents: Number of drone agents
            max_steps: Maximum steps per episode
            map_path: Path to load a pre-defined map
            randomize: Whether to generate random maps each episode
            render_mode: 'human' for display, 'rgb_array' for recording
            sensor_config: Dictionary mapping agent_id to specific sensor instances
            default_sensor_params: Parameters for creating default sensors
            communication: Communication interface (defaults to LocalCommunication)
            discovery_reward: Reward per newly discovered cell
            collision_penalty: Penalty for colliding with obstacles
            step_penalty: Small penalty per step to encourage efficiency
            completion_bonus: Bonus for discovering all reachable cells
        """
        super().__init__()

        # Map configuration
        self.map_path = map_path
        self.randomize = randomize

        # If map_path is provided, load it to get dimensions
        if self.map_path:
            temp_map = self._load_map(self.map_path)
            self.height, self.width = temp_map.shape
        else:
            # Use provided dimensions for generated maps
            self.width = width
            self.height = height

        self.num_agents = num_agents
        self.max_steps = max_steps

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        # Communication system
        self.comm = communication if communication else LocalCommunication()

        # Sensor configuration
        self.sensor_config = sensor_config if sensor_config else {}
        self.default_sensor_params = default_sensor_params if default_sensor_params else {
            'max_range': 10,
            'fov_deg': 45
        }

        # Reward parameters
        self.discovery_reward = discovery_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus

        # State variables (initialized in reset)
        self.true_map = None
        self.global_map = None
        self.drones: List[DroneState] = []
        self.reachable_mask = None
        self.total_reachable = 0
        self.current_step = 0

        # Define action and observation spaces
        self._define_spaces()

    def _define_spaces(self):
        """Define Gymnasium action and observation spaces."""
        # Action space: 4N actions (4 actions per drone)
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.num_agents)

        # Observation space: unified state representation
        self.observation_space = spaces.Dict({
            'global_map': spaces.Box(
                low=-1, high=6,
                shape=(self.height, self.width),
                dtype=np.int8
            ),
            'positions': spaces.Box(
                low=0, high=max(self.width, self.height),
                shape=(self.num_agents, 2), dtype=np.int32
            ),
            'facings': spaces.Box(
                low=0, high=3,
                shape=(self.num_agents,), dtype=np.int32
            ),
            'active': spaces.Box(
                low=0, high=1,
                shape=(self.num_agents,), dtype=np.int8
            ),
        })

    def _create_sensor(self, drone_id: int) -> BaseSensor:
        """
        Create or retrieve a sensor for a specific drone.

        Args:
            drone_id: ID of the drone needing a sensor

        Returns:
            Sensor instance for the drone
        """
        if drone_id in self.sensor_config:
            return self.sensor_config[drone_id]
        else:
            # Create default camera sensor
            return CameraSensor(**self.default_sensor_params)

    def _generate_map(self) -> np.ndarray:
        """
        Generate a random map for exploration.

        Returns:
            2D numpy array representing the map
        """
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Add walls on borders
        grid[0, :] = TileType.WALL
        grid[-1, :] = TileType.WALL
        grid[:, 0] = TileType.WALL
        grid[:, -1] = TileType.WALL

        # Add random internal walls
        num_walls = int(self.width * self.height * 0.15)
        for _ in range(num_walls):
            x = np.random.randint(1, self.width - 1)
            y = np.random.randint(1, self.height - 1)
            grid[y, x] = TileType.WALL

        # Add some doors and windows
        num_doors = max(2, int(self.width * self.height * 0.02))
        for _ in range(num_doors):
            x = np.random.randint(1, self.width - 1)
            y = np.random.randint(1, self.height - 1)
            if grid[y, x] == TileType.FREE_SPACE:
                grid[y, x] = np.random.choice([TileType.DOOR_CLOSED, TileType.DOOR_OPEN])

        # Add entry points for drones
        for i in range(self.num_agents):
            attempts = 0
            while attempts < 100:
                x = np.random.randint(1, self.width - 1)
                y = np.random.randint(1, self.height - 1)
                if grid[y, x] == TileType.FREE_SPACE:
                    grid[y, x] = TileType.ENTRY_POINT
                    break
                attempts += 1

        return grid

    def _load_map(self, path: str) -> np.ndarray:
        """Load a map from file."""
        return np.loadtxt(path, dtype=np.int8)

    def _find_entry_points(self) -> List[Tuple[int, int]]:
        """
        Find or create entry points for drone spawning.

        Returns:
            List of (x, y) entry point coordinates
        """
        entry_points = []

        # Find existing entry points
        for y in range(self.height):
            for x in range(self.width):
                if self.true_map[y, x] == TileType.ENTRY_POINT:
                    entry_points.append((x, y))

        # Create more entry points if needed
        while len(entry_points) < self.num_agents:
            free_spaces = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.true_map[y, x] in {TileType.FREE_SPACE, TileType.DOOR_OPEN}:
                        free_spaces.append((x, y))

            if free_spaces:
                x, y = free_spaces[np.random.randint(len(free_spaces))]
                self.true_map[y, x] = TileType.ENTRY_POINT
                entry_points.append((x, y))
            else:
                break

        return entry_points[:self.num_agents]

    def _compute_reachable_mask(self) -> np.ndarray:
        """
        Compute which cells are reachable/discoverable using BFS.

        Returns:
            Boolean mask of reachable cells
        """
        reachable = np.zeros((self.height, self.width), dtype=bool)
        visited = np.zeros((self.height, self.width), dtype=bool)

        # Find all entry points to start BFS
        queue = []
        for y in range(self.height):
            for x in range(self.width):
                if self.true_map[y, x] == TileType.ENTRY_POINT:
                    queue.append((x, y))

        # BFS to find reachable cells
        while queue:
            x, y = queue.pop(0)
            if visited[y, x]:
                continue
            visited[y, x] = True
            reachable[y, x] = True

            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if not visited[ny, nx]:
                        tile = self.true_map[ny, nx]
                        if tile not in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                            queue.append((nx, ny))

        # Also mark walls adjacent to reachable cells as discoverable
        final_reachable = reachable.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.true_map[y, x] in {TileType.WALL, TileType.DOOR_CLOSED}:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if reachable[ny, nx]:
                                final_reachable[y, x] = True
                                break

        return final_reachable

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)

        # Reset communication
        self.comm.reset()

        # Generate or load map
        if self.map_path:
            self.true_map = self._load_map(self.map_path)
            # Update dimensions based on loaded map
            self.height, self.width = self.true_map.shape
            # Redefine spaces if dimensions changed
            self._define_spaces()
        elif self.randomize:
            self.true_map = self._generate_map()
        else:
            # Default simple map
            self.true_map = np.zeros((self.height, self.width), dtype=np.int8)
            self.true_map[0, :] = TileType.WALL
            self.true_map[-1, :] = TileType.WALL
            self.true_map[:, 0] = TileType.WALL
            self.true_map[:, -1] = TileType.WALL
            # Add entry points
            for i in range(self.num_agents):
                x = 1 + i * 2
                y = self.height // 2
                if x < self.width - 1:
                    self.true_map[y, x] = TileType.ENTRY_POINT

        # Initialize global map (all unknown)
        self.global_map = np.full((self.height, self.width), TileType.UNKNOWN, dtype=np.int8)

        # Find entry points
        entry_points = self._find_entry_points()

        # Initialize drones with sensors
        self.drones = []
        for i in range(self.num_agents):
            if i < len(entry_points):
                x, y = entry_points[i]
            else:
                x, y = self.width // 2, self.height // 2

            drone = DroneState(
                drone_id=i,
                pos=(x, y),
                facing=DIRECTIONS[i % 4],
                active=i == 0 or self.num_agents == 1,  # First drone starts active
                entry_time=i * 10,  # Stagger entry times
                sensor=self._create_sensor(i)
            )
            self.drones.append(drone)

        # Compute reachable mask for progress tracking
        self.reachable_mask = self._compute_reachable_mask()
        self.total_reachable = np.sum(self.reachable_mask)

        # Reset episode tracking
        self.current_step = 0

        # Share initial global map through communication
        self.comm.set_global_map(self.global_map)

        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute actions and step the environment.

        Args:
            actions: Array of N actions, one for each drone

        Returns:
            Tuple of (observations, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Initialize total reward for the entire environment
        total_reward = 0.0
        total_discoveries = 0
        total_collisions = 0

        # Process each drone
        for i, drone in enumerate(self.drones):
            # Activate drone if it's time
            if not drone.active and self.current_step >= drone.entry_time:
                drone.active = True

            if not drone.active:
                continue

            # Get action for this drone
            action = Action(actions[i])

            # Add step penalty
            total_reward += self.step_penalty

            # Execute action
            collision_occurred = False
            if action == Action.TURN_LEFT:
                drone.turn('TURN_LEFT')
            elif action == Action.TURN_RIGHT:
                drone.turn('TURN_RIGHT')
            elif action == Action.FORWARD:
                # Calculate new position
                dx, dy = DIRECTION_DELTAS[drone.facing]
                new_x = drone.pos[0] + dx
                new_y = drone.pos[1] + dy

                # Check for collision
                collision = False

                # Boundary collision
                if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                    collision = True
                # Obstacle collision
                elif self.true_map[new_y, new_x] in {TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS}:
                    collision = True
                else:
                    # Check collision with other active drones
                    for other_drone in self.drones:
                        if drone.drone_id != other_drone.drone_id and other_drone.active:
                            if other_drone.pos == (new_x, new_y):
                                collision = True
                                break

                if collision:
                    # Apply collision penalty to total reward
                    total_reward += self.collision_penalty
                    total_collisions += 1
                    drone.add_collision()
                    collision_occurred = True
                else:
                    # Move to new position
                    drone.update_position((new_x, new_y))

            # Perform sensing (regardless of action, but limited discoveries on collision)
            observations = drone.sensor.sense(drone.pos, drone.facing, self.true_map)

            # Update global map and count discoveries
            new_discoveries = []
            for x, y, value in observations:
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.global_map[y, x] == TileType.UNKNOWN:
                        self.global_map[y, x] = value
                        # Reduce discovery reward if collision just occurred (sensor disruption)
                        if not collision_occurred:
                            total_discoveries += 1
                        new_discoveries.append((x, y, value))

            # Update drone discoveries
            drone.add_discoveries(new_discoveries)

            # Broadcast discoveries through communication
            if new_discoveries:
                self.comm.broadcast_map_update(new_discoveries)

            # Broadcast drone state through communication
            self.comm.broadcast_state(i, drone.to_dict())

        # Add discovery reward to total
        total_reward += total_discoveries * self.discovery_reward

        # Update global map in communication
        self.comm.set_global_map(self.global_map)

        # Check termination conditions
        terminated = False
        truncated = False

        # Check if exploration is complete
        discovered = np.sum((self.global_map != TileType.UNKNOWN) & self.reachable_mask)
        if discovered >= self.total_reachable * 1.0:  # 99% to account for edge cases
            terminated = True
            # Add completion bonus to total reward
            total_reward += self.completion_bonus

        # Check step limit
        if self.current_step >= self.max_steps:
            truncated = True

        # Get observations and info
        observations = self._get_observations()
        info = self._get_info()

        return observations, total_reward, terminated, truncated, info

    def _get_observations(self) -> Dict:
        """
        Get unified observation for all agents.

        Returns:
            Dictionary with global map, positions array, facings array, and active array
        """
        # Collect positions, facings, and active states
        positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        facings = np.zeros(self.num_agents, dtype=np.int32)
        active = np.zeros(self.num_agents, dtype=np.int8)

        for drone in self.drones:
            i = drone.drone_id
            positions[i] = drone.pos
            facings[i] = drone.get_facing_idx()
            active[i] = int(drone.active)

        return {
            'global_map': self.global_map.copy(),
            'positions': positions,
            'facings': facings,
            'active': active,
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Get environment information.

        Returns:
            Dictionary with environment metrics and state
        """
        discovered = np.sum((self.global_map != TileType.UNKNOWN) & self.reachable_mask)
        progress = discovered / self.total_reachable if self.total_reachable > 0 else 0.0

        return {
            'step': self.current_step,
            'progress': progress,
            'discovered_cells': discovered,
            'total_reachable': self.total_reachable,
            'collision_counts': [d.collision_count for d in self.drones],
            'sensor_types': [d.sensor.get_sensor_type() for d in self.drones],
            'communication_states': self.comm.get_all_states(),
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode is None:
            return None

        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 12)

            # Create screen (true map + observed map side by side)
            screen_width = self.width * TILE_SIZE * 2 + 50
            screen_height = self.height * TILE_SIZE + 100
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Multi-Agent SLAM Environment")
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((30, 30, 30))

        # Draw true map (left side)
        for y in range(self.height):
            for x in range(self.width):
                tile = self.true_map[y, x]
                color = TILE_COLORS.get(tile, (150, 150, 150))
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw observed map (right side)
        offset_x = self.width * TILE_SIZE + 50
        for y in range(self.height):
            for x in range(self.width):
                tile = self.global_map[y, x]
                color = TILE_COLORS.get(tile, (150, 150, 150))
                rect = pygame.Rect(offset_x + x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw drones
        for drone in self.drones:
            if drone.active:
                color = DRONE_COLORS[drone.drone_id % len(DRONE_COLORS)]

                # Draw on both maps
                for offset in [0, offset_x]:
                    center_x = offset + drone.pos[0] * TILE_SIZE + TILE_SIZE // 2
                    center_y = drone.pos[1] * TILE_SIZE + TILE_SIZE // 2
                    pygame.draw.circle(self.screen, color, (center_x, center_y), 6)

                    # Draw facing direction arrow
                    dx, dy = DIRECTION_DELTAS[drone.facing]
                    end_x = center_x + dx * TILE_SIZE // 3
                    end_y = center_y + dy * TILE_SIZE // 3
                    pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), 2)

                    # Draw collision indicator
                    if drone.collision_count > 0:
                        pygame.draw.circle(self.screen, (255, 0, 0), (center_x, center_y), 8, 2)

        # Draw labels
        true_label = self.font.render("True Map", True, (255, 255, 255))
        obs_label = self.font.render("Observed Map", True, (255, 255, 255))
        self.screen.blit(true_label, (10, self.height * TILE_SIZE + 10))
        self.screen.blit(obs_label, (offset_x + 10, self.height * TILE_SIZE + 10))

        # Draw progress bar
        discovered = np.sum((self.global_map != TileType.UNKNOWN) & self.reachable_mask)
        progress = discovered / self.total_reachable if self.total_reachable > 0 else 0.0

        bar_y = self.height * TILE_SIZE + 40
        bar_width = self.width * TILE_SIZE * 2 + 30
        bar_height = 20

        pygame.draw.rect(self.screen, (60, 60, 60), (10, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0), (10, bar_y, int(bar_width * progress), bar_height))

        # Draw progress text
        progress_text = self.font.render(
            f"Progress: {progress*100:.1f}% ({discovered}/{self.total_reachable}) | Step: {self.current_step}/{self.max_steps}",
            True, (255, 255, 255)
        )
        self.screen.blit(progress_text, (10, bar_y + 25))

        # Update display
        pygame.display.flip()
        self.clock.tick(FPS)

        # Return RGB array if requested
        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

        return None

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None