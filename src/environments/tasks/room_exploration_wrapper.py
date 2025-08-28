"""
Ultra-optimized Room Exploration Environment Wrapper

Key optimizations:
1. Pre-computed room and doorway data
2. Auto-exploration until sufficient room discovered
3. Numpy vectorized operations
4. Cached coverage calculations
5. Minimal redundant computation
"""

import gymnasium as gym
import numpy as np
import pygame
from typing import Dict, Tuple, Optional, Set, List
from numba import njit

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, Action


@njit
def fast_coverage_check(global_map: np.ndarray, cells_x: np.ndarray, cells_y: np.ndarray, threshold: float) -> tuple:
    """Fast coverage calculation using numba."""
    discovered = 0
    total = len(cells_x)

    for i in range(total):
        if global_map[cells_y[i], cells_x[i]] != TileType.UNKNOWN:
            discovered += 1

    coverage = discovered / total if total > 0 else 0.0
    return coverage >= threshold, coverage


@njit
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


class RoomExplorationWrapper(BaseTaskWrapper):
    """
    Ultra-optimized room exploration wrapper with auto-exploration.

    Task:
    1. (Auto) Explore until a reasonable room area is discovered
    2. Complete exploration of the current room
    3. Avoid passing through doorways (failure condition)
    """

    def __init__(
        self,
        env_config: Dict = None,
        # Pre-computed room data
        precomputed_rooms: Dict = None,
        # Auto-exploration parameters
        auto_explore: bool = True,
        max_exploration_steps: int = 500,
        min_room_discovery: float = 0.1,  # Discover at least 30% of room before starting
        exploration_strategy: str = "frontier",
        # Task parameters
        exploration_reward: float = 0.1,
        door_penalty: float = -10.0,
        completion_reward: float = 10.0,
        step_penalty: float = -0.001,
        coverage_threshold: float = 1.0,
        max_task_steps: int = 500,
    ):
        super().__init__(env_config)

        # Store pre-computed data
        if precomputed_rooms is not None:
            self.all_doorways = precomputed_rooms['doorways']
            self.all_rooms = precomputed_rooms['rooms']
            self.room_boundaries = precomputed_rooms['room_boundaries']
        else:
            # Fallback - compute at runtime
            self.all_doorways = set()
            self.all_rooms = []
            self.room_boundaries = {}

        # Convert doorways to numpy array for fast checks
        if self.all_doorways:
            doorway_list = list(self.all_doorways)
            self.doorways_x = np.array([d[0] for d in doorway_list], dtype=np.int32)
            self.doorways_y = np.array([d[1] for d in doorway_list], dtype=np.int32)
        else:
            self.doorways_x = np.array([], dtype=np.int32)
            self.doorways_y = np.array([], dtype=np.int32)

        # Auto-exploration parameters
        self.auto_explore = auto_explore
        self.max_exploration_steps = max_exploration_steps
        self.min_room_discovery = min_room_discovery
        self.exploration_strategy = exploration_strategy
        self.is_exploring = False
        self.exploration_steps = 0

        # Task parameters
        self.exploration_reward = exploration_reward
        self.door_penalty = door_penalty
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
        self.coverage_threshold = coverage_threshold
        self.max_task_steps = max_task_steps

        # Room tracking
        self.current_room = None
        self.current_room_cells = set()
        self.current_room_array_x = None
        self.current_room_array_y = None
        self.discovered_doorways = set()

        # Cache - Use separate hashes for different checks
        self.last_doorway_hash = None  # For doorway checking
        self.last_completion_hash = None  # For completion checking
        self.last_coverage = 0.0
        self.last_completion_check = False
        self.passed_through_door = False
        self.completion_achieved = False

    def reset(self, **kwargs):
        """Reset environment and run auto-exploration."""
        obs, info = super().reset(**kwargs)

        # Reset state
        self._reset_task()

        # Identify current room based on starting position
        self._identify_current_room()

        # Run auto-exploration if enabled
        if self.auto_explore:
            obs, info = self._run_auto_exploration()

        return obs, info

    def _reset_task(self):
        """Reset task-specific state."""
        self.current_room = None
        self.current_room_cells = set()
        self.current_room_array_x = None
        self.current_room_array_y = None
        self.discovered_doorways = set()
        self.last_doorway_hash = None
        self.last_completion_hash = None
        self.last_coverage = 0.0
        self.last_completion_check = False
        self.passed_through_door = False
        self.completion_achieved = False
        self.is_exploring = False
        self.exploration_steps = 0

    def _identify_current_room(self):
        """Identify which room the drone starts in."""
        start_pos = tuple(self.env.drones[0].pos)

        # Find room containing start position
        for room_id, room_cells in enumerate(self.all_rooms):
            if start_pos in room_cells:
                self.current_room = room_id
                # Include room cells and boundary walls
                self.current_room_cells = room_cells.copy()
                if room_id in self.room_boundaries:
                    self.current_room_cells.update(self.room_boundaries[room_id])
                break

        # If no pre-computed room, compute it
        if self.current_room is None:
            self._compute_current_room_runtime()

        # Convert to numpy arrays for fast checking
        if self.current_room_cells:
            cells_list = list(self.current_room_cells)
            self.current_room_array_x = np.array([c[0] for c in cells_list], dtype=np.int32)
            self.current_room_array_y = np.array([c[1] for c in cells_list], dtype=np.int32)
        else:
            self.current_room_array_x = np.array([], dtype=np.int32)
            self.current_room_array_y = np.array([], dtype=np.int32)

    def _compute_current_room_runtime(self):
        """Compute current room at runtime if not pre-computed."""
        true_map = self.env.true_map
        start_pos = self.env.drones[0].pos

        # BFS to find room cells
        room_cells = set()
        queue = [start_pos]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Don't cross doorways
            if (x, y) in self.all_doorways:
                continue

            # Check if traversable
            if true_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN, TileType.ENTRY_POINT]:
                room_cells.add((x, y))

                # Add neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                        if (nx, ny) not in visited:
                            queue.append((nx, ny))

        # Include adjacent walls
        self.current_room_cells = room_cells.copy()
        for x, y in room_cells:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    if true_map[ny, nx] in [TileType.WALL, TileType.DOOR_CLOSED]:
                        self.current_room_cells.add((nx, ny))

    def _run_auto_exploration(self):
        """Run automatic exploration until sufficient room area is discovered."""
        # print(f"Starting auto-exploration (max {self.max_exploration_steps} steps)...")
        self.is_exploring = True

        for step in range(self.max_exploration_steps):
            # Check room discovery progress
            if self._check_room_discovery_progress():
                # print(f"Room sufficiently discovered after {step} exploration steps")
                break

            # Choose exploration action
            if self.exploration_strategy == "frontier":
                action = self._frontier_exploration_action()
            else:
                action = self._random_exploration_action()

            # Execute action
            actions = np.array([action], dtype=np.int32)
            obs, _, _, _, _ = self.env.step(actions)
            self.exploration_steps = step + 1

        self.is_exploring = False
        # print(f"Exploration complete. Room coverage: {self.last_coverage:.1%}")

        return self._get_observations(), self._get_info()

    def _check_room_discovery_progress(self) -> bool:
        """Check if enough of the room has been discovered."""
        if len(self.current_room_array_x) == 0:
            return False

        _, coverage = fast_coverage_check(
            self.env.global_map,
            self.current_room_array_x,
            self.current_room_array_y,
            self.min_room_discovery
        )
        self.last_coverage = coverage
        return coverage >= self.min_room_discovery

    def _frontier_exploration_action(self):
        """Choose action for frontier-based exploration."""
        drone = self.env.drones[0]
        drone_x, drone_y = drone.pos
        global_map = self.env.global_map

        # Find frontiers within the room
        frontiers = []
        for cell in self.current_room_cells:
            x, y = cell
            if global_map[y, x] == TileType.UNKNOWN:
                # Check if adjacent to known space
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.env.width and 0 <= ny < self.env.height and
                        global_map[ny, nx] in [TileType.FREE_SPACE, TileType.DOOR_OPEN]):
                        frontiers.append((x, y))
                        break

        if not frontiers:
            return self._random_exploration_action()

        # Find nearest frontier
        min_dist = float('inf')
        target = None
        for fx, fy in frontiers:
            dist = manhattan_distance(drone_x, drone_y, fx, fy)
            if dist < min_dist:
                min_dist = dist
                target = (fx, fy)

        if target:
            return self._move_towards_target(drone, target)

        return self._random_exploration_action()

    def _move_towards_target(self, drone, target):
        """Choose action to move towards target."""
        dx = target[0] - drone.pos[0]
        dy = target[1] - drone.pos[1]

        facing_idx = drone.get_facing_idx()
        facing_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        current_dx, current_dy = facing_deltas[facing_idx]

        # Determine desired direction
        if abs(dx) > abs(dy):
            desired_dx = 1 if dx > 0 else -1
            desired_dy = 0
        else:
            desired_dx = 0
            desired_dy = 1 if dy > 0 else -1

        # If facing right direction, move forward
        if (current_dx, current_dy) == (desired_dx, desired_dy):
            return Action.FORWARD

        # Calculate turns
        desired_facing = facing_deltas.index((desired_dx, desired_dy))
        turns_right = (desired_facing - facing_idx) % 4

        if turns_right <= 2:
            return Action.TURN_RIGHT
        else:
            return Action.TURN_LEFT

    def _random_exploration_action(self):
        """Random exploration with forward bias."""
        return np.random.choice(4, p=[0.7, 0.15, 0.15, 0.0])

    def _check_doorways_fast(self):
        """Fast doorway discovery check."""
        global_map = self.env.global_map

        # Check map change - use doorway-specific hash
        map_hash = hash(global_map.tobytes())
        if map_hash == self.last_doorway_hash:
            return False
        self.last_doorway_hash = map_hash

        # Check doorways using numpy arrays
        new_doorways = False
        for i in range(len(self.doorways_x)):
            x, y = self.doorways_x[i], self.doorways_y[i]
            if (x, y) not in self.discovered_doorways:
                # Quick visibility check
                if global_map[y, x] != TileType.UNKNOWN:
                    self.discovered_doorways.add((x, y))
                    new_doorways = True

        return new_doorways

    def _check_room_completion(self):
        """Check if room exploration is complete (cached)."""
        # Use completion-specific hash for caching
        current_hash = hash(self.env.global_map.tobytes())

        # Use cached result if map hasn't changed
        if self.last_completion_hash == current_hash:
            return self.last_completion_check

        # Update the hash for next time
        self.last_completion_hash = current_hash

        # Check coverage
        if len(self.current_room_array_x) > 0:
            is_complete, coverage = fast_coverage_check(
                self.env.global_map,
                self.current_room_array_x,
                self.current_room_array_y,
                self.coverage_threshold
            )
            self.last_coverage = coverage
            self.last_completion_check = is_complete
            return is_complete

        return False

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute room exploration reward."""
        drone_pos = tuple(obs['positions'][0])

        # Check for new doorways
        self._check_doorways_fast()

        # Base reward
        reward = base_reward * self.exploration_reward + self.step_penalty

        # Check doorway crossing
        if drone_pos in self.discovered_doorways and not self.passed_through_door:
            reward += self.door_penalty
            self.passed_through_door = True

        # Check room completion
        if not self.completion_achieved and not self.passed_through_door:
            if self._check_room_completion():
                reward += self.completion_reward
                self.completion_achieved = True

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check task completion status."""
        if self.passed_through_door:
            return TaskStatus.FAILURE

        if self.completion_achieved:
            return TaskStatus.SUCCESS

        if self.task_step >= self.max_task_steps:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def _get_observations(self):
        """Get observations from base environment."""
        return self.env._get_observations()

    def _get_info(self):
        """Get info with task details."""
        info = self.env._get_info()
        info.update({
            'room_coverage': self.last_coverage,
            'discovered_doorways': len(self.discovered_doorways),
            'passed_through_door': self.passed_through_door,
            'completion_achieved': self.completion_achieved,
            'exploration_steps': self.exploration_steps,
            'is_exploring': self.is_exploring,
        })
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render with room and doorway visualization."""
        if self.env.render_mode is None:
            return None

        base_render = self.env.render()

        if self.env.screen is not None:
            import pygame

            drone_pos = self.env.drones[0].pos

            # Highlight room cells on true map (left)
            for x, y in self.current_room_cells:
                if (x, y) not in self.all_doorways:
                    rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                    # Light blue tint for room area
                    s = pygame.Surface((TILE_SIZE - 1, TILE_SIZE - 1))
                    s.set_alpha(30)
                    s.fill((0, 100, 255))
                    self.env.screen.blit(s, (x * TILE_SIZE, y * TILE_SIZE))

            # Highlight doorways
            for map_offset in [0, self.env.width * TILE_SIZE + 50]:
                for x, y in self.discovered_doorways:
                    color = (255, 0, 0) if self.passed_through_door else (255, 255, 0)
                    rect = pygame.Rect(
                        map_offset + x * TILE_SIZE,
                        y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(self.env.screen, color, rect, 3)

                    if drone_pos == (x, y):
                        s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                        s.set_alpha(50)
                        s.fill(color)
                        self.env.screen.blit(s, (map_offset + x * TILE_SIZE, y * TILE_SIZE))

            # Status text
            if self.env.font:
                if self.is_exploring:
                    status = f"AUTO-EXPLORING... Steps: {self.exploration_steps}/{self.max_exploration_steps}"
                    color = (255, 128, 0)
                elif self.passed_through_door:
                    status = f"FAILED: Passed through doorway! Steps: {self.task_step}"
                    color = (255, 0, 0)
                elif self.completion_achieved:
                    status = f"SUCCESS: Room explored! Steps: {self.task_step}"
                    color = (0, 255, 0)
                else:
                    coverage_pct = self.last_coverage * 100
                    status = f"Exploring - Steps: {self.task_step} | Coverage: {coverage_pct:.1f}%"
                    color = (255, 255, 255)

                text_surface = self.env.font.render(status, True, color)
                self.env.screen.blit(text_surface, (10, 10))

                # Room info
                room_text = f"Room cells: {len(self.current_room_cells)} | Doors: {len(self.discovered_doorways)}"
                room_surface = self.env.font.render(room_text, True, (200, 200, 200))
                self.env.screen.blit(room_surface, (10, 30))

            pygame.display.flip()

        return base_render