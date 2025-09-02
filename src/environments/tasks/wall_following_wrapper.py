"""
Wall Following Wrapper - MINIMAL FIXES VERSION

Key fixes:
1. Proper metric reset after pre-search
2. Wall-lock verification
3. Better collision detection
"""

from typing import Dict, Tuple, Set, Optional, List
import numpy as np
import time

from gymnasium import spaces

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, DIRECTION_DELTAS


class WallFollowingWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training wall-following behavior.
    """

    def __init__(self, env_config: Dict = None):
        super().__init__(env_config)

        # Wall tracking
        self.target_wall_segment: Set[Tuple[int, int]] = set()
        self.accessible_wall_cells: Set[Tuple[int, int]] = set()
        self.wall_boundaries: Set[Tuple[int, int]] = set()
        self.discovered_cells: Set[Tuple[int, int]] = set()

        # Behavior tracking
        self.phase = 'approaching'
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0

        # Pre-search timing
        self.pre_search_time = 0.0
        self.pre_search_steps = 0
        self.collision_count = 0
        self.last_known_collisions = 0  # Track collisions from base env

    def reset(self, **kwargs):
        """Reset environment and perform efficient pre-search for walls."""
        # Reset task-specific state first
        self.collision_count = 0
        self.last_known_collisions = 0
        self._reset_task()

        # Call parent reset
        obs, info = super().reset(**kwargs)

        # Perform pre-search WITHOUT affecting episode metrics
        obs = self._perform_isolated_presearch(obs)

        # Initialize wall tracking
        self._initialize_wall_tracking(obs)

        # Verify wall was found
        if not self.wall_locked:
            print(f"Warning: No wall found after {self.pre_search_steps} steps! Retrying...")
            # Try once more with extended search
            obs = self._perform_isolated_presearch(obs, max_steps=1000)
            self._initialize_wall_tracking(obs)

        # Add pre-search info
        info['pre_search_steps'] = self.pre_search_steps
        info['wall_locked'] = self.wall_locked

        return obs, info

    def _perform_isolated_presearch(self, initial_obs, max_steps=500):
        """Perform pre-search in isolation from training episode."""
        obs = initial_obs
        self.pre_search_steps = 0

        # Save environment state
        saved_step = self.env.current_step
        saved_drone_states = []
        for drone in self.env.drones:
            saved_drone_states.append({
                'pos': drone.pos,
                'facing': drone.facing,
                'collision_count': drone.collision_count
            })

        # Search for wall
        while self.pre_search_steps < max_steps:
            visible_walls = self._find_visible_walls(obs['global_map'])
            if visible_walls:
                break

            action = np.random.choice([0, 1, 2, 3], p=[0.6, 0.15, 0.15, 0.1])
            obs, _, _, _, _ = self.env.step(np.array([action]))
            self.pre_search_steps += 1

        # Restore environment state
        self.env.current_step = saved_step
        for i, drone in enumerate(self.env.drones):
            drone.pos = saved_drone_states[i]['pos']
            drone.facing = saved_drone_states[i]['facing']
            drone.collision_count = saved_drone_states[i]['collision_count']

        # Reset all metrics after pre-search
        self._reset_all_metrics()

        return obs

    def _reset_all_metrics(self):
        """Completely reset all metrics after pre-search."""
        # Reset wrapper metrics
        self.task_step = 0
        self.collision_count = 0
        self.last_known_collisions = 0

        # Reset base environment metrics
        if hasattr(self.env, 'current_step'):
            self.env.current_step = 0

        # Reset drone collision counts and discoveries
        for drone in self.env.drones:
            drone.collision_count = 0
            drone.total_discoveries = 0
            drone.discoveries = []

    def _reset_task(self):
        """Reset wall-following specific state."""
        self.target_wall_segment = set()
        self.accessible_wall_cells = set()
        self.wall_boundaries = set()
        self.discovered_cells = set()
        self.phase = 'approaching'
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0
        self.pre_search_time = 0.0
        self.pre_search_steps = 0
        self.collision_count = 0
        self.last_known_collisions = 0

    def step(self, action):
        """Execute action and compute task-specific rewards."""
        # Handle action format from DQN
        if isinstance(action, np.ndarray):
            if action.shape == ():  # Scalar array
                action = int(action.item())
            elif len(action.shape) == 1 and action.shape[0] == 1:
                action = int(action[0])
        elif not isinstance(action, (int, np.integer)):
            action = int(action)

        # Convert to multi-agent format
        actions = np.array([action], dtype=np.int32)

        # Step base environment
        obs, base_reward, terminated, truncated, info = self.env.step(actions)

        # Better collision detection using info
        collision_occurred = False
        if 'collision_counts' in info and len(info['collision_counts']) > 0:
            current_collisions = info['collision_counts'][0]
            if current_collisions > self.last_known_collisions:
                collision_occurred = True
                self.collision_count += (current_collisions - self.last_known_collisions)
                self.last_known_collisions = current_collisions

        # Compute task-specific reward
        task_reward = self._compute_task_reward(obs, action, base_reward, collision_occurred)

        # Check task completion
        self.task_status = self._check_task_status(obs, action)

        # Handle termination
        if self.task_status == TaskStatus.SUCCESS:
            task_reward += 10.0  # Final bonus
            terminated = True
            info['task_success'] = True
            info['wall_coverage'] = 1.0
        elif self.task_status == TaskStatus.FAILURE:
            truncated = True
            info['task_success'] = False
            coverage = len(self.discovered_cells) / len(self.accessible_wall_cells) if self.accessible_wall_cells else 0
            info['wall_coverage'] = coverage

        # Increment task step
        self.task_step += 1

        # Add info
        info['task_status'] = self.task_status.value
        info['task_step'] = self.task_step
        info['collision_count'] = self.collision_count
        info.update(self.get_info())

        return obs, task_reward, terminated, truncated, info

    def _initialize_wall_tracking(self, obs):
        """Initialize wall tracking based on the first visible wall."""
        pos = tuple(obs['positions'][0])
        facing = obs['facings'][0]
        global_map = obs['global_map']
        true_map = self.env.true_map

        # Store the initial facing for wall tracing decision
        self.initial_facing = facing

        # Find and lock onto the closest visible wall
        target_wall = self._select_target_wall(pos, facing, global_map)

        if target_wall:
            # Lock onto this wall segment immediately
            self.target_wall_segment = self._trace_single_wall_segment(
                target_wall, pos, true_map, global_map
            )
            self.accessible_wall_cells = self._find_accessible_wall_cells(
                self.target_wall_segment, pos, true_map
            )
            self.wall_boundaries = self._find_extended_wall_boundaries(self.accessible_wall_cells, true_map)
            self.wall_locked = True

            # Check if we're already adjacent to the wall
            if self._is_adjacent_to_wall(pos):
                self.phase = 'following'
                self.wall_contact_steps = 1
            else:
                self.phase = 'approaching'

            self.last_distance = self._distance_to_wall(pos)

    def _find_visible_walls(self, global_map) -> Set[Tuple[int, int]]:
        """Find all currently visible (not unknown) wall cells."""
        visible_walls = set()
        for y in range(global_map.shape[0]):
            for x in range(global_map.shape[1]):
                if global_map[y, x] == TileType.WALL:
                    visible_walls.add((x, y))
        return visible_walls

    def _select_target_wall(self, pos: Tuple[int, int], facing: int, global_map) -> Optional[Tuple[int, int]]:
        """Select which wall cell to target based on facing direction."""
        # Facing directions: 0=North(-y), 1=East(+x), 2=South(+y), 3=West(-x)
        facing_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = facing_dirs[facing]

        # First, check for walls in the direction we're facing
        for distance in range(1, 10):  # Look up to 10 cells ahead
            wx, wy = pos[0] + dx * distance, pos[1] + dy * distance
            if 0 <= wx < global_map.shape[1] and 0 <= wy < global_map.shape[0]:
                if global_map[wy, wx] == TileType.WALL:
                    return (wx, wy)

        # If no wall directly ahead, check for walls adjacent to the agent
        adjacent_walls = []
        for adx, ady in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            wx, wy = pos[0] + adx, pos[1] + ady
            if 0 <= wx < global_map.shape[1] and 0 <= wy < global_map.shape[0]:
                if global_map[wy, wx] == TileType.WALL:
                    adjacent_walls.append((wx, wy))

        if adjacent_walls:
            # Prefer the wall in our facing direction if it exists
            facing_wall = (pos[0] + dx, pos[1] + dy)
            if facing_wall in adjacent_walls:
                return facing_wall
            return adjacent_walls[0]

        # Otherwise, find all visible walls and choose closest
        visible_walls = self._find_visible_walls(global_map)
        if not visible_walls:
            return None

        # Find the closest wall cell
        min_dist = float('inf')
        closest_wall = None

        for wx, wy in visible_walls:
            dist = abs(wx - pos[0]) + abs(wy - pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_wall = (wx, wy)

        return closest_wall

    def _trace_single_wall_segment(self, start_pos: Tuple[int, int], agent_pos: Tuple[int, int],
                                   true_map, global_map) -> Set[Tuple[int, int]]:
        """Trace a SINGLE continuous wall segment based on agent's facing direction."""
        wx, wy = start_pos
        segment = {start_pos}

        # Get agent's facing direction
        facing = self.initial_facing if hasattr(self, 'initial_facing') else 0

        # Check immediate neighbors in true map
        has_north = (wy > 0 and true_map[wy-1, wx] == TileType.WALL)
        has_south = (wy < true_map.shape[0]-1 and true_map[wy+1, wx] == TileType.WALL)
        has_west = (wx > 0 and true_map[wy, wx-1] == TileType.WALL)
        has_east = (wx < true_map.shape[1]-1 and true_map[wy, wx+1] == TileType.WALL)

        # Count connections
        vertical_connections = int(has_north) + int(has_south)
        horizontal_connections = int(has_west) + int(has_east)

        # Determine wall orientation
        is_vertical = False

        if vertical_connections > 0 and horizontal_connections > 0:
            # Corner or junction - decide based on agent's view angle
            wall_rel_x = wx - agent_pos[0]
            wall_rel_y = wy - agent_pos[1]

            if facing == 0:  # Looking North
                is_vertical = False if (wall_rel_y < 0 and horizontal_connections > 0) else True
            elif facing == 1:  # Looking East
                is_vertical = True if (wall_rel_x > 0 and vertical_connections > 0) else False
            elif facing == 2:  # Looking South
                is_vertical = False if (wall_rel_y > 0 and horizontal_connections > 0) else True
            elif facing == 3:  # Looking West
                is_vertical = True if (wall_rel_x < 0 and vertical_connections > 0) else False
        elif vertical_connections > 0 and horizontal_connections == 0:
            is_vertical = True
        elif horizontal_connections > 0 and vertical_connections == 0:
            is_vertical = False
        else:
            # Isolated wall cell
            return segment

        if is_vertical:
            # Trace vertical wall, stop at junctions/corners
            for y in range(wy - 1, -1, -1):
                if true_map[y, wx] != TileType.WALL:
                    break
                connections = 0
                if wx > 0 and true_map[y, wx-1] == TileType.WALL:
                    connections += 1
                if wx < true_map.shape[1]-1 and true_map[y, wx+1] == TileType.WALL:
                    connections += 1
                if connections > 0:
                    break
                segment.add((wx, y))

            for y in range(wy + 1, true_map.shape[0]):
                if true_map[y, wx] != TileType.WALL:
                    break
                connections = 0
                if wx > 0 and true_map[y, wx-1] == TileType.WALL:
                    connections += 1
                if wx < true_map.shape[1]-1 and true_map[y, wx+1] == TileType.WALL:
                    connections += 1
                if connections > 0:
                    break
                segment.add((wx, y))
        else:
            # Trace horizontal wall, stop at junctions/corners
            for x in range(wx - 1, -1, -1):
                if true_map[wy, x] != TileType.WALL:
                    break
                connections = 0
                if wy > 0 and true_map[wy-1, x] == TileType.WALL:
                    connections += 1
                if wy < true_map.shape[0]-1 and true_map[wy+1, x] == TileType.WALL:
                    connections += 1
                if connections > 0:
                    break
                segment.add((x, wy))

            for x in range(wx + 1, true_map.shape[1]):
                if true_map[wy, x] != TileType.WALL:
                    break
                connections = 0
                if wy > 0 and true_map[wy-1, x] == TileType.WALL:
                    connections += 1
                if wy < true_map.shape[0]-1 and true_map[wy+1, x] == TileType.WALL:
                    connections += 1
                if connections > 0:
                    break
                segment.add((x, wy))

        return segment

    def _find_accessible_wall_cells(self, wall_segment: Set[Tuple[int, int]],
                                   agent_pos: Tuple[int, int], true_map) -> Set[Tuple[int, int]]:
        """Find which wall cells from the segment are accessible from the agent's position."""
        # Find all free spaces reachable from agent position
        reachable_spaces = set()
        to_visit = [agent_pos]
        visited = set()

        while to_visit:
            x, y = to_visit.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if 0 <= x < true_map.shape[1] and 0 <= y < true_map.shape[0]:
                if true_map[y, x] in [TileType.FREE_SPACE, TileType.ENTRY_POINT, TileType.DOOR_OPEN]:
                    reachable_spaces.add((x, y))

                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in visited:
                            to_visit.append((nx, ny))

        # Find wall cells that are adjacent to reachable spaces
        accessible_cells = set()
        for wx, wy in wall_segment:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if (nx, ny) in reachable_spaces:
                    accessible_cells.add((wx, wy))
                    break

        return accessible_cells

    def _find_extended_wall_boundaries(self, accessible_cells: Set[Tuple[int, int]], true_map) -> Set[Tuple[int, int]]:
        """
        Find the extended boundary cells - the cells one position beyond the start and end of the accessible wall segment.
        These will be the red frame cells.
        """
        if not accessible_cells:
            return set()

        if len(accessible_cells) == 1:
            # For a single cell, just return it
            return accessible_cells

        # Convert to list and sort to find endpoints
        accessible_list = list(accessible_cells)

        # Check if vertical or horizontal
        first_cell = accessible_list[0]
        is_vertical = all(cell[0] == first_cell[0] for cell in accessible_list)

        if is_vertical:
            # Sort by y coordinate
            accessible_list.sort(key=lambda cell: cell[1])
            x = accessible_list[0][0]

            # Get the actual wall endpoints
            first_y = accessible_list[0][1]
            last_y = accessible_list[-1][1]

            # SIMPLY extend one cell beyond in each direction
            extended_boundaries = set()

            # Extend one cell up (north) - don't check what's there
            extended_y = first_y - 1
            if extended_y >= 0:
                extended_boundaries.add((x, extended_y))
            else:
                # If out of bounds, use the original
                extended_boundaries.add((x, first_y))

            # Extend one cell down (south) - don't check what's there
            extended_y = last_y + 1
            if extended_y < true_map.shape[0]:
                extended_boundaries.add((x, extended_y))
            else:
                # If out of bounds, use the original
                extended_boundaries.add((x, last_y))

        else:
            # Sort by x coordinate
            accessible_list.sort(key=lambda cell: cell[0])
            y = accessible_list[0][1]

            # Get the actual wall endpoints
            first_x = accessible_list[0][0]
            last_x = accessible_list[-1][0]

            # SIMPLY extend one cell beyond in each direction
            extended_boundaries = set()

            # Extend one cell left (west) - don't check what's there
            extended_x = first_x - 1
            if extended_x >= 0:
                extended_boundaries.add((extended_x, y))
            else:
                # If out of bounds, use the original
                extended_boundaries.add((first_x, y))

            # Extend one cell right (east) - don't check what's there
            extended_x = last_x + 1
            if extended_x < true_map.shape[1]:
                extended_boundaries.add((extended_x, y))
            else:
                # If out of bounds, use the original
                extended_boundaries.add((last_x, y))

        return extended_boundaries

    def _is_adjacent_to_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to the target wall."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (pos[0] + dx, pos[1] + dy) in self.target_wall_segment:
                return True
        return False

    def _distance_to_wall(self, pos: Tuple[int, int]) -> float:
        """Calculate minimum Manhattan distance to target wall."""
        if not self.target_wall_segment:
            return float('inf')
        return min(abs(wx - pos[0]) + abs(wy - pos[1])
                  for wx, wy in self.target_wall_segment)

    def _update_discoveries(self, pos: Tuple[int, int], global_map):
        """Update discovered cells based on current position and vision."""
        # Only track cells that are part of the task
        task_cells = self.accessible_wall_cells | self.wall_boundaries

        for wx, wy in task_cells:
            if 0 <= wx < global_map.shape[1] and 0 <= wy < global_map.shape[0]:
                if global_map[wy, wx] != TileType.UNKNOWN:
                    self.discovered_cells.add((wx, wy))

    def _compute_task_reward(self, obs, action, base_reward, collision_occurred) -> float:
        """
        Sophisticated reward that incentivizes discovering the wall in minimum steps.
        Key principle: Only reward NEW discoveries, penalize inefficient exploration.
        """
        pos = tuple(obs['positions'][0])
        global_map = obs['global_map']

        # Track previously discovered cells
        old_discovered = len(self.discovered_cells)

        # Update discoveries (now includes boundaries)
        self._update_discoveries(pos, global_map)
        new_discovered = len(self.discovered_cells)

        # Calculate coverage (now includes boundaries in the total)
        total_to_discover = len(self.accessible_wall_cells | self.wall_boundaries)
        coverage = len(self.discovered_cells) / total_to_discover if total_to_discover > 0 else 0

        # BASE PENALTY - varies by situation
        if not self.wall_locked:
            # Still searching for wall
            reward = -0.01  # Small penalty during search

        elif new_discovered > old_discovered:
            # MADE PROGRESS - this is the ONLY positive reward case
            cells_discovered = new_discovered - old_discovered
            remaining = total_to_discover - new_discovered

            # Reward based on progress toward completion
            # Higher reward as we get closer to finishing (encourages completion)
            if remaining == 0:
                # Just completed the wall!
                reward = 1.0 * cells_discovered
            elif remaining <= 2:
                # Almost done - high reward to encourage finishing
                reward = 0.5 * cells_discovered
            elif remaining <= 5:
                # Close to completion
                reward = 0.3 * cells_discovered
            else:
                # Normal discovery
                reward = 0.2 * cells_discovered

            # Reset stagnation counter
            self.no_new_discovery_steps = 0

        else:
            # NO PROGRESS - apply penalties based on situation
            self.no_new_discovery_steps += 1

            # Check if adjacent to wall
            is_adjacent = self._is_adjacent_to_wall(pos)

            # Check if we're between discovered sections (useful traversal)
            is_traversing = False
            if is_adjacent and self.wall_boundaries and len(self.wall_boundaries) == 2:
                # Check if one boundary is discovered and other isn't
                discovered_boundaries = self.wall_boundaries & self.discovered_cells
                if len(discovered_boundaries) == 1:
                    # We've discovered one boundary, check if moving toward the other
                    undiscovered = self.wall_boundaries - discovered_boundaries
                    if undiscovered:
                        target = next(iter(undiscovered))
                        # Calculate if we're on the path between boundaries
                        is_traversing = self._is_on_wall_path(pos, target)

            if is_traversing:
                # Traversing along wall toward undiscovered boundary - small penalty
                reward = -0.02
            elif is_adjacent:
                # Next to wall but not making progress - medium penalty
                reward = -0.03
            else:
                # Away from wall - larger penalty
                reward = -0.05

            # Additional penalty that increases with stagnation
            if self.no_new_discovery_steps > 5:
                stagnation_penalty = min(0.1, self.no_new_discovery_steps * 0.01)
                reward -= stagnation_penalty

        # COLLISION PENALTY - use the flag we detected
        if collision_occurred:
            reward -= 2.0

        # CRITICAL ZONE PENALTY - if very close to completion but not finishing
        if coverage > 0.9 and new_discovered == old_discovered:
            # So close to done but not finishing - increasing penalty
            reward -= 0.1

        return reward

    def _is_on_wall_path(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """
        Check if position is on a valid path along the wall toward target.
        This helps identify useful traversal movements.
        """
        if not self.accessible_wall_cells:
            return False

        # Simple heuristic: check if we're adjacent to wall and moving reduces distance to target
        if not self._is_adjacent_to_wall(pos):
            return False

        # Check if we're getting closer to target along the wall
        current_dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        # Check if this position is part of the wall segment
        # (between the discovered and undiscovered parts)
        for wx, wy in self.accessible_wall_cells:
            if abs(wx - pos[0]) + abs(wy - pos[1]) == 1:  # Adjacent to this wall cell
                wall_to_target = abs(wx - target[0]) + abs(wy - target[1])
                if wall_to_target < current_dist:
                    return True

        return False

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if wall-following task is complete."""
        # Use task_step for timeout (ignores pre-search steps)
        if self.task_step >= 499:
            return TaskStatus.FAILURE

        if self.wall_locked and self.accessible_wall_cells:
            self._update_discoveries(tuple(obs['positions'][0]), obs['global_map'])

            total_to_discover = len(self.accessible_wall_cells | self.wall_boundaries)
            if len(self.discovered_cells) >= total_to_discover:
                return TaskStatus.SUCCESS

            coverage = len(self.discovered_cells) / total_to_discover
            if coverage >= 0.95:
                return TaskStatus.SUCCESS

        return TaskStatus.IN_PROGRESS

    def get_info(self) -> Dict:
        """Get additional task-specific information."""
        # Calculate coverage based on both wall cells and boundaries
        total_to_discover = len(self.accessible_wall_cells | self.wall_boundaries)
        coverage = 0.0
        if total_to_discover > 0:
            coverage = len(self.discovered_cells) / total_to_discover

        info = {
            'phase': self.phase,
            'wall_locked': self.wall_locked,
            'wall_contact_steps': self.wall_contact_steps,
            'wall_coverage': coverage,
            'discovered_accessible': len(self.discovered_cells),
            'total_accessible': len(self.accessible_wall_cells),
            'total_wall_cells': len(self.target_wall_segment),
            'total_with_boundaries': total_to_discover,
            'pre_search_time': self.pre_search_time,
            'pre_search_steps': self.pre_search_steps,
            'collision_count': self.collision_count,
        }
        return info

    def render(self):
        """Override render to highlight target wall with proper coloring."""
        super().render()

        if hasattr(self, 'env') and self.env.screen is not None and self.target_wall_segment:
            import pygame
            from environments.base.constants import TILE_SIZE

            # Draw red FRAMES (not solid fills) for boundary cells
            for wx, wy in self.wall_boundaries:
                rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                # Draw only the frame with 3 pixel width
                pygame.draw.rect(self.env.screen, (255, 0, 0), rect, 3)

            # Color accessible cells in orange (these are solid)
            for wx, wy in self.accessible_wall_cells:
                if (wx, wy) not in self.wall_boundaries:
                    rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                    pygame.draw.rect(self.env.screen, (255, 128, 0), rect, 3)

            # Color discovered cells in green on observed map
            offset_x = self.env.width * TILE_SIZE + 50
            for wx, wy in self.discovered_cells:
                rect = pygame.Rect(offset_x + wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.env.screen, (0, 255, 0), rect, 3)

            pygame.display.flip()