"""
Wall Following Wrapper for single-agent training - OPTIMIZED VERSION.

This wrapper trains an agent to:
1. Start only when a wall is already visible (efficient pre-search)
2. Approach and stick to the closest visible wall
3. Follow along only that single wall from end to end
4. Stop once that single wall is fully explored (INCLUDING boundaries/red cells)
"""

from typing import Dict, Tuple, Set, Optional, List
import numpy as np
import time
from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, DIRECTION_DELTAS


class WallFollowingWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training wall-following behavior.

    The environment starts only when a wall is visible, using an efficient
    pre-search phase during reset.
    """

    def __init__(self, env_config: Dict = None):
        super().__init__(env_config)

        # Wall tracking
        self.target_wall_segment: Set[Tuple[int, int]] = set()
        self.accessible_wall_cells: Set[Tuple[int, int]] = set()
        self.wall_boundaries: Set[Tuple[int, int]] = set()
        self.discovered_cells: Set[Tuple[int, int]] = set()

        # Behavior tracking
        self.phase = 'approaching'  # Start directly in approaching phase
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0

        # Pre-search timing
        self.pre_search_time = 0.0
        self.pre_search_steps = 0

    def reset(self, **kwargs):
        """Reset environment and perform efficient pre-search for walls."""
        # First, do the standard reset
        result = super().reset(**kwargs)

        # Handle both old gym (returns obs) and new gym (returns obs, info) formats
        if isinstance(result, tuple):
            obs, info = result
            return_info = True
        else:
            obs = result
            info = {}
            return_info = False

        # Now perform pre-search to find a wall
        start_time = time.time()
        self.pre_search_steps = 0

        # Efficient pre-search: random walk until we find a wall
        obs = self._efficient_pre_search(obs)

        self.pre_search_time = time.time() - start_time

        # Initialize wall tracking based on found wall
        self._initialize_wall_tracking(obs)

        # Return in the same format we received
        if return_info:
            info['pre_search_time'] = self.pre_search_time
            info['pre_search_steps'] = self.pre_search_steps
            return obs, info
        else:
            return obs

    def _efficient_pre_search(self, initial_obs):
        """
        Perform efficient random walk until a wall is visible.
        Uses a smart exploration strategy to quickly find walls.
        """
        obs = initial_obs
        max_pre_search_steps = 500
        found_wall = False

        # Action probabilities for smart exploration
        # Prefer forward movement with occasional turns
        action_probs = [0.6, 0.15, 0.15, 0.1]  # [forward, turn_left, turn_right, backward]
        actions = [0, 1, 2, 3]

        while not found_wall and self.pre_search_steps < max_pre_search_steps:
            # Check if we can see any walls
            global_map = obs['global_map']
            visible_walls = self._find_visible_walls(global_map)

            if visible_walls:
                found_wall = True
                break

            # Take a random action with smart probabilities
            action = np.random.choice(actions, p=action_probs)

            # The SLAM environment expects an array of actions (one per agent)
            # For single agent, we need to wrap the action in an array
            action_array = np.array([action])

            # Step the base environment directly (bypass wrapper logic during pre-search)
            step_result = self.env.step(action_array)
            # Handle both old (4-tuple) and new (5-tuple) gym step formats
            if len(step_result) == 5:
                obs, _, _, _, info = step_result
            else:
                obs, _, _, info = step_result
            self.pre_search_steps += 1

            # Occasionally do a full rotation to scan surroundings
            if self.pre_search_steps % 20 == 0:
                for _ in range(4):  # Full 360 rotation
                    action_array = np.array([1])  # Turn left
                    step_result = self.env.step(action_array)
                    if len(step_result) == 5:
                        obs, _, _, _, _ = step_result
                    else:
                        obs, _, _, _ = step_result
                    global_map = obs['global_map']
                    if self._find_visible_walls(global_map):
                        found_wall = True
                        break

        return obs

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
            self.wall_boundaries = self._find_wall_boundaries(self.accessible_wall_cells)
            self.wall_locked = True

            # Check if we're already adjacent to the wall
            if self._is_adjacent_to_wall(pos):
                self.phase = 'following'
                self.wall_contact_steps = 1
            else:
                self.phase = 'approaching'

            self.last_distance = self._distance_to_wall(pos)

    def _reset_task(self):
        """Reset wall-following specific state."""
        self.target_wall_segment = set()
        self.accessible_wall_cells = set()
        self.wall_boundaries = set()
        self.discovered_cells = set()
        self.phase = 'approaching'  # Default to approaching
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0
        self.pre_search_time = 0.0
        self.pre_search_steps = 0

    def _find_visible_walls(self, global_map) -> Set[Tuple[int, int]]:
        """Find all currently visible (not unknown) wall cells."""
        visible_walls = set()
        for y in range(global_map.shape[0]):
            for x in range(global_map.shape[1]):
                if global_map[y, x] == TileType.WALL:
                    visible_walls.add((x, y))
        return visible_walls

    def _select_target_wall(self, pos: Tuple[int, int], facing: int, global_map) -> Optional[Tuple[int, int]]:
        """
        Select which wall cell to target based on facing direction.
        Prioritize walls the agent is looking at.
        """
        # Facing directions: 0=North(-y), 1=East(+x), 2=South(+y), 3=West(-x)
        facing_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = facing_dirs[facing]

        # First, check for walls in the direction we're facing
        # Check multiple cells ahead in facing direction
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
        """
        Trace a SINGLE continuous wall segment based on agent's facing direction.
        At corners/junctions, choose the segment aligned with viewing direction.
        """
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

        # Determine wall orientation based on facing and connections
        is_vertical = False

        # If this is a corner or junction, decide based on agent's view angle
        if vertical_connections > 0 and horizontal_connections > 0:
            # Corner or junction detected
            # Facing: 0=North, 1=East, 2=South, 3=West

            # Calculate if agent is looking more at horizontal or vertical wall
            wall_rel_x = wx - agent_pos[0]
            wall_rel_y = wy - agent_pos[1]

            if facing == 0:  # Looking North
                # If wall is above us, and has horizontal extension, follow horizontal
                if wall_rel_y < 0:
                    is_vertical = False if horizontal_connections > 0 else True
                else:
                    is_vertical = True
            elif facing == 1:  # Looking East
                # If wall is to our right, and has vertical extension, follow vertical
                if wall_rel_x > 0:
                    is_vertical = True if vertical_connections > 0 else False
                else:
                    is_vertical = False
            elif facing == 2:  # Looking South
                # If wall is below us, and has horizontal extension, follow horizontal
                if wall_rel_y > 0:
                    is_vertical = False if horizontal_connections > 0 else True
                else:
                    is_vertical = True
            elif facing == 3:  # Looking West
                # If wall is to our left, and has vertical extension, follow vertical
                if wall_rel_x < 0:
                    is_vertical = True if vertical_connections > 0 else False
                else:
                    is_vertical = False
        elif vertical_connections > 0 and horizontal_connections == 0:
            is_vertical = True
        elif horizontal_connections > 0 and vertical_connections == 0:
            is_vertical = False
        else:
            # Isolated wall cell - just return it
            return segment

        if is_vertical:
            # Trace vertical wall, but stop at junctions/corners
            # Go up
            for y in range(wy - 1, -1, -1):
                if true_map[y, wx] != TileType.WALL:
                    break
                # Check for junction/corner
                connections = 0
                if wx > 0 and true_map[y, wx-1] == TileType.WALL:
                    connections += 1
                if wx < true_map.shape[1]-1 and true_map[y, wx+1] == TileType.WALL:
                    connections += 1
                if connections > 0:  # This is a corner/junction
                    break
                segment.add((wx, y))

            # Go down
            for y in range(wy + 1, true_map.shape[0]):
                if true_map[y, wx] != TileType.WALL:
                    break
                # Check for junction/corner
                connections = 0
                if wx > 0 and true_map[y, wx-1] == TileType.WALL:
                    connections += 1
                if wx < true_map.shape[1]-1 and true_map[y, wx+1] == TileType.WALL:
                    connections += 1
                if connections > 0:  # This is a corner/junction
                    break
                segment.add((wx, y))
        else:
            # Trace horizontal wall, but stop at junctions/corners
            # Go left
            for x in range(wx - 1, -1, -1):
                if true_map[wy, x] != TileType.WALL:
                    break
                # Check for junction/corner
                connections = 0
                if wy > 0 and true_map[wy-1, x] == TileType.WALL:
                    connections += 1
                if wy < true_map.shape[0]-1 and true_map[wy+1, x] == TileType.WALL:
                    connections += 1
                if connections > 0:  # This is a corner/junction
                    break
                segment.add((x, wy))

            # Go right
            for x in range(wx + 1, true_map.shape[1]):
                if true_map[wy, x] != TileType.WALL:
                    break
                # Check for junction/corner
                connections = 0
                if wy > 0 and true_map[wy-1, x] == TileType.WALL:
                    connections += 1
                if wy < true_map.shape[0]-1 and true_map[wy+1, x] == TileType.WALL:
                    connections += 1
                if connections > 0:  # This is a corner/junction
                    break
                segment.add((x, wy))

        return segment

    def _find_accessible_wall_cells(self, wall_segment: Set[Tuple[int, int]],
                                   agent_pos: Tuple[int, int], true_map) -> Set[Tuple[int, int]]:
        """
        Find which wall cells from the segment are accessible from the agent's current position.
        """
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

    def _find_wall_boundaries(self, accessible_cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Find the boundary cells (red cells) - the start and end of the accessible wall segment.
        """
        if not accessible_cells:
            return set()

        if len(accessible_cells) == 1:
            return accessible_cells

        # Convert to list and sort to find endpoints
        accessible_list = list(accessible_cells)

        # Check if vertical or horizontal
        first_cell = accessible_list[0]
        is_vertical = all(cell[0] == first_cell[0] for cell in accessible_list)

        if is_vertical:
            accessible_list.sort(key=lambda cell: cell[1])
        else:
            accessible_list.sort(key=lambda cell: cell[0])

        # Return first and last cells as boundaries
        return {accessible_list[0], accessible_list[-1]}

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

    def _is_moving_along_wall(self, obs, action: int) -> bool:
        """
        Check if the agent is moving along the wall's primary axis.
        This encourages scanning behavior appropriate to wall orientation.
        """
        if not self.accessible_wall_cells:
            return False

        pos = tuple(obs['positions'][0])
        facing = obs['facings'][0]  # 0=North, 1=East, 2=South, 3=West

        # Determine if wall is primarily vertical or horizontal
        wall_list = list(self.accessible_wall_cells)
        if len(wall_list) < 2:
            return True  # Single cell wall, any movement is fine

        # Check wall orientation
        is_vertical = all(cell[0] == wall_list[0][0] for cell in wall_list)

        # Find which side of the wall the agent is on
        wall_x = wall_list[0][0] if is_vertical else None
        wall_y = wall_list[0][1] if not is_vertical else None

        # Actions: 0=forward, 1=turn_left, 2=turn_right, 3=backward
        # Facing: 0=North(-y), 1=East(+x), 2=South(+y), 3=West(-x)

        if is_vertical:
            # Vertical wall - we want to move North/South along it
            agent_on_left = pos[0] < wall_x if wall_x else False
            agent_on_right = pos[0] > wall_x if wall_x else False

            # Check if action will move us along the wall (north/south)
            if action == 0:  # Forward
                # Good if facing north or south
                return facing in [0, 2]
            elif action == 3:  # Backward
                # Good if facing north or south (moves opposite direction)
                return facing in [0, 2]
            elif action in [1, 2]:  # Turning
                # Turning can be good to orient properly
                return True
        else:
            # Horizontal wall - we want to move East/West along it
            agent_above = pos[1] < wall_y if wall_y else False
            agent_below = pos[1] > wall_y if wall_y else False

            # Check if action will move us along the wall (east/west)
            if action == 0:  # Forward
                # Good if facing east or west
                return facing in [1, 3]
            elif action == 3:  # Backward
                # Good if facing east or west (moves opposite direction)
                return facing in [1, 3]
            elif action in [1, 2]:  # Turning
                # Turning can be good to orient properly
                return True

        return False

    def _update_discoveries(self, pos: Tuple[int, int], global_map):
        """Update discovered cells based on current position and vision - INCLUDING boundaries."""
        # Check ALL accessible cells, including boundaries
        for wx, wy in self.accessible_wall_cells:
            if global_map[wy, wx] != TileType.UNKNOWN:
                self.discovered_cells.add((wx, wy))

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute wall-following specific reward."""
        reward = -0.01  # Small step penalty

        pos = tuple(obs['positions'][0])
        global_map = obs['global_map']

        # Since we start with a wall already found, we're either approaching or following

        # Phase: APPROACHING - Get to the wall
        if self.phase == 'approaching':
            dist = self._distance_to_wall(pos)

            if self._is_adjacent_to_wall(pos):
                self.phase = 'following'
                self.wall_contact_steps = 1
                reward += 15.0
            else:
                if dist < self.last_distance:
                    reward += 2.0
                elif dist > self.last_distance:
                    reward -= 1.0

            self.last_distance = dist

        # Phase: FOLLOWING - Explore the entire wall
        elif self.phase == 'following':
            adjacent = self._is_adjacent_to_wall(pos)

            old_discovered = len(self.discovered_cells)
            self._update_discoveries(pos, global_map)
            new_discovered = len(self.discovered_cells)

            if adjacent:
                self.wall_contact_steps += 1
                reward += 1.5

                if new_discovered > old_discovered:
                    reward += (new_discovered - old_discovered) * 3.0
                    self.no_new_discovery_steps = 0
                else:
                    self.no_new_discovery_steps += 1

                # Bonus for moving along the wall's primary axis
                if self._is_moving_along_wall(obs, action):
                    reward += 0.5
            else:
                reward -= 3.0
                self.wall_contact_steps = 0

        # Collision penalty
        if base_reward < -0.5:
            reward -= 5.0

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if wall-following task is complete - must discover ALL accessible cells including boundaries."""
        if self.wall_locked and self.accessible_wall_cells:
            self._update_discoveries(tuple(obs['positions'][0]), obs['global_map'])

            # Must discover ALL accessible cells (no exclusion of boundaries)
            if len(self.discovered_cells) >= len(self.accessible_wall_cells):
                return TaskStatus.SUCCESS

            # Allow for 95% coverage as fallback
            coverage = len(self.discovered_cells) / len(self.accessible_wall_cells)
            if coverage >= 0.95:
                return TaskStatus.SUCCESS

            # If stuck for too long with high coverage
            if self.no_new_discovery_steps > 30 and coverage >= 0.85:
                return TaskStatus.SUCCESS

        if self.task_step > 500:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def get_info(self) -> Dict:
        """Get additional task-specific information."""
        coverage = 0.0
        if self.accessible_wall_cells:
            coverage = len(self.discovered_cells) / len(self.accessible_wall_cells)

        info = {
            'phase': self.phase,
            'wall_locked': self.wall_locked,
            'wall_contact_steps': self.wall_contact_steps,
            'wall_coverage': coverage,
            'discovered_accessible': len(self.discovered_cells),
            'total_accessible': len(self.accessible_wall_cells),
            'total_wall_cells': len(self.target_wall_segment),
            'pre_search_time': self.pre_search_time,
            'pre_search_steps': self.pre_search_steps,
        }
        return info

    def render(self):
        """Override render to highlight target wall with proper coloring."""
        super().render()

        if hasattr(self, 'env') and self.env.screen is not None and self.target_wall_segment:
            import pygame
            from environments.base.constants import TILE_SIZE

            # Color boundary cells in red (visual only - they still need to be discovered)
            for wx, wy in self.wall_boundaries:
                rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.env.screen, (255, 0, 0), rect)

            # Color accessible cells in orange
            for wx, wy in self.accessible_wall_cells:
                if (wx, wy) not in self.wall_boundaries:
                    rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                    pygame.draw.rect(self.env.screen, (255, 128, 0), rect)

            # Color discovered cells in green on observed map
            offset_x = self.env.width * TILE_SIZE + 50
            for wx, wy in self.discovered_cells:
                rect = pygame.Rect(offset_x + wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.env.screen, (0, 255, 0), rect)

            pygame.display.flip()