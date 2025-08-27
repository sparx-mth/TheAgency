"""
Wall Following Wrapper for single-agent training - FIXED VERSION.

This wrapper trains an agent to:
1. Find the CLOSEST visible wall
2. Approach and stick to it
3. Follow along only that single wall from end to end
4. Stop once that single wall is fully explored
"""

from typing import Dict, Tuple, Set, Optional, List
import numpy as np
from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, DIRECTION_DELTAS


class WallFollowingWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training wall-following behavior.

    The agent must:
    1. Find the closest visible wall
    2. Approach it and get adjacent
    3. Follow it to discover the entire wall segment from their side
    4. Episode ends when that single wall is fully explored
    """

    def __init__(self, env_config: Dict = None):
        super().__init__(env_config)

        # Wall tracking
        self.target_wall_segment: Set[Tuple[int, int]] = set()
        self.accessible_wall_cells: Set[Tuple[int, int]] = set()
        self.wall_boundaries: Set[Tuple[int, int]] = set()
        self.discovered_cells: Set[Tuple[int, int]] = set()

        # Behavior tracking
        self.phase = 'searching'
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0

    def _reset_task(self):
        """Reset wall-following specific state."""
        self.target_wall_segment = set()
        self.accessible_wall_cells = set()
        self.wall_boundaries = set()
        self.discovered_cells = set()
        self.phase = 'searching'
        self.wall_locked = False
        self.wall_contact_steps = 0
        self.last_distance = float('inf')
        self.no_new_discovery_steps = 0

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
        Select which wall cell to target. Prioritize walls adjacent to the agent,
        then nearest visible walls.
        """
        # First, check for walls directly adjacent to the agent
        adjacent_walls = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            wx, wy = pos[0] + dx, pos[1] + dy
            if 0 <= wx < global_map.shape[1] and 0 <= wy < global_map.shape[0]:
                if global_map[wy, wx] == TileType.WALL:
                    adjacent_walls.append((wx, wy))

        # If we have adjacent walls, pick one (prefer the one we're facing)
        if adjacent_walls:
            facing_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
            dx, dy = facing_dirs[facing]
            facing_wall = (pos[0] + dx, pos[1] + dy)

            if facing_wall in adjacent_walls:
                return facing_wall
            else:
                return adjacent_walls[0]

        # Otherwise, find all visible walls
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
        Trace a SINGLE continuous wall segment closest to the agent.
        Stop at corners, T-junctions, or gaps.
        """
        wx, wy = start_pos
        segment = {start_pos}

        # Determine if this is part of a vertical or horizontal wall
        # Check immediate neighbors in true map
        has_north = (wy > 0 and true_map[wy-1, wx] == TileType.WALL)
        has_south = (wy < true_map.shape[0]-1 and true_map[wy+1, wx] == TileType.WALL)
        has_west = (wx > 0 and true_map[wy, wx-1] == TileType.WALL)
        has_east = (wx < true_map.shape[1]-1 and true_map[wy, wx+1] == TileType.WALL)

        # Count connections
        vertical_connections = int(has_north) + int(has_south)
        horizontal_connections = int(has_west) + int(has_east)

        # If this is a corner or junction (3+ connections), just return this single cell
        total_connections = vertical_connections + horizontal_connections
        if total_connections >= 3:
            return segment

        # Determine primary direction based on connections
        is_vertical = vertical_connections > horizontal_connections

        # If equal, choose based on agent position
        if vertical_connections == horizontal_connections:
            dx = abs(agent_pos[0] - wx)
            dy = abs(agent_pos[1] - wy)
            is_vertical = dy >= dx

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

    def _update_discoveries(self, pos: Tuple[int, int], global_map):
        """Update discovered cells based on current position and vision."""
        for wx, wy in self.accessible_wall_cells:
            if (wx, wy) not in self.wall_boundaries:
                if global_map[wy, wx] != TileType.UNKNOWN:
                    self.discovered_cells.add((wx, wy))

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """Compute wall-following specific reward."""
        reward = -0.01  # Small step penalty

        pos = tuple(obs['positions'][0])
        facing = obs['facings'][0]
        global_map = obs['global_map']
        true_map = self.env.true_map

        # Phase: SEARCHING - Find a wall to follow
        if self.phase == 'searching' and not self.wall_locked:
            target_wall = self._select_target_wall(pos, facing, global_map)

            if target_wall:
                # Lock onto this SINGLE wall segment
                self.target_wall_segment = self._trace_single_wall_segment(
                    target_wall, pos, true_map, global_map
                )
                self.accessible_wall_cells = self._find_accessible_wall_cells(
                    self.target_wall_segment, pos, true_map
                )
                self.wall_boundaries = self._find_wall_boundaries(self.accessible_wall_cells)
                self.wall_locked = True
                self.phase = 'approaching'
                reward += 5.0
                #print(f"\nLocked onto single wall with {len(self.target_wall_segment)} cells")
                #print(f"Accessible: {len(self.accessible_wall_cells)} cells")
                #print(f"Boundaries: {len(self.wall_boundaries)} cells")

        # Phase: APPROACHING - Get to the wall
        if self.phase == 'approaching':
            dist = self._distance_to_wall(pos)

            if self._is_adjacent_to_wall(pos):
                self.phase = 'following'
                self.wall_contact_steps = 1
                reward += 15.0
                #print(f"\nReached the wall! Starting to follow...")
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
                    cells_to_discover = len(self.accessible_wall_cells) - len(self.wall_boundaries)
                    # if cells_to_discover > 0:
                        #print(f"Discovered: {new_discovered}/{cells_to_discover} wall cells")
                else:
                    self.no_new_discovery_steps += 1
            else:
                reward -= 3.0
                self.wall_contact_steps = 0

        # Collision penalty
        if base_reward < -0.5:
            reward -= 5.0

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if wall-following task is complete."""
        if self.phase == 'searching' and self.task_step > 150:
            return TaskStatus.FAILURE

        if self.wall_locked and self.accessible_wall_cells:
            self._update_discoveries(tuple(obs['positions'][0]), obs['global_map'])

            cells_to_discover = self.accessible_wall_cells - self.wall_boundaries

            if cells_to_discover:
                coverage = len(self.discovered_cells) / len(cells_to_discover)

                # if self.task_step % 10 == 0:
                    #print(f"Coverage: {coverage:.1%} ({len(self.discovered_cells)}/{len(cells_to_discover)} cells)")

                if self.phase in ['following', 'approaching']:
                    if len(self.discovered_cells) >= len(cells_to_discover):
                        #print(f"\nWall fully explored! Discovered all {len(self.discovered_cells)} cells")
                        return TaskStatus.SUCCESS

                    if coverage >= 0.95:
                        #print(f"\nWall exploration complete! Coverage: {coverage:.1%}")
                        return TaskStatus.SUCCESS

                    if self.no_new_discovery_steps > 30 and coverage >= 0.85:
                        #print(f"\nWall exploration complete (no new discoveries)! Coverage: {coverage:.1%}")
                        return TaskStatus.SUCCESS
            else:
                #print(f"\nWall segment complete (no cells to discover)")
                return TaskStatus.SUCCESS

        if self.task_step > 500:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def get_info(self) -> Dict:
        """Get additional task-specific information."""
        coverage = 0.0
        cells_to_discover = self.accessible_wall_cells - self.wall_boundaries
        if cells_to_discover:
            coverage = len(self.discovered_cells) / len(cells_to_discover)

        info = {
            'phase': self.phase,
            'wall_locked': self.wall_locked,
            'wall_contact_steps': self.wall_contact_steps,
            'wall_coverage': coverage,
            'discovered_accessible': len(self.discovered_cells),
            'total_accessible': len(self.accessible_wall_cells),
            'total_wall_cells': len(self.target_wall_segment),
        }
        return info

    def render(self):
        """Override render to highlight target wall with proper coloring."""
        super().render()

        if hasattr(self, 'env') and self.env.screen is not None and self.target_wall_segment:
            import pygame
            from environments.base.constants import TILE_SIZE

            # Color boundary cells in red
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