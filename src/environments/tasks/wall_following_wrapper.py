"""
Wall Following Wrapper for single-agent training.

This wrapper trains an agent to:
1. Find the first visible wall (or nearest if multiple, or facing if equidistant)
2. Approach and stick to it
3. Follow along the entire wall from end to end
4. Stop once that single wall is fully explored from the agent's accessible side
"""

from typing import Dict, Tuple, Set, Optional, List
import numpy as np
from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, DIRECTION_DELTAS


class WallFollowingWrapper(BaseTaskWrapper):
    """
    Environment wrapper for training wall-following behavior.

    The agent must:
    1. Find the first visible wall (prioritized by: only wall > nearest > facing)
    2. Approach it and get adjacent
    3. Follow it to discover the entire wall segment from their side
    4. Episode ends when that single wall is fully explored
    """

    def __init__(self, env_config: Dict = None):
        super().__init__(env_config)

        # Wall tracking
        self.target_wall_segment: Set[Tuple[int, int]] = set()
        self.accessible_wall_cells: Set[Tuple[int, int]] = set()  # Wall cells accessible from agent's room
        self.wall_boundaries: Set[Tuple[int, int]] = set()  # Boundary cells (red)
        self.discovered_cells: Set[Tuple[int, int]] = set()

        # Behavior tracking
        self.phase = 'searching'  # 'searching', 'approaching', 'following'
        self.wall_locked = False  # Once we lock onto a wall, we stick with it
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
            # Check which wall we're facing
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

        # Calculate distances to all visible walls
        wall_distances = {}
        for wx, wy in visible_walls:
            dist = abs(wx - pos[0]) + abs(wy - pos[1])
            if dist not in wall_distances:
                wall_distances[dist] = []
            wall_distances[dist].append((wx, wy))

        # Get the minimum distance
        min_dist = min(wall_distances.keys())
        nearest_walls = wall_distances[min_dist]

        # If only one wall at minimum distance, select it
        if len(nearest_walls) == 1:
            return nearest_walls[0]

        # Multiple walls at same distance - prefer one in front
        facing_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        dx, dy = facing_dirs[facing]

        best_wall = None
        best_score = -float('inf')

        for wx, wy in nearest_walls:
            # Vector from agent to wall
            to_wall_x = wx - pos[0]
            to_wall_y = wy - pos[1]

            # Dot product gives alignment score
            score = to_wall_x * dx + to_wall_y * dy

            if score > best_score:
                best_score = score
                best_wall = (wx, wy)

        return best_wall if best_wall else nearest_walls[0]

    def _trace_wall_segment(self, start_pos: Tuple[int, int], agent_pos: Tuple[int, int], true_map) -> Set[Tuple[int, int]]:
        """
        Trace a single straight wall line (either vertical or horizontal).
        A vertical wall has same X coordinate, horizontal has same Y coordinate.
        """
        segment = {start_pos}
        wx, wy = start_pos

        # Determine if this wall extends vertically or horizontally
        has_vertical_neighbor = False
        has_horizontal_neighbor = False

        # Check for vertical neighbors (same X, different Y)
        if 0 <= wy - 1 < true_map.shape[0] and true_map[wy - 1, wx] == TileType.WALL:
            has_vertical_neighbor = True
        if 0 <= wy + 1 < true_map.shape[0] and true_map[wy + 1, wx] == TileType.WALL:
            has_vertical_neighbor = True

        # Check for horizontal neighbors (same Y, different X)
        if 0 <= wx - 1 < true_map.shape[1] and true_map[wy, wx - 1] == TileType.WALL:
            has_horizontal_neighbor = True
        if 0 <= wx + 1 < true_map.shape[1] and true_map[wy, wx + 1] == TileType.WALL:
            has_horizontal_neighbor = True

        # Determine wall orientation
        # If it has both, prefer the one that creates a longer continuous line
        if has_vertical_neighbor and has_horizontal_neighbor:
            # Count length in each direction
            vertical_length = 1
            horizontal_length = 1

            # Count vertical extent
            for dy in range(1, true_map.shape[0]):
                if 0 <= wy + dy < true_map.shape[0] and true_map[wy + dy, wx] == TileType.WALL:
                    vertical_length += 1
                else:
                    break
            for dy in range(1, true_map.shape[0]):
                if 0 <= wy - dy < true_map.shape[0] and true_map[wy - dy, wx] == TileType.WALL:
                    vertical_length += 1
                else:
                    break

            # Count horizontal extent
            for dx in range(1, true_map.shape[1]):
                if 0 <= wx + dx < true_map.shape[1] and true_map[wy, wx + dx] == TileType.WALL:
                    horizontal_length += 1
                else:
                    break
            for dx in range(1, true_map.shape[1]):
                if 0 <= wx - dx < true_map.shape[1] and true_map[wy, wx - dx] == TileType.WALL:
                    horizontal_length += 1
                else:
                    break

            # Choose the longer direction
            is_vertical = vertical_length > horizontal_length
        else:
            is_vertical = has_vertical_neighbor

        # Now trace the wall in the determined direction
        if is_vertical:
            # Trace vertical wall (same X, changing Y)
            # Go up
            for y in range(wy - 1, -1, -1):
                if true_map[y, wx] == TileType.WALL:
                    segment.add((wx, y))
                else:
                    break
            # Go down
            for y in range(wy + 1, true_map.shape[0]):
                if true_map[y, wx] == TileType.WALL:
                    segment.add((wx, y))
                else:
                    break
        else:
            # Trace horizontal wall (same Y, changing X)
            # Go left
            for x in range(wx - 1, -1, -1):
                if true_map[wy, x] == TileType.WALL:
                    segment.add((x, wy))
                else:
                    break
            # Go right
            for x in range(wx + 1, true_map.shape[1]):
                if true_map[wy, x] == TileType.WALL:
                    segment.add((x, wy))
                else:
                    break

        return segment

    def _find_accessible_wall_cells(self, wall_segment: Set[Tuple[int, int]], agent_pos: Tuple[int, int], true_map) -> Set[Tuple[int, int]]:
        """
        Find which wall cells from the segment are accessible from the agent's current position.
        This determines the portion of the wall the agent should explore.
        """
        # First, find all free spaces reachable from agent position
        reachable_spaces = set()
        to_visit = [agent_pos]
        visited = set()

        while to_visit:
            x, y = to_visit.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Check if this is a free space
            if 0 <= x < true_map.shape[1] and 0 <= y < true_map.shape[0]:
                if true_map[y, x] in [TileType.FREE_SPACE, TileType.ENTRY_POINT, TileType.DOOR_OPEN]:
                    reachable_spaces.add((x, y))

                    # Continue BFS through free spaces
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in visited:
                            to_visit.append((nx, ny))

        # Find wall cells that are adjacent to reachable spaces
        adjacent_wall_cells = []
        for wx, wy in wall_segment:
            # Check if this wall cell is next to any reachable space
            is_accessible = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = wx + dx, wy + dy
                if (nx, ny) in reachable_spaces:
                    is_accessible = True
                    break
            if is_accessible:
                adjacent_wall_cells.append((wx, wy))

        if not adjacent_wall_cells:
            return set()

        # For a straight wall, we want the continuous portion that's accessible
        # Sort cells and find the continuous segment
        if len(adjacent_wall_cells) > 1:
            # Check if vertical (same X) or horizontal (same Y)
            first_cell = adjacent_wall_cells[0]
            is_vertical = all(cell[0] == first_cell[0] for cell in adjacent_wall_cells)

            if is_vertical:
                # Sort by Y coordinate
                adjacent_wall_cells.sort(key=lambda cell: cell[1])
            else:
                # Sort by X coordinate
                adjacent_wall_cells.sort(key=lambda cell: cell[0])

            # Find ALL continuous sequences (there might be gaps due to doors, etc.)
            sequences = []
            current_sequence = [adjacent_wall_cells[0]]

            for i in range(1, len(adjacent_wall_cells)):
                prev = adjacent_wall_cells[i-1]
                curr = adjacent_wall_cells[i]

                # Check if continuous (adjacent cells)
                if is_vertical:
                    is_continuous = (curr[1] - prev[1] == 1)
                else:
                    is_continuous = (curr[0] - prev[0] == 1)

                if is_continuous:
                    current_sequence.append(curr)
                else:
                    # Gap found - save current sequence and start new one
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = [curr]

            # Don't forget the last sequence
            if current_sequence:
                sequences.append(current_sequence)

            # Pick the longest continuous sequence, or the one closest to agent
            if len(sequences) == 1:
                return set(sequences[0])
            else:
                # Multiple sequences - pick the closest one to agent
                best_sequence = None
                best_distance = float('inf')

                for sequence in sequences:
                    # Calculate average distance to agent
                    avg_distance = sum(abs(cell[0] - agent_pos[0]) + abs(cell[1] - agent_pos[1])
                                     for cell in sequence) / len(sequence)
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_sequence = sequence

                return set(best_sequence) if best_sequence else set(adjacent_wall_cells)
        else:
            return set(adjacent_wall_cells)

    def _find_wall_boundaries(self, accessible_cells: Set[Tuple[int, int]], wall_segment: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Find the boundary cells (red cells) - these are the endpoints of the accessible portion.
        Boundaries are either:
        1. The first and last cells of the accessible portion
        2. Wall cells just beyond the accessible portion (if they exist)
        """
        if not accessible_cells:
            return set()

        boundaries = set()
        accessible_list = list(accessible_cells)

        if len(accessible_list) == 1:
            # Single cell - it's both start and end
            boundaries.add(accessible_list[0])
            return boundaries

        # Check if vertical or horizontal
        first_cell = accessible_list[0]
        is_vertical = all(cell[0] == first_cell[0] for cell in accessible_list)

        if is_vertical:
            # Vertical wall - sort by Y
            accessible_list.sort(key=lambda cell: cell[1])
            x = first_cell[0]
            min_y = accessible_list[0][1]
            max_y = accessible_list[-1][1]

            # Add the first and last accessible cells as boundaries
            boundaries.add((x, min_y))
            boundaries.add((x, max_y))

            # Also check if there are wall cells just beyond the accessible range
            # These would be the "true" boundaries blocking further exploration
            if (x, min_y - 1) in wall_segment and (x, min_y - 1) not in accessible_cells:
                boundaries.add((x, min_y - 1))
            if (x, max_y + 1) in wall_segment and (x, max_y + 1) not in accessible_cells:
                boundaries.add((x, max_y + 1))
        else:
            # Horizontal wall - sort by X
            accessible_list.sort(key=lambda cell: cell[0])
            y = first_cell[1]
            min_x = accessible_list[0][0]
            max_x = accessible_list[-1][0]

            # Add the first and last accessible cells as boundaries
            boundaries.add((min_x, y))
            boundaries.add((max_x, y))

            # Check for wall cells just beyond the accessible range
            if (min_x - 1, y) in wall_segment and (min_x - 1, y) not in accessible_cells:
                boundaries.add((min_x - 1, y))
            if (max_x + 1, y) in wall_segment and (max_x + 1, y) not in accessible_cells:
                boundaries.add((max_x + 1, y))

        return boundaries

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
        # Check which accessible wall cells have been discovered (visible in global map)
        # Only count cells between boundaries (orange cells, not red)
        for wx, wy in self.accessible_wall_cells:
            if (wx, wy) not in self.wall_boundaries:  # Don't need to discover boundary cells
                if global_map[wy, wx] != TileType.UNKNOWN:
                    self.discovered_cells.add((wx, wy))

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """
        Compute wall-following specific reward.
        """
        reward = -0.01  # Small step penalty

        pos = tuple(obs['positions'][0])
        facing = obs['facings'][0]
        global_map = obs['global_map']

        # Get true map from the base environment
        true_map = self.env.true_map

        # Phase: SEARCHING - Find a wall to follow
        if self.phase == 'searching' and not self.wall_locked:
            target_wall = self._select_target_wall(pos, facing, global_map)

            if target_wall:
                # Lock onto this wall segment using true map
                self.target_wall_segment = self._trace_wall_segment(target_wall, pos, true_map)
                # Find which cells are accessible from agent's room
                self.accessible_wall_cells = self._find_accessible_wall_cells(
                    self.target_wall_segment, pos, true_map
                )
                # Find boundary cells
                self.wall_boundaries = self._find_wall_boundaries(
                    self.accessible_wall_cells, self.target_wall_segment
                )
                self.wall_locked = True
                self.phase = 'approaching'
                reward += 5.0  # Bonus for finding and locking onto a wall
                print(f"\nLocked onto wall segment with {len(self.target_wall_segment)} total cells")
                print(f"Accessible from this room: {len(self.accessible_wall_cells)} cells")
                print(f"Boundaries: {len(self.wall_boundaries)} cells")

        # Phase: APPROACHING - Get to the wall
        if self.phase == 'approaching':
            dist = self._distance_to_wall(pos)

            if self._is_adjacent_to_wall(pos):
                # Reached the wall!
                self.phase = 'following'
                self.wall_contact_steps = 1
                reward += 15.0  # Big bonus for reaching wall
                print(f"\nReached the wall! Starting to follow...")
            else:
                # Reward getting closer
                if dist < self.last_distance:
                    reward += 2.0
                elif dist > self.last_distance:
                    reward -= 1.0

            self.last_distance = dist

        # Phase: FOLLOWING - Explore the entire wall from this side
        elif self.phase == 'following':
            adjacent = self._is_adjacent_to_wall(pos)

            # Update discoveries
            old_discovered = len(self.discovered_cells)
            self._update_discoveries(pos, global_map)
            new_discovered = len(self.discovered_cells)

            if adjacent:
                self.wall_contact_steps += 1
                reward += 1.5  # Reward for staying with wall

                # Reward new discoveries
                if new_discovered > old_discovered:
                    reward += (new_discovered - old_discovered) * 3.0
                    self.no_new_discovery_steps = 0
                    print(f"Discovered: {new_discovered}/{len(self.accessible_wall_cells)} accessible wall cells")
                else:
                    self.no_new_discovery_steps += 1

            else:
                # Penalty for leaving wall
                reward -= 3.0
                self.wall_contact_steps = 0

        # Collision penalty
        if base_reward < -0.5:  # Collision detected in base env
            reward -= 5.0

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if wall-following task is complete."""
        # Timeout for finding a wall
        if self.phase == 'searching' and self.task_step > 150:
            return TaskStatus.FAILURE

        if self.wall_locked and self.accessible_wall_cells:
            # Update discoveries
            global_map = obs['global_map']
            self._update_discoveries(tuple(obs['positions'][0]), global_map)

            # Calculate how many non-boundary cells need to be discovered
            cells_to_discover = self.accessible_wall_cells - self.wall_boundaries

            if cells_to_discover:
                # Calculate coverage of non-boundary accessible wall cells
                coverage = len(self.discovered_cells) / len(cells_to_discover)

                # Debug print to see what's happening
                if self.task_step % 10 == 0:  # Print every 10 steps
                    print(f"Coverage: {coverage:.1%} ({len(self.discovered_cells)}/{len(cells_to_discover)} non-boundary cells)")

                if self.phase == 'following' or self.phase == 'approaching':
                    # Success: discovered all non-boundary accessible wall cells
                    if len(self.discovered_cells) >= len(cells_to_discover):
                        print(f"\nWall fully explored! Discovered all {len(self.discovered_cells)} cells between boundaries")
                        return TaskStatus.SUCCESS

                    # Alternative: very high coverage
                    if coverage >= 0.95:
                        print(f"\nWall exploration complete! Coverage: {coverage:.1%}")
                        return TaskStatus.SUCCESS

                    # Also check if we haven't discovered anything new for a while with good coverage
                    if self.no_new_discovery_steps > 30 and coverage >= 0.85:
                        print(f"\nWall exploration complete (no new discoveries)! Coverage: {coverage:.1%}")
                        return TaskStatus.SUCCESS
            else:
                # Edge case: all accessible cells are boundaries
                print(f"\nWall segment has no cells to discover (all boundaries)")
                return TaskStatus.SUCCESS

        # General timeout
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
        }
        return info

    def render(self):
        """Override render to highlight target wall with proper coloring."""
        # Call parent render first
        super().render()

        # If we have pygame and a target wall, color it appropriately
        if hasattr(self, 'env') and self.env.screen is not None and self.target_wall_segment:
            import pygame
            from environments.base.constants import TILE_SIZE

            # First color the boundary cells in red (corners/ends of accessible portion)
            for wx, wy in self.wall_boundaries:
                rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.env.screen, (255, 0, 0), rect)

            # Then color accessible cells in orange (these need to be discovered)
            for wx, wy in self.accessible_wall_cells:
                if (wx, wy) not in self.wall_boundaries:  # Don't overwrite red boundaries
                    rect = pygame.Rect(wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                    pygame.draw.rect(self.env.screen, (255, 128, 0), rect)

            # Color discovered cells in green on the observed map (right side)
            offset_x = self.env.width * TILE_SIZE + 50
            for wx, wy in self.discovered_cells:
                rect = pygame.Rect(offset_x + wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.env.screen, (0, 255, 0), rect)

            pygame.display.flip()