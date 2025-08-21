"""
curriculum_wrapper.py - Progressive curriculum learning wrapper for SLAM environment
Reveals most of the map initially, keeping only a configurable square hidden.
The hidden square is placed randomly on the map.
"""

import numpy as np
import gymnasium as gym
import random
from typing import Tuple, Dict, Any, Optional


class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning by initially revealing parts of the map.

    The wrapper starts with most of the map visible, keeping only a small square hidden.
    This square is placed randomly on the map and gradually increases in size as training progresses.
    """

    def __init__(self, env, hidden_size: int = 8, random_position: bool = True,
                 fixed_position: Optional[Tuple[int, int]] = None):
        """
        Initialize the curriculum wrapper.

        Args:
            env: Base SLAM environment (must be 32x32)
            hidden_size: Size of the initially hidden square (e.g., 8 for 8x8)
            random_position: If True, place hidden square randomly. If False, use center or fixed_position
            fixed_position: If provided, use this as the top-left corner of the hidden square (x, y)
        """
        super().__init__(env)
        self.hidden_size = hidden_size
        self.map_size = 32  # Fixed map size
        self.random_position = random_position
        self.fixed_position = fixed_position

        # Calculate adaptive parameters based on hidden area
        self.hidden_cells = hidden_size * hidden_size

        # Adaptive max_steps: 500 steps per 100 cells
        self.adaptive_max_steps = int((self.hidden_cells / 100) * 500)
        # Minimum 200 steps even for small areas
        self.adaptive_max_steps = max(200, self.adaptive_max_steps)

        # Adaptive completion bonus: hidden_size^2 / 2
        self.adaptive_completion_bonus = self.hidden_cells / 2.0

        # Override environment parameters
        self.env.max_steps = self.adaptive_max_steps
        self.env.completion_bonus = self.adaptive_completion_bonus

        # Initialize hidden square boundaries (will be set in reset)
        self.hidden_min_x = 0
        self.hidden_max_x = 0
        self.hidden_min_y = 0
        self.hidden_max_y = 0

        print(f"CurriculumWrapper initialized:")
        print(f"  Hidden area: {hidden_size}x{hidden_size} ({self.hidden_cells} cells)")
        print(f"  Max steps: {self.adaptive_max_steps}")
        print(f"  Completion bonus: {self.adaptive_completion_bonus:.1f}")
        print(f"  Random position: {self.random_position}")

    def _choose_hidden_position(self) -> Tuple[int, int]:
        """
        Choose the position for the hidden square.

        Returns:
            Tuple of (min_x, min_y) for the top-left corner of the hidden square
        """
        if self.fixed_position is not None:
            # Use the provided fixed position
            min_x, min_y = self.fixed_position
        elif self.random_position:
            # Choose a random position that keeps the square within bounds
            max_x_start = self.map_size - self.hidden_size
            max_y_start = self.map_size - self.hidden_size

            # Ensure we don't go out of bounds
            min_x = random.randint(0, max(0, max_x_start))
            min_y = random.randint(0, max(0, max_y_start))
        else:
            # Use center position (original behavior)
            center = self.map_size // 2
            half_hidden = self.hidden_size // 2
            min_x = center - half_hidden
            min_y = center - half_hidden

        return min_x, min_y

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the environment and reveal most of the map.

        Returns:
            Tuple of (observations, info)
        """
        # First, do normal reset
        obs, info = self.env.reset(**kwargs)

        # Choose position for hidden square
        self.hidden_min_x, self.hidden_min_y = self._choose_hidden_position()
        self.hidden_max_x = self.hidden_min_x + self.hidden_size
        self.hidden_max_y = self.hidden_min_y + self.hidden_size

        # Ensure boundaries are within map
        self.hidden_max_x = min(self.hidden_max_x, self.map_size)
        self.hidden_max_y = min(self.hidden_max_y, self.map_size)

        # print(f"  Hidden square: [{self.hidden_min_x}:{self.hidden_max_x}, {self.hidden_min_y}:{self.hidden_max_y}]")

        # Reveal everything except the hidden square
        revealed_count = 0
        for y in range(self.map_size):
            for x in range(self.map_size):
                # Check if this cell is outside the hidden square
                if not (self.hidden_min_x <= x < self.hidden_max_x and
                       self.hidden_min_y <= y < self.hidden_max_y):
                    # Reveal this cell by copying from true map to global map
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        self.env.global_map[y, x] = self.env.true_map[y, x]
                        revealed_count += 1

        # Find a suitable starting position inside the hidden area
        entry_placed = False
        possible_positions = []

        # Collect all valid positions in hidden area
        for y in range(self.hidden_min_y, self.hidden_max_y):
            for x in range(self.hidden_min_x, self.hidden_max_x):
                # Check if it's a walkable tile (free space or open door)
                if self.env.true_map[y, x] in [0, 3]:  # FREE_SPACE=0, DOOR_OPEN=3
                    possible_positions.append((x, y))

        # Place drone at a random valid position in hidden area
        if possible_positions:
            x, y = random.choice(possible_positions)
            self.env.drones[0].pos = (x, y)
            entry_placed = True
        else:
            # Fallback: place at center of hidden area if no valid positions
            x = (self.hidden_min_x + self.hidden_max_x) // 2
            y = (self.hidden_min_y + self.hidden_max_y) // 2
            self.env.drones[0].pos = (x, y)
            print(f"Warning: No valid positions in hidden area, placing at ({x}, {y})")

        # Update reachable mask to only count cells in the hidden square
        self.env.reachable_mask = np.zeros_like(self.env.reachable_mask, dtype=bool)

        # Mark hidden cells as reachable if they're explorable
        for y in range(self.hidden_min_y, self.hidden_max_y):
            for x in range(self.hidden_min_x, self.hidden_max_x):
                # Include all tiles in hidden area (even walls need to be discovered)
                self.env.reachable_mask[y, x] = True

        # Update total reachable count
        self.env.total_reachable = np.sum(self.env.reachable_mask)

        # Update observations after modifications
        obs = self.env._get_observations()
        info = self.env._get_info()

        # Add curriculum-specific info
        info['curriculum_stage'] = {
            'hidden_size': self.hidden_size,
            'hidden_cells': self.hidden_cells,
            'revealed_initially': revealed_count,
            'max_steps': self.adaptive_max_steps,
            'completion_bonus': self.adaptive_completion_bonus,
            'drone_start_pos': self.env.drones[0].pos,
            'hidden_position': (self.hidden_min_x, self.hidden_min_y),
            'hidden_bounds': {
                'x': (self.hidden_min_x, self.hidden_max_x),
                'y': (self.hidden_min_y, self.hidden_max_y)
            }
        }

        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute a step in the wrapped environment.

        Args:
            action: Action to execute

        Returns:
            Standard gym step return tuple
        """
        # Pass action directly to the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add curriculum info to every step
        info['hidden_cells'] = self.hidden_cells
        info['hidden_size'] = self.hidden_size
        info['hidden_position'] = (self.hidden_min_x, self.hidden_min_y)

        return obs, reward, terminated, truncated, info

    def render(self):
        """Forward render call to the base environment."""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        """Forward close call to the base environment."""
        return self.env.close()

    def get_curriculum_info(self) -> Dict[str, Any]:
        """
        Get current curriculum configuration.

        Returns:
            Dictionary with curriculum parameters
        """
        return {
            'hidden_size': self.hidden_size,
            'hidden_cells': self.hidden_cells,
            'max_steps': self.adaptive_max_steps,
            'completion_bonus': self.adaptive_completion_bonus,
            'hidden_bounds': {
                'x_min': self.hidden_min_x,
                'x_max': self.hidden_max_x,
                'y_min': self.hidden_min_y,
                'y_max': self.hidden_max_y
            },
            'random_position': self.random_position
        }