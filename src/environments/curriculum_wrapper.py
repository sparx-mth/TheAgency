"""
curriculum_wrapper.py - Progressive curriculum learning wrapper for SLAM environment
Reveals most of the map initially, keeping only a configurable square hidden.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any


class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning by initially revealing parts of the map.

    The wrapper starts with most of the map visible, keeping only a small square hidden.
    This square gradually increases in size as training progresses.
    """

    def __init__(self, env, hidden_size: int = 8):
        """
        Initialize the curriculum wrapper.

        Args:
            env: Base SLAM environment (must be 32x32)
            hidden_size: Size of the initially hidden square (e.g., 8 for 8x8)
        """
        super().__init__(env)
        self.hidden_size = hidden_size
        self.map_size = 32  # Fixed map size

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

        # Calculate hidden square boundaries (centered)
        center = self.map_size // 2
        half_hidden = self.hidden_size // 2
        self.hidden_min = center - half_hidden
        self.hidden_max = center + half_hidden

        print(f"CurriculumWrapper initialized:")
        print(f"  Hidden area: {hidden_size}x{hidden_size} ({self.hidden_cells} cells)")
        print(f"  Max steps: {self.adaptive_max_steps}")
        print(f"  Completion bonus: {self.adaptive_completion_bonus:.1f}")
        print(f"  Hidden square: [{self.hidden_min}:{self.hidden_max}, {self.hidden_min}:{self.hidden_max}]")

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the environment and reveal most of the map.

        Returns:
            Tuple of (observations, info)
        """
        # First, do normal reset
        obs, info = self.env.reset(**kwargs)

        # Reveal everything except the hidden square
        revealed_count = 0
        for y in range(self.map_size):
            for x in range(self.map_size):
                # Check if this cell is outside the hidden square
                if not (self.hidden_min <= x < self.hidden_max and
                       self.hidden_min <= y < self.hidden_max):
                    # Reveal this cell by copying from true map to global map
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        self.env.global_map[y, x] = self.env.true_map[y, x]
                        revealed_count += 1

        # Find a suitable starting position inside the hidden area
        entry_placed = False
        possible_positions = []

        # Collect all valid positions in hidden area
        for y in range(self.hidden_min, self.hidden_max):
            for x in range(self.hidden_min, self.hidden_max):
                # Check if it's a walkable tile (free space or open door)
                if self.env.true_map[y, x] in [0, 3]:  # FREE_SPACE=0, DOOR_OPEN=3
                    possible_positions.append((x, y))

        # Place drone at a random valid position in hidden area
        if possible_positions:
            import random
            x, y = random.choice(possible_positions)
            self.env.drones[0].pos = (x, y)
            entry_placed = True
        else:
            # Fallback: place at center of hidden area if no valid positions
            x = (self.hidden_min + self.hidden_max) // 2
            y = (self.hidden_min + self.hidden_max) // 2
            self.env.drones[0].pos = (x, y)
            print(f"Warning: No valid positions in hidden area, placing at ({x}, {y})")

        # Update reachable mask to only count cells in the hidden square
        self.env.reachable_mask = np.zeros_like(self.env.reachable_mask, dtype=bool)

        # Mark hidden cells as reachable if they're explorable
        for y in range(self.hidden_min, self.hidden_max):
            for x in range(self.hidden_min, self.hidden_max):
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
            'drone_start_pos': self.env.drones[0].pos
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

        return obs, reward, terminated, truncated, info

    def render(self):
        """Forward render call to the base environment."""
        return self.env.render()

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
                'min': self.hidden_min,
                'max': self.hidden_max
            }
        }
