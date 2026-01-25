"""
Curriculum learning wrapper that reveals most of the map initially.
Only a configurable square remains hidden for the agent to explore.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional


class CurriculumWrapper(gym.Wrapper):
    """
    Reveals most of the map, keeping only a small square hidden.

    Useful for curriculum learning: start with small hidden areas,
    gradually increase as agents improve.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_size: int = 8,
        random_position: bool = False,
        fixed_position: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            env: Base SLAM environment
            hidden_size: Size of hidden square (e.g., 8 for 8x8)
            random_position: If True, randomize hidden square position each reset
            fixed_position: Fixed (x, y) top-left corner for hidden square
        """
        super().__init__(env)

        self.hidden_size = hidden_size
        self.map_width = env.width
        self.map_height = env.height
        self.random_position = random_position
        self.fixed_position = fixed_position

        # Hidden area stats
        self.hidden_cells = hidden_size * hidden_size

        # Adaptive parameters
        self.adaptive_max_steps = max(200, int((self.hidden_cells / 100) * 500))
        self.adaptive_completion_bonus = self.hidden_cells / 2.0

        # Hidden square bounds (set in reset)
        self.hidden_x = 0
        self.hidden_y = 0

    def _choose_hidden_position(self) -> Tuple[int, int]:
        """Choose top-left corner of hidden square."""
        if self.fixed_position is not None:
            return self.fixed_position

        if self.random_position:
            max_x = max(0, self.map_width - self.hidden_size)
            max_y = max(0, self.map_height - self.hidden_size)
            return (
                np.random.randint(0, max_x + 1),
                np.random.randint(0, max_y + 1)
            )

        # Center position
        return (
            (self.map_width - self.hidden_size) // 2,
            (self.map_height - self.hidden_size) // 2
        )

    def _is_hidden(self, x: int, y: int) -> bool:
        """Check if cell is in hidden area."""
        return (self.hidden_x <= x < self.hidden_x + self.hidden_size and
                self.hidden_y <= y < self.hidden_y + self.hidden_size)

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)

        # Override max steps and completion bonus
        self.env.max_steps = self.adaptive_max_steps
        if hasattr(self.env, 'rewards'):
            self.env.rewards['completion'] = self.adaptive_completion_bonus

        # Choose hidden position
        self.hidden_x, self.hidden_y = self._choose_hidden_position()

        # Reveal everything except hidden square
        from simulator.constants import TileType
        for y in range(self.map_height):
            for x in range(self.map_width):
                if not self._is_hidden(x, y):
                    self.env.global_map[y, x] = self.env.true_map[y, x]

        # Find valid start position in hidden area
        valid_positions = []
        for y in range(self.hidden_y, min(self.hidden_y + self.hidden_size, self.map_height)):
            for x in range(self.hidden_x, min(self.hidden_x + self.hidden_size, self.map_width)):
                if self.env.true_map[y, x] in (TileType.FREE_SPACE, TileType.ENTRY_POINT, TileType.DOOR_OPEN):
                    valid_positions.append((x, y))

        # Place first drone in hidden area
        if valid_positions and len(self.env.drones) > 0:
            pos = valid_positions[np.random.randint(len(valid_positions))]
            self.env.drones[0].pos = pos

        # Update reachable mask to only count hidden cells
        self.env.reachable_mask = np.zeros((self.map_height, self.map_width), dtype=bool)
        for y in range(self.hidden_y, min(self.hidden_y + self.hidden_size, self.map_height)):
            for x in range(self.hidden_x, min(self.hidden_x + self.hidden_size, self.map_width)):
                self.env.reachable_mask[y, x] = True

        self.env.total_reachable = int(np.sum(self.env.reachable_mask))

        # Get updated observations
        obs = self.env._get_obs()
        info = self.env._get_info()
        info['curriculum'] = {
            'hidden_size': self.hidden_size,
            'hidden_position': (self.hidden_x, self.hidden_y),
            'max_steps': self.adaptive_max_steps,
        }

        return obs, info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['hidden_size'] = self.hidden_size
        return obs, reward, terminated, truncated, info

    def set_hidden_size(self, size: int):
        """Update hidden size for curriculum progression."""
        self.hidden_size = size
        self.hidden_cells = size * size
        self.adaptive_max_steps = max(200, int((self.hidden_cells / 100) * 500))
        self.adaptive_completion_bonus = self.hidden_cells / 2.0