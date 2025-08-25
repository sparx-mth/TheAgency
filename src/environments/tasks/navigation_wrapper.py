"""
Navigation task wrapper for training an agent to reach discovered locations.

The agent explores freely for 100 steps, then receives a destination goal
from previously discovered locations and must navigate to it.
"""

import numpy as np
from typing import Optional, Tuple
import pygame

from environments.tasks.base_task_wrapper import BaseTaskWrapper, TaskStatus
from environments.base.constants import TileType, TILE_SIZE, FPS


class NavigationWrapper(BaseTaskWrapper):
    """
    Environment wrapper for navigation to known locations.

    Task flow:
    1. First 100 steps: Free exploration (no goal, no task rewards)
    2. After 100 steps: Goal assigned from discovered locations
    3. Success: Reaching the goal position
    4. Failure: Exceeding max steps after goal assignment
    """

    def __init__(self, env_config: dict = None, max_steps_to_goal: int = 200):
        """
        Initialize navigation wrapper.

        Args:
            env_config: Configuration for base environment
            max_steps_to_goal: Maximum steps allowed after goal is set
        """
        super().__init__(env_config)

        self.exploration_steps = 20  # Steps before goal assignment
        self.max_steps_to_goal = max_steps_to_goal

        # Task-specific state
        self.goal_position: Optional[Tuple[int, int]] = None
        self.discovered_positions = set()
        self.steps_since_goal = 0
        self.prev_distance_to_goal = None  # Track distance for reward shaping

    def _reset_task(self):
        """Reset navigation task state."""
        self.goal_position = None
        self.discovered_positions = set()
        self.steps_since_goal = 0
        self.prev_distance_to_goal = None

    def step(self, action):
        """Execute action with navigation-specific logic."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Track discovered positions during exploration
        if self.task_step <= self.exploration_steps:
            global_map = obs['global_map']
            for y in range(global_map.shape[0]):
                for x in range(global_map.shape[1]):
                    if global_map[y, x] != TileType.UNKNOWN:
                        self.discovered_positions.add((x, y))

        # Assign goal after exploration phase
        if self.task_step == self.exploration_steps and self.goal_position is None:
            self._assign_goal()

        # Add goal info to observation
        info['goal_position'] = self.goal_position
        info['steps_to_goal'] = self.steps_since_goal if self.goal_position else 0

        return obs, reward, terminated, truncated, info

    def _assign_goal(self):
        """Assign a goal from discovered free spaces."""
        # Filter for free spaces only
        free_spaces = []
        global_map = self.env.global_map

        for pos in self.discovered_positions:
            x, y = pos
            if global_map[y, x] in [TileType.FREE_SPACE, TileType.DOOR_OPEN]:
                # Don't set goal at current position
                current_pos = self.env.drones[0].pos
                if pos != current_pos:
                    free_spaces.append(pos)

        if free_spaces:
            # Randomly select a goal from discovered free spaces
            self.goal_position = free_spaces[np.random.randint(len(free_spaces))]

            # Initialize distance tracking for reward shaping
            current_pos = self.env.drones[0].pos
            self.prev_distance_to_goal = abs(current_pos[0] - self.goal_position[0]) + \
                                        abs(current_pos[1] - self.goal_position[1])

            print(f"Goal assigned at position: {self.goal_position}")
            print(f"Initial distance to goal: {self.prev_distance_to_goal}")
        else:
            print("Warning: No valid free spaces discovered for goal assignment")

    def _compute_task_reward(self, obs, action, base_reward) -> float:
        """
        Compute navigation reward with distance-based shaping.

        Before goal: No task reward (exploration phase)
        After goal:
            - Positive reward for getting closer
            - Negative reward for getting farther
            - Large bonus for reaching goal
        """
        # No task reward during exploration phase
        if self.goal_position is None:
            return 0.0

        self.steps_since_goal += 1

        # Current position - obs is still multi-agent format here
        current_pos = tuple(obs['positions'][0])  # Get first drone's position

        # Check if goal reached
        if current_pos == self.goal_position:
            return 200.0  # Large success reward for reaching goal

        # Calculate Manhattan distance to goal
        current_distance = abs(current_pos[0] - self.goal_position[0]) + \
                          abs(current_pos[1] - self.goal_position[1])

        # Distance-based reward shaping
        reward = 0.0

        if self.prev_distance_to_goal is not None:
            # Calculate change in distance
            distance_change = self.prev_distance_to_goal - current_distance

            if distance_change > 0:
                # Got closer to goal - positive reward
                reward = 1.0 * distance_change  # +1.0 per unit closer
            elif distance_change < 0:
                # Got farther from goal - negative reward
                reward = 0.5 * distance_change  # -0.5 per unit farther
            else:
                # Same distance (turned or stayed) - small penalty
                reward = -0.1

        # Update previous distance for next step
        self.prev_distance_to_goal = current_distance

        # Add small time penalty to encourage efficiency
        reward -= 0.01

        return reward

    def _check_task_status(self, obs, action) -> TaskStatus:
        """Check if navigation task is complete."""
        # No task status during exploration
        if self.goal_position is None:
            return TaskStatus.IN_PROGRESS

        current_pos = tuple(obs['positions'][0])  # Get first drone's position

        # Success: Reached goal
        if current_pos == self.goal_position:
            return TaskStatus.SUCCESS

        # Failure: Too many steps after goal assignment
        if self.steps_since_goal > self.max_steps_to_goal:
            return TaskStatus.FAILURE

        return TaskStatus.IN_PROGRESS

    def render(self) -> Optional[np.ndarray]:
        """Render with goal visualization."""
        # Call base environment render
        rgb_array = self.env.render()

        # Add goal marker if pygame is initialized and goal exists
        if self.goal_position and self.env.screen:
            # Draw goal on both maps (true and observed)
            for offset_x in [0, self.env.width * TILE_SIZE + 50]:
                goal_x = offset_x + self.goal_position[0] * TILE_SIZE + TILE_SIZE // 2
                goal_y = self.goal_position[1] * TILE_SIZE + TILE_SIZE // 2

                # Draw goal as a star/target
                # Outer circle (red)
                pygame.draw.circle(self.env.screen, (255, 0, 0),
                                 (goal_x, goal_y), 8, 2)
                # Inner circle (white)
                pygame.draw.circle(self.env.screen, (255, 255, 255),
                                 (goal_x, goal_y), 4)

                # Draw "GOAL" text above
                if self.env.font:
                    goal_text = self.env.font.render("GOAL", True, (255, 0, 0))
                    self.env.screen.blit(goal_text,
                                       (goal_x - 15, goal_y - 20))

            # Update display
            pygame.display.flip()

            # Return updated RGB array if needed
            if self.env.render_mode == 'rgb_array':
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.env.screen)),
                    axes=(1, 0, 2)
                )

        return rgb_array


# Example usage
if __name__ == "__main__":
    # Create navigation environment
    env_config = {
        'width': 20,
        'height': 20,
        'num_agents': 1,
        'max_steps': 500,
        'render_mode': 'human'
    }

    env = NavigationWrapper(env_config)

    # Run a simple test episode
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_count += 1

        # Print when goal is assigned
        if step_count == 100:
            print(f"Goal assigned: {info.get('goal_position')}")

        # Print progress
        if step_count % 50 == 0:
            print(f"Step {step_count}, Task Status: {info.get('task_status')}, Reward: {reward:.2f}")

        # Render
        env.render()

        if done:
            print(f"Episode finished. Status: {info.get('task_status')}")
            break

    env.close()