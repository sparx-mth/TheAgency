"""
Manual testing script for task environments.
Use keyboard to control the agent and observe behavior.

Controls:
- W/↑: Move forward
- A/←: Turn left
- D/→: Turn right
- S/↓: Stay/No action
- Q: Quit
- R: Reset environment
"""

import pygame
import numpy as np
from typing import Any
from environments.tasks.wall_following_wrapper import WallFollowingWrapper
from environments.base.constants import Action


def print_info(obs: dict, reward: float, done: bool, info: dict, action_name: str):
    """Print comprehensive environment information."""
    print("\n" + "=" * 60)
    print(f"ACTION: {action_name}")
    print(f"Position: {obs['position']}")
    print(f"Facing: {obs['facing']} (0=N, 1=E, 2=S, 3=W)")
    print(f"Reward: {reward:.2f}")
    print(f"Done: {done}")
    print(f"Task Status: {info.get('task_status', 'N/A')}")
    print(f"Step: {info.get('task_step', 0)}")

    # Task-specific info if available
    if hasattr(env, 'get_info'):
        task_info = env.get_info()
        print(f"\nTask Info:")
        for key, value in task_info.items():
            print(f"  {key}: {value}")

    # Map stats
    global_map = obs['global_map']
    visible_cells = np.sum(global_map != -1)  # -1 is UNKNOWN
    wall_cells = np.sum(global_map == 1)  # 1 is WALL
    print(f"\nMap Stats:")
    print(f"  Visible cells: {visible_cells}")
    print(f"  Wall cells found: {wall_cells}")


def run_manual_test(env_class: Any, env_config: dict = None):
    """
    Run manual testing for any task environment.

    Args:
        env_class: Environment class to test
        env_config: Configuration for the environment
    """
    global env  # Make env accessible to print_info

    # Default config with small FOV camera
    default_config = {
        'width': 32,
        'height': 32,
        'num_agents': 1,
        'max_steps': 1000,
        'randomize': True,
        'render_mode': 'human',
        'default_sensor_params': {
            'max_range': 5,  # Short range
            'fov_deg': 30,  # Narrow FOV
            'num_rays': 10
        }
    }

    if env_config:
        default_config.update(env_config)

    # Create environment
    env = env_class(default_config)

    # Reset and get initial observation
    obs, info = env.reset()
    env.render()

    print("\n" + "=" * 60)
    print("ENVIRONMENT MANUAL TESTER")
    print("=" * 60)
    print("Controls:")
    print("  W/↑: Forward")
    print("  A/←: Turn Left")
    print("  D/→: Turn Right")
    print("  S/↓: Stay")
    print("  R: Reset")
    print("  Q: Quit")
    print("=" * 60)

    print_info(obs, 0, False, info, "INITIAL")

    # Main game loop
    clock = pygame.time.Clock()
    running = True

    while running:
        action = None
        action_name = None

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Movement keys
                if event.key in [pygame.K_w, pygame.K_UP]:
                    action = Action.FORWARD
                    action_name = "FORWARD"
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    action = Action.TURN_LEFT
                    action_name = "TURN_LEFT"
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    action = Action.TURN_RIGHT
                    action_name = "TURN_RIGHT"
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    action = Action.STAY
                    action_name = "STAY"

                # Control keys
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("\n" + "=" * 60)
                    print("ENVIRONMENT RESET")
                    print_info(obs, 0, False, info, "RESET")

                elif event.key == pygame.K_q:
                    running = False

        # Execute action if one was selected
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print_info(obs, reward, done, info, action_name)

            if done:
                print("\n" + "=" * 60)
                print("EPISODE FINISHED!")
                if info.get('task_status') == 1:  # SUCCESS
                    print("✓ TASK COMPLETED SUCCESSFULLY!")
                elif info.get('task_status') == 2:  # FAILURE
                    print("✗ TASK FAILED!")
                print("Press R to reset or Q to quit")
                print("=" * 60)

        # Render
        env.render()
        clock.tick(10)  # 10 FPS for manual control

    # Cleanup
    env.close()
    pygame.quit()


if __name__ == "__main__":
    # Test the wall-following environment
    print("Testing Wall Following Environment...")

    # Custom config for testing with specific map
    test_config = {
        'map_path': '/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_14.txt',
        'randomize': False,  # Use the loaded map
        'default_sensor_params': {
            'max_range': 4,  # Very short range
            'fov_deg': 25,  # Very narrow FOV
            'num_rays': 20
        }
    }

    run_manual_test(WallFollowingWrapper, test_config)