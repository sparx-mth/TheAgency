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
from environments.tasks.room_entry_wrapper import RoomEntryWrapper
from environments.base.constants import Action


def print_info(obs: dict, reward: float, done: bool, info: dict, action_name: str):
    """Print comprehensive environment information."""
    print("\n" + "=" * 60)
    print(f"ACTION: {action_name}")
    print(f"Position: {obs['position']}")
    print(f"Facing: {obs['facing']} (0=N, 1=E, 2=S, 3=W)")
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Task Status: {info.get('task_status', 'N/A')} (0=Progress, 1=Success, 2=Failure)")
    print(f"Step: {info.get('task_step', 0)}")

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
            'num_rays': 90
        }
    }

    if env_config:
        default_config.update(env_config)

    # Create environment
    env = env_class(env_config=default_config)

    # Reset and get initial observation
    obs, info = env.reset()
    env.render()

    print("\n" + "=" * 60)
    print(f"TESTING: {env_class.__name__}")
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


def select_environment():
    """Interactive menu to select which environment to test."""
    print("\n" + "=" * 60)
    print("SLAM TASK ENVIRONMENT TESTER")
    print("=" * 60)
    print("\nSelect environment to test:")
    print("1. Wall Following")
    print("2. Room Entry")
    print("3. Room Exit (Not implemented yet)")
    print("Q. Quit")
    print("-" * 60)

    while True:
        choice = input("Enter your choice (1-3 or Q): ").strip().upper()

        if choice == '1':
            return WallFollowingWrapper, "Wall Following"
        elif choice == '2':
            return RoomEntryWrapper, "Room Entry"
        elif choice == '3':
            print("Room Exit environment not implemented yet.")
            continue
        elif choice == 'Q':
            return None, None
        else:
            print("Invalid choice. Please try again.")


def select_map_option():
    """Select whether to use a specific map or random generation."""
    print("\nMap options:")
    print("1. Use random map")
    print("2. Load specific map file")

    choice = input("Enter your choice (1-2): ").strip()

    if choice == '2':
        map_path = input("Enter map file path (or press Enter for default): ").strip()
        if not map_path:
            map_path = '/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_11.txt'
        return {'map_path': map_path, 'randomize': False}
    else:
        return {'randomize': True}


if __name__ == "__main__":
    while True:
        # Select environment
        env_class, env_name = select_environment()

        if env_class is None:
            print("Exiting...")
            break

        print(f"\nSelected: {env_name}")

        # Select map options
        map_config = select_map_option()

        # Configure sensor parameters
        print("\nSensor configuration:")
        print("1. Default (range=5, fov=30°)")
        print("2. Short range (range=3, fov=25°)")
        print("3. Long range (range=10, fov=45°)")

        sensor_choice = input("Enter your choice (1-3): ").strip()

        if sensor_choice == '2':
            sensor_params = {
                'max_range': 3,
                'fov_deg': 25,
                'num_rays': 20
            }
        elif sensor_choice == '3':
            sensor_params = {
                'max_range': 10,
                'fov_deg': 45,
                'num_rays': 90
            }
        else:
            sensor_params = {
                'max_range': 5,
                'fov_deg': 30,
                'num_rays': 60
            }

        # Build config
        test_config = {
            **map_config,
            'default_sensor_params': sensor_params
        }

        # Run the test
        print(f"\nStarting {env_name} environment...")
        run_manual_test(env_class, test_config)

        # Ask if user wants to test another environment
        print("\n" + "=" * 60)
        another = input("Test another environment? (y/n): ").strip().lower()
        if another != 'y':
            break

    print("\nGoodbye!")