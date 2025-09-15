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
from environments.tasks.room_exploration_wrapper import RoomExplorationWrapper
from environments.tasks.navigation_wrapper import NavigationWrapper
from environments.base.constants import Action

# Import utility functions for pre-computation
from environments.tasks.doorway_utils import precompute_doorways
from environments.tasks.room_utils import precompute_room_data
# Note: wall_utils removed - WallFollowingWrapper handles walls dynamically


def print_info(obs: dict, reward: float, done: bool, info: dict, action_name: str):
    """Print comprehensive environment information including collision count."""
    print("\n" + "=" * 60)
    print(f"ACTION: {action_name}")
    print(f"Position: {obs['positions']}")
    print(f"Facing: {obs['facings']} (0=N, 1=E, 2=S, 3=W)")
    print(f"Reward: {reward:.3f}")
    print(f"Done: {done}")
    print(f"Task Status: {info.get('task_status', 'N/A')} (0=Progress, 1=Success, 2=Failure)")
    print(f"Step: {info.get('task_step', 0)}")

    # COLLISION TRACKING
    print(f"Collisions: {info.get('collision_count', 0)} (total this episode)")
    if 'collision_occurred' in info and info['collision_occurred']:
        print("  COLLISION DETECTED THIS STEP!")

    # Wall following specific info (updated for new version)
    if 'pre_search_time' in info and info['pre_search_time'] > 0:
        print(f"Pre-search: {info['pre_search_steps']} steps in {info['pre_search_time']:.3f}s")

    if 'wall_locked' in info:
        print(f"Wall Locked: {info['wall_locked']}")
        if info['wall_locked']:
            print(f"  Total wall cells: {info.get('total_wall_cells', 0)}")
            print(f"  Accessible cells: {info.get('total_accessible', 0)}")
            print(f"  Discovered: {info.get('discovered_accessible', 0)}")

    if 'wall_coverage' in info:
        print(f"Wall Coverage: {info['wall_coverage']:.1%}")

    if 'normalized_discovery_progress' in info:
        print(f"Discovery Progress: {info['normalized_discovery_progress']:.1%}")

    if 'phase' in info:
        print(f"Phase: {info['phase']}")

    if 'wall_contact_steps' in info and info['wall_contact_steps'] > 0:
        print(f"Wall Contact Steps: {info['wall_contact_steps']}")

    # Auto-exploration info (for other environments)
    if 'is_exploring' in info:
        if info['is_exploring']:
            print(f"AUTO-EXPLORING: Step {info.get('exploration_steps', 0)}")
        elif info.get('exploration_steps', 0) > 0:
            print(f"Auto-exploration completed in {info['exploration_steps']} steps")

    # Task-specific info for other environments
    if 'discovered_doorways' in info:
        print(f"Discovered Doorways: {info['discovered_doorways']}")
    if 'has_passed_through' in info:
        print(f"Passed Through Doorway: {info['has_passed_through']}")
    if 'room_coverage' in info:
        print(f"Room Coverage: {info['room_coverage']:.1%}")

    # Navigation-specific info
    if 'goal_position' in info and info['goal_position']:
        print(f"Goal Position: {info['goal_position']}")
        print(f"Steps to Goal: {info.get('steps_to_goal', 0)}")

    # Map stats
    global_map = obs['global_map']
    visible_cells = np.sum(global_map != -1)  # -1 is UNKNOWN
    wall_cells = np.sum(global_map == 1)  # 1 is WALL
    print(f"\nMap Stats:")
    print(f"  Visible cells: {visible_cells}")
    print(f"  Wall cells found: {wall_cells}")


def run_manual_test(env_class: Any, env_config: dict = None, precomputed_data: dict = None):
    """
    Run manual testing for any task environment.

    Args:
        env_class: Environment class to test
        env_config: Configuration for the environment
        precomputed_data: Pre-computed data for the environment (if applicable)
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
            'max_range': 2,  # Short range
            'fov_deg': 60,  # Narrow FOV
            'num_rays': 24
        }
    }

    if env_config:
        default_config.update(env_config)

    # Create environment with appropriate pre-computed data
    kwargs = {'env_config': default_config}
    if precomputed_data:
        kwargs.update(precomputed_data)

    env = env_class(**kwargs)

    # Reset and get initial observation
    reset_result = env.reset()

    # Handle both old and new gym API
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    env.render()

    print("\n" + "=" * 60)
    print(f"TESTING: {env_class.__name__}")
    print("=" * 60)

    # Special notice for Wall Following
    if env_class == WallFollowingWrapper:
        print("NOTE: Wall Following now automatically searches for a wall")
        print("      before starting. The episode begins with a wall visible.")
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
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        obs, info = reset_result
                    else:
                        obs = reset_result
                        info = {}
                    print("\n" + "=" * 60)
                    print("ENVIRONMENT RESET")
                    if env_class == WallFollowingWrapper and 'pre_search_time' in info:
                        print(f"Pre-search completed: {info['pre_search_steps']} steps in {info['pre_search_time']:.3f}s")
                    print_info(obs, 0, False, info, "RESET")

                elif event.key == pygame.K_q:
                    running = False

        # Execute action if one was selected
        if action is not None:
            # Convert action to array for SLAM environment
            action_array = np.array([action])

            step_result = env.step(action_array)

            # Handle both old and new gym API
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

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
    print(info)
    # Cleanup
    env.close()
    pygame.quit()


def select_environment():
    """Interactive menu to select which environment to test."""
    print("\n" + "=" * 60)
    print("SLAM TASK ENVIRONMENT TESTER")
    print("=" * 60)
    print("\nSelect environment to test:")
    print("1. Wall Following (Optimized)")
    print("2. Room Entry (Doorway Passing)")
    print("3. Room Exploration")
    print("4. Navigation to Goal")
    print("Q. Quit")
    print("-" * 60)

    while True:
        choice = input("Enter your choice (1-4 or Q): ").strip().upper()

        if choice == '1':
            return WallFollowingWrapper, "Wall Following (Optimized)"
        elif choice == '2':
            return RoomEntryWrapper, "Room Entry (Doorway Passing)"
        elif choice == '3':
            return RoomExplorationWrapper, "Room Exploration"
        elif choice == '4':
            return NavigationWrapper, "Navigation to Goal"
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

    if choice == '1':
        return {'randomize': True}, None
    else:
        map_path = input("Enter map file path (or press Enter for default): ").strip()
        if not map_path:
            map_path = '/home/user/nadav/TheAgency/resources/planner/maps/house_map_19.txt'
        return {'map_path': map_path, 'randomize': False}, map_path


def precompute_environment_data(env_class, map_path):
    """Pre-compute data for environments that require it."""
    if map_path is None:
        # For random maps, we can't pre-compute, environments will handle it
        return {}

    precomputed_data = {}

    if env_class == RoomEntryWrapper:
        # Pre-compute doorways
        doorways = precompute_doorways(map_path)
        precomputed_data['precomputed_doorways'] = doorways
        print(f"Pre-computed {len(doorways)} doorways")

    elif env_class == RoomExplorationWrapper:
        # Pre-compute room data
        room_data = precompute_room_data(map_path)
        precomputed_data['precomputed_rooms'] = room_data
        print(f"Pre-computed {len(room_data['rooms'])} rooms")

    elif env_class == WallFollowingWrapper:
        # WallFollowingWrapper now handles everything automatically
        print("Wall Following: Automatic pre-search for walls enabled")
        print("                Episodes start with a wall already visible")

    return precomputed_data


def select_auto_explore_option(env_class):
    """Select whether to enable auto-exploration (not applicable to Wall Following)."""
    # Wall Following now always does pre-search, so skip this option
    if env_class == WallFollowingWrapper:
        return False

    print("\nAuto-exploration option:")
    print("1. Enable auto-exploration (agent explores until task-ready)")
    print("2. Disable auto-exploration (manual exploration)")

    choice = input("Enter your choice (1-2): ").strip()
    return choice != '2'


if __name__ == "__main__":
    while True:
        # Select environment
        env_class, env_name = select_environment()

        if env_class is None:
            print("Exiting...")
            break

        print(f"\nSelected: {env_name}")

        # Select map options
        map_config, map_path = select_map_option()

        # Configure sensor parameters
        print("\nSensor configuration:")
        print("1. Default (range=2 fov=60°)")
        print("2. Short range (range=3, fov=25°)")
        print("3. Long range (range=7, fov=45°)")

        sensor_choice = input("Enter your choice (1-3): ").strip()

        if sensor_choice == '2':
            sensor_params = {
                'max_range': 3,
                'fov_deg': 25,
                'num_rays': 20
            }
        elif sensor_choice == '3':
            sensor_params = {
                'max_range': 7,
                'fov_deg': 45,
                'num_rays': 90
            }
        else:
            sensor_params = {
                'max_range': 2,
                'fov_deg': 60,
                'num_rays': 24
            }

        # Build config
        test_config = {
            **map_config,
            'default_sensor_params': sensor_params
        }

        # Pre-compute environment data if using a specific map
        precomputed_data = precompute_environment_data(env_class, map_path)

        # Ask about auto-exploration for applicable environments (not Wall Following)
        if env_class in [RoomEntryWrapper, RoomExplorationWrapper]:
            auto_explore = select_auto_explore_option(env_class)
            precomputed_data['auto_explore'] = auto_explore
            if auto_explore:
                print("Auto-exploration enabled - agent will explore automatically at start")
            else:
                print("Auto-exploration disabled - manual exploration required")

        # Run the test
        print(f"\nStarting {env_name} environment...")
        run_manual_test(env_class, test_config, precomputed_data)

        # Ask if user wants to test another environment
        print("\n" + "=" * 60)
        another = input("Test another environment? (y/n): ").strip().lower()
        if another != 'y':
            break

    print("\nGoodbye!")