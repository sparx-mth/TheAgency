"""
visualize_agent.py - Generic script to visualize trained agents
Supports: Wall Following, Room Entry, Navigation, Room Exploration
"""

import numpy as np
import time
from stable_baselines3 import DQN
from sensors.camera_sensor import CameraSensor
from environments.tasks.doorway_utils import precompute_doorways
from environments.tasks.room_utils import precompute_room_data

# ============================================================
# CONFIGURATION - MODIFY THIS SECTION
# ============================================================

# Choose which agent/environment to visualize
TASK = "wall_following"  # Options: "wall_following", "room_entry", "navigation", "room_exploration"

# Model paths for each task
MODELS = {
    "wall_following": "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/dqn_wall_following_final.zip",
    "room_entry": "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/dqn_room_entry_final.zip",
    "navigation": "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/dqn_navigation_final.zip",
    "room_exploration": "/home/nadavc/PycharmProjects/TheAgency_workspace/src/rl/dqn/models/dqn_room_exploration_final.zip"
}

# Map path
MAP_PATH = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt"

# Visualization settings
N_EPISODES = 1
RENDER_FPS = 30

# ============================================================
# PRECOMPUTE DATA (Done once at startup)
# ============================================================

# Precompute doorways for room_entry task
PRECOMPUTED_DOORWAYS = None
if TASK == "room_entry":
    print("Precomputing doorways...")
    PRECOMPUTED_DOORWAYS = precompute_doorways(MAP_PATH)
    print(f"Found {len(PRECOMPUTED_DOORWAYS)} doorways")

# Precompute room data for room_exploration task
PRECOMPUTED_ROOMS = None
if TASK == "room_exploration":
    print("Precomputing room data...")
    PRECOMPUTED_ROOMS = precompute_room_data(MAP_PATH)
    print(f"Found {len(PRECOMPUTED_ROOMS['rooms'])} rooms")
    print(f"Found {len(PRECOMPUTED_ROOMS['doorways'])} doorways")

# ============================================================
# ENVIRONMENT CREATION
# ============================================================

def create_wall_following_env():
    """Create wall-following environment."""
    from environments.tasks.wall_following_wrapper import WallFollowingWrapper

    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    height, width = loaded_map.shape

    sensor = CameraSensor(max_range=2, fov_deg=90, num_rays=12)

    env_config = {
        'width': width,
        'height': height,
        'num_agents': 1,
        'max_steps': 1000,
        'map_path': MAP_PATH,
        'render_mode': 'human',
        'sensor_config': {0: sensor},
    }

    return WallFollowingWrapper(env_config=env_config)


def create_room_entry_env():
    """Create room entry environment with precomputed doorways."""
    from environments.tasks.room_entry_wrapper import RoomEntryWrapper

    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    height, width = loaded_map.shape

    sensor = CameraSensor(max_range=3, fov_deg=90, num_rays=16)

    env_config = {
        'width': width,
        'height': height,
        'num_agents': 1,
        'max_steps': 1000,
        'map_path': MAP_PATH,
        'render_mode': 'human',
        'sensor_config': {0: sensor},
    }

    # Use precomputed doorways
    if PRECOMPUTED_DOORWAYS is None or len(PRECOMPUTED_DOORWAYS) == 0:
        print("Warning: No doorways found. Precomputing now...")
        doorways = precompute_doorways(MAP_PATH)
    else:
        doorways = PRECOMPUTED_DOORWAYS

    return RoomEntryWrapper(
        env_config=env_config,
        precomputed_doorways=doorways,
        auto_explore=True,
        max_exploration_steps=100
    )


def create_navigation_env():
    """Create navigation environment."""
    from environments.tasks.navigation_wrapper import NavigationWrapper

    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    height, width = loaded_map.shape

    sensor = CameraSensor(max_range=3, fov_deg=90, num_rays=16)

    env_config = {
        'width': width,
        'height': height,
        'num_agents': 1,
        'max_steps': 1000,
        'map_path': MAP_PATH,
        'render_mode': 'human',
        'sensor_config': {0: sensor},
    }

    return NavigationWrapper(
        env_config=env_config,
        exploration_steps=20,
        goal_selection="farthest"
    )


def create_room_exploration_env():
    """Create room exploration environment with precomputed room data."""
    from environments.tasks.room_exploration_wrapper import RoomExplorationWrapper

    loaded_map = np.loadtxt(MAP_PATH, dtype=np.int8)
    height, width = loaded_map.shape

    sensor = CameraSensor(max_range=3, fov_deg=90, num_rays=16)

    env_config = {
        'width': width,
        'height': height,
        'num_agents': 1,
        'max_steps': 1000,
        'map_path': MAP_PATH,
        'render_mode': 'human',
        'sensor_config': {0: sensor},
    }

    # Use precomputed room data
    if PRECOMPUTED_ROOMS is None:
        print("Warning: No room data found. Precomputing now...")
        room_data = precompute_room_data(MAP_PATH)
    else:
        room_data = PRECOMPUTED_ROOMS

    return RoomExplorationWrapper(
        env_config=env_config,
        precomputed_rooms=room_data,
        auto_explore=False,
        max_exploration_steps=100
    )


# Environment factory
ENV_CREATORS = {
    "wall_following": create_wall_following_env,
    "room_entry": create_room_entry_env,
    "navigation": create_navigation_env,
    "room_exploration": create_room_exploration_env
}

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_agent(model_path, env, task_name, n_episodes=10, fps=10):
    """
    Generic agent visualization function.

    Args:
        model_path: Path to trained model
        env: Environment instance
        task_name: Name of the task for display
        n_episodes: Number of episodes to show
        fps: Target frames per second
    """
    # Load model
    print(f"Loading {task_name} model from {model_path}")
    try:
        model = DQN.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: Could not load model from {model_path}")
        print(f"Error details: {e}")
        print("Using random actions for demonstration...")
        model = None

    frame_delay = 1.0 / fps

    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get action
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Random action if no model

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Render
            env.render()
            time.sleep(frame_delay)

            # Task-specific progress display
            if steps % 50 == 0:
                if task_name == "wall_following":
                    coverage = info.get('wall_coverage', 0) * 100
                    print(f"  Step {steps}: Wall Coverage {coverage:.1f}%")
                elif task_name == "room_entry":
                    doors = info.get('discovered_doorways', 0)
                    passed = info.get('has_passed_through', False)
                    print(f"  Step {steps}: Doors {doors}, Passed: {passed}")
                elif task_name == "navigation":
                    dist = info.get('distance_to_goal', -1)
                    print(f"  Step {steps}: Distance to Goal: {dist}")
                elif task_name == "room_exploration":
                    coverage = info.get('room_coverage', 0) * 100
                    print(f"  Step {steps}: Room Coverage {coverage:.1f}%")

        # Episode summary
        success = info.get('task_success', False)
        status = "SUCCESS" if success else "FAILED"

        print(f"\n  Episode {episode + 1} {status}")
        print(f"  Steps: {steps}, Total Reward: {total_reward:.1f}")

        # Task-specific summary
        if task_name == "wall_following":
            coverage = info.get('wall_coverage', 0) * 100
            print(f"  Final Wall Coverage: {coverage:.1f}%")
        elif task_name == "room_entry":
            print(f"  Doorways Found: {info.get('discovered_doorways', 0)}")
            print(f"  Passed Through: {info.get('has_passed_through', False)}")
        elif task_name == "navigation":
            print(f"  Goal Reached: {success}")
            print(f"  Final Distance: {info.get('distance_to_goal', -1)}")
        elif task_name == "room_exploration":
            coverage = info.get('room_coverage', 0) * 100
            print(f"  Final Room Coverage: {coverage:.1f}%")
            print(f"  Door Crossed: {info.get('passed_through_door', False)}")

        # Brief pause between episodes
        time.sleep(1)

    env.close()
    print(f"\n{task_name.replace('_', ' ').title()} visualization complete!")


def main():
    """Main function."""
    print("=" * 60)
    print(f"AGENT VISUALIZATION - {TASK.replace('_', ' ').upper()}")
    print(f"Model: {MODELS.get(TASK, 'Not configured')}")
    print(f"Map: {MAP_PATH}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Target FPS: {RENDER_FPS}")
    print("=" * 60)

    # Validate task selection
    if TASK not in ENV_CREATORS:
        print(f"\nError: Unknown task '{TASK}'")
        print(f"Available tasks: {', '.join(ENV_CREATORS.keys())}")
        return

    # Create environment
    print(f"\nCreating {TASK} environment...")
    try:
        env = ENV_CREATORS[TASK]()
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Visualize agent
    model_path = MODELS.get(TASK)
    if not model_path:
        print(f"Warning: No model path configured for {TASK}")
        model_path = "dummy_path"  # Will use random actions

    visualize_agent(model_path, env, TASK, N_EPISODES, RENDER_FPS)


if __name__ == "__main__":
    main()