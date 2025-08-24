"""
show_agent.py - Load trained PPO weights and show the agent solving house_map_10
Uses exact same environment parameters as training
"""

import os
import pygame
import time
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.base.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor


def create_env_exact_same():
    """Create environment with EXACT same settings as training."""

    # EXACT same sensor as training
    sensor = CameraSensor(
        max_range=5,      # Same as training
        fov_deg=90,       # Same as training
        num_rays=20       # Same as training
    )

    # EXACT same environment parameters as training
    env = MultiAgentSLAMEnv(
        width=10,
        height=10,
        num_agents=1,
        max_steps=5000,    # Same as training
        map_path="/home/user/nadav/TheAgency/resources/planner/maps/house_map_13.txt",
        randomize=False,  # Same as training - always use the same map
        render_mode='human',  # ONLY DIFFERENCE: render mode for visualization
        sensor_config={0: sensor},
        # EXACT same reward structure as training
        discovery_reward=1.0,      # Same as training
        collision_penalty=-0.1,    # Same as training
        step_penalty=0.0,          # Same as training
        completion_bonus=50.0,     # Same as training
    )

    return Monitor(env)


def show_agent():
    """Load trained agent and show it solving the map."""

    print("="*60)
    print("SHOWING TRAINED PPO AGENT ON HOUSE MAP 10")
    print("Using EXACT same parameters as training")
    print("="*60)

    # Check if model exists
    model_path = "./models/simple/final_model"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\nError: Model not found at {model_path}.zip")
        print("Please run train_simple.py first to train the agent")
        return

    # Load the trained PPO model
    print(f"\nLoading trained model from {model_path}.zip...")
    model = PPO.load(model_path)
    print(" Model loaded successfully")

    # Create environment with EXACT same configuration
    print("\nCreating environment (exact same as training)...")
    env = create_env_exact_same()

    # Wrap in DummyVecEnv (same as training)
    vec_env = DummyVecEnv([lambda: env])

    # Load the EXACT normalization statistics from training
    norm_path = "./models/simple/vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"Loading normalization statistics from {norm_path}...")
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False  # Set to evaluation mode
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        print(" Normalization loaded (using exact training statistics)")
    else:
        print(" Warning: Normalization file not found - results may differ")

    print("\n" + "="*60)
    print("VISUALIZATION STARTING")
    print("="*60)
    print("\nYou should see a pygame window showing:")
    print("  • Left panel: True map")
    print("  • Right panel: What the agent has discovered")
    print("  • Yellow dot: The agent")
    print("  • Progress bar at bottom")
    print("\nControls:")
    print("  • Close window or press Ctrl+C to exit")
    print("  • The agent will continuously solve the map")
    print("="*60 + "\n")

    # Initialize pygame
    pygame.init()
    pygame.display.init()

    # Get the actual environment from vec_env for rendering
    actual_env = vec_env.envs[0]
    if hasattr(actual_env, 'env'):  # If it's wrapped in Monitor
        actual_env = actual_env.env

    episode = 0
    try:
        while True:
            episode += 1
            print(f"\nEpisode {episode}")
            print("-" * 40)

            # Reset environment
            obs = vec_env.reset()

            # Initial render
            actual_env.render()

            done = False
            steps = 0
            total_reward = 0.0

            # Run episode
            while not done:
                # Get action from trained model (deterministic for consistent behavior)
                action, _states = model.predict(obs, deterministic=True)

                # Execute action
                obs, reward, done, info = vec_env.step(action)

                # IMPORTANT: Explicitly call render after each step
                actual_env.render()

                # Update counters
                steps += 1
                total_reward += reward[0]

                # Small delay to make visualization visible
                time.sleep(0.03)  # ~33 FPS

                # Print progress every 25 steps
                if steps % 25 == 0:
                    progress = info[0].get('progress', 0) * 100
                    discovered = info[0].get('discovered_cells', 0)
                    print(f"  Step {steps:3d}: Progress {progress:5.1f}% | Discovered {discovered:2d} cells")

                # Check for window close
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

            # Episode complete
            final_progress = info[0].get('progress', 0) * 100
            final_discovered = info[0].get('discovered_cells', 0)

            print(f"\n Episode {episode} Complete!")
            print(f"  • Total steps: {steps}")
            print(f"  • Total reward: {total_reward:.2f}")
            print(f"  • Final progress: {final_progress:.1f}%")
            print(f"  • Cells discovered: {final_discovered}/92")

            if final_progress >= 99:
                print("  SUCCESS! Entire map explored!")
            elif final_progress >= 80:
                print("  Good exploration but missed some areas")
            else:
                print("  Incomplete exploration - agent got stuck")

            # Pause before next episode
            print("\nPausing 3 seconds before next episode...")
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n Visualization stopped by user")

    finally:
        vec_env.close()
        pygame.quit()
        print("\nDone!")


if __name__ == "__main__":
    # Run the visualization
    show_agent()