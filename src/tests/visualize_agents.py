"""
Interactive SLAM Environment Viewer with Simple Sensor Control
Uses the same simple configuration approach as env_manual.py
"""

import pygame
import numpy as np
import time
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Import environments and agents
from environments.base.slam_env import MultiAgentSLAMEnv
from environments.tasks.room_exploration_wrapper import RoomExplorationWrapper
from environments.tasks.navigation_wrapper import NavigationWrapper
from environments.tasks.room_entry_wrapper import RoomEntryWrapper
from environments.tasks.wall_following_wrapper import WallFollowingWrapper
from environments.tasks.room_utils import precompute_room_data
from environments.tasks.doorway_utils import precompute_doorways
from agents.room_frontier_agent import RoomFrontierAgent
from agents.frontier_agent import FrontierAgent
from agents.a_star_navigation_agent import AStarNavigationAgent
from agents.doorway_traversal_agent import DoorwayEntryAgent
from agents.wall_following_agent import WallFollowingAgent as LogicalWallAgent
from environments.base.constants import TILE_SIZE, TileType


class InteractiveSLAMViewer:
    """Interactive viewer for SLAM environments with simple sensor control."""

    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.fps = 10
        self.running = False
        self.pygame_initialized = True  # Track pygame initialization state
        self.verbose = False  # Toggle for verbose printing

        # Default map path
        self.map_path = None
        self.randomize = True

        # Simple sensor parameters (like env_manual.py)
        self.sensor_params = {
            'max_range': 2,
            'fov_deg': 45,
            'num_rays': 30
        }

        # Initialize precomputed data attributes
        self.precomputed_rooms = None
        self.precomputed_doorways = None

        # Selected environment and agent
        self.env = None
        self.agent = None
        self.num_agents = 1

    def ensure_pygame_initialized(self):
        """Ensure pygame is initialized before use"""
        if not self.pygame_initialized:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True

    def select_map_option(self):
        """Select map configuration - simple like env_manual.py"""
        print("\n" + "="*50)
        print("MAP CONFIGURATION")
        print("="*50)
        print("1. Use random map")
        print("2. Load specific map file")

        choice = input("\nEnter your choice (1-2) [1]: ").strip()

        if choice == '2':
            self.randomize = False
            map_path = input("Enter map file path (or press Enter for map_19): ").strip()
            if not map_path:
                # Default to map_19
                map_path = '/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_19.txt'

            # Check if file exists
            if not Path(map_path).exists():
                print(f"Warning: Map file not found at {map_path}")
                alt_path = input("Enter alternative path or press Enter to use random map: ").strip()
                if alt_path and Path(alt_path).exists():
                    self.map_path = alt_path
                else:
                    print("Using random map generation")
                    self.randomize = True
                    self.map_path = None
            else:
                self.map_path = map_path
                print(f"Using map: {self.map_path}")
        else:
            self.randomize = True
            self.map_path = None
            print("Using random map generation")

    def select_sensor_config(self):
        """Simple sensor configuration like env_manual.py"""
        print("\n" + "="*50)
        print("SENSOR CONFIGURATION")
        print("="*50)
        print("1. Short range (range=5, fov=30°)")
        print("2. Medium range (range=2, fov=45°) [default]")
        print("3. Long range (range=15, fov=60°)")
        print("4. Wide angle (range=8, fov=90°)")
        print("5. Custom")

        choice = input("\nEnter your choice (1-5) [2]: ").strip()

        if choice == '1':
            self.sensor_params = {
                'max_range': 5,
                'fov_deg': 30,
                'num_rays': 20
            }
        elif choice == '3':
            self.sensor_params = {
                'max_range': 15,
                'fov_deg': 60,
                'num_rays': 40
            }
        elif choice == '4':
            self.sensor_params = {
                'max_range': 8,
                'fov_deg': 90,
                'num_rays': 45
            }
        elif choice == '5':
            # Custom configuration
            print("\nEnter custom sensor parameters:")

            range_str = input(f"Max range (1-50) [{self.sensor_params['max_range']}]: ").strip()
            if range_str and range_str.isdigit():
                self.sensor_params['max_range'] = min(50, max(1, int(range_str)))

            fov_str = input(f"Field of view (10-360°) [{self.sensor_params['fov_deg']}]: ").strip()
            if fov_str and fov_str.isdigit():
                self.sensor_params['fov_deg'] = min(360, max(10, int(fov_str)))

            rays_str = input(f"Number of rays (10-100) [{self.sensor_params['num_rays']}]: ").strip()
            if rays_str and rays_str.isdigit():
                self.sensor_params['num_rays'] = min(100, max(10, int(rays_str)))
        else:
            # Default medium range
            self.sensor_params = {
                'max_range': 2,
                'fov_deg': 45,
                'num_rays': 30
            }

        print(f"\nUsing sensor config: range={self.sensor_params['max_range']}, "
              f"fov={self.sensor_params['fov_deg']}°, rays={self.sensor_params['num_rays']}")

    def select_environment(self):
        """Simple environment selection"""
        print("\n" + "="*50)
        print("SELECT ENVIRONMENT")
        print("="*50)
        print("1. Wall Following (boundary discovery)")
        print("2. Room Entry (doorway traversal)")
        print("3. Room Exploration (doorway avoidance)")
        print("4. Navigation (reach goal)")
        print("5. Basic SLAM")

        choice = input("\nEnter your choice (1-5): ").strip()

        # Base configuration for all environments
        base_config = {
            'num_agents': self.num_agents,
            'max_steps': 1000,
            'render_mode': 'human',
            'randomize': self.randomize,
            'default_sensor_params': self.sensor_params
        }

        if self.map_path:
            base_config['map_path'] = self.map_path

        if choice == '1':
            # Wall Following
            print("Selected: Wall Following")
            self.env = WallFollowingWrapper(env_config=base_config)

        elif choice == '2':
            # Room Entry
            print("Selected: Room Entry")
            config = {'env_config': base_config}

            # Precompute doorways if using specific map
            if self.map_path and self.precomputed_doorways is None:
                print("Precomputing doorways...")
                self.precomputed_doorways = precompute_doorways(self.map_path)
                print(f"Found {len(self.precomputed_doorways)} doorways")

            if self.precomputed_doorways:
                config['precomputed_doorways'] = self.precomputed_doorways

            # Ask about auto-exploration
            auto = input("Enable auto-exploration? (y/n) [y]: ").strip().lower()
            config['auto_explore'] = auto != 'n'

            self.env = RoomEntryWrapper(**config)

        elif choice == '3':
            # Room Exploration
            print("Selected: Room Exploration")
            config = {'env_config': base_config}

            # Precompute rooms if using specific map
            if self.map_path and self.precomputed_rooms is None:
                print("Precomputing room data...")
                self.precomputed_rooms = precompute_room_data(self.map_path)
                print(f"Found {len(self.precomputed_rooms['rooms'])} rooms")

            if self.precomputed_rooms:
                config['precomputed_rooms'] = self.precomputed_rooms

            self.env = RoomExplorationWrapper(**config)

        elif choice == '4':
            # Navigation
            print("Selected: Navigation")
            config = {
                'env_config': base_config,
                'exploration_steps': 50,
                'max_steps_to_goal': 200
            }
            self.env = NavigationWrapper(**config)

        else:
            # Basic SLAM
            print("Selected: Basic SLAM")
            self.env = MultiAgentSLAMEnv(**base_config)

    def select_agent(self, quick_select=False):
        """Simple agent selection"""
        if not quick_select:
            print("\n" + "="*50)
            print("SELECT AGENT")
            print("="*50)

        print("1. Logical Wall Agent (deterministic)")
        print("2. Frontier Agent (exploration)")
        print("3. Room Frontier Agent (doorway aware)")
        print("4. A* Navigation Agent (pathfinding)")
        print("5. Doorway Entry Agent (door traversal)")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            # Logical Wall Agent
            print("\nSelected: Logical Wall Agent")

            if not quick_select:
                # Ask for wall following direction
                direction = input("Follow direction (left/right) [right]: ").strip().lower()
                if direction not in ['left', 'right']:
                    direction = 'right'

                # Ask for search pattern
                pattern = input("Search pattern (forward/spiral) [forward]: ").strip().lower()
                if pattern not in ['forward', 'spiral']:
                    pattern = 'forward'

                print(f"Configuration: follow_{direction}, {pattern}_search")

            self.agent = LogicalWallAgent(
                num_agents=self.num_agents,
            )

        elif choice == '2':
            print("Selected: Frontier Agent")
            # Frontier Agent expects camera_range parameter
            self.agent = FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )

        elif choice == '3':
            print("Selected: Room Frontier Agent")
            # Room Frontier Agent expects camera_range parameter
            self.agent = RoomFrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )

        elif choice == '4':
            print("Selected: A* Navigation Agent")
            # A* Agent only takes num_agents and optionally replan_frequency
            self.agent = AStarNavigationAgent(
                num_agents=self.num_agents,
            )

        else:
            print("Selected: Doorway Entry Agent")
            # Doorway Entry Agent only takes num_agents
            self.agent = DoorwayEntryAgent(num_agents=self.num_agents)

    def print_agent_state(self, obs, info, action=None, step=0):
        """Print detailed agent state information"""
        print("\n" + "-"*40)
        print(f"Step {step}")

        # Print agent type
        agent_type = self.agent.__class__.__name__
        print(f"Agent: {agent_type}")

        # Print execution state
        if hasattr(self.agent, 'execution_state'):
            state = self.agent.execution_state
            print(f"Status: {state.value if hasattr(state, 'value') else state}")

            # Print error message if in error state
            if hasattr(self.agent, 'get_error_message'):
                error_msg = self.agent.get_error_message()
                if error_msg:
                    print(f"Error: {error_msg}")

        # Print position if available
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'agent_positions'):
            pos = self.env.env.agent_positions[0]
            print(f"Position: ({pos[0]}, {pos[1]})")

        # Print action if provided
        if action is not None:
            action_val = action[0] if isinstance(action, np.ndarray) else action
            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'STAY']
            print(f"Action: {action_names[action_val]}")

        # Print agent-specific state if available
        if hasattr(self.agent, 'get_state_info'):
            state_info = self.agent.get_state_info()
            for key, value in state_info.items():
                print(f"{key}: {value}")

        # Print agent metrics if available
        if hasattr(self.agent, 'get_metrics'):
            metrics = self.agent.get_metrics()
            # Skip execution_state and error_message as we already printed them
            for key, value in metrics.items():
                if key not in ['execution_state', 'error_message']:
                    print(f"{key}: {value}")

        # Print task-specific info
        if 'wall_coverage' in info:
            print(f"Wall Coverage: {info['wall_coverage']*100:.1f}%")
        if 'room_coverage' in info:
            print(f"Room Coverage: {info['room_coverage']*100:.1f}%")
        if 'exploration_coverage' in info:
            print(f"Exploration: {info['exploration_coverage']*100:.1f}%")

        print("-"*40)

    def change_agent_during_episode(self, obs, info):
        """Allow changing agent during episode"""
        print("\n" + "="*30)
        print("CHANGE AGENT")
        print("="*30)
        print("Current agent:", self.agent.__class__.__name__)
        print("-"*30)
        print("1. Logical Wall Agent")
        print("2. Frontier Agent")
        print("3. Room Frontier Agent")
        print("4. A* Navigation Agent")
        print("5. Doorway Entry Agent")
        print("0. Cancel")

        choice = input("\nSelect new agent (0-5): ").strip()

        if choice == '1':
            self.agent = LogicalWallAgent(num_agents=self.num_agents)
            print("Switched to: Logical Wall Agent")
        elif choice == '2':
            self.agent = FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
            print("Switched to: Frontier Agent")
        elif choice == '3':
            self.agent = RoomFrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
            print("Switched to: Room Frontier Agent")
        elif choice == '4':
            self.agent = AStarNavigationAgent(num_agents=self.num_agents)
            print("Switched to: A* Navigation Agent")
        elif choice == '5':
            self.agent = DoorwayEntryAgent(num_agents=self.num_agents)
            print("Switched to: Doorway Entry Agent")
        else:
            print("Agent change cancelled")
            return False

        # Reset the new agent with current state
        self.agent.reset()
        return True

    def run_episode(self):
        """Run a single episode with the selected configuration"""
        # Ensure pygame is initialized before running episode
        self.ensure_pygame_initialized()

        print("\n" + "="*50)
        print("STARTING EPISODE")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Sensor: range={self.sensor_params['max_range']}, "
              f"fov={self.sensor_params['fov_deg']}°, "
              f"rays={self.sensor_params['num_rays']}")
        print("="*50)
        print("Controls:")
        print("  SPACE = pause/resume")
        print("  ESC   = quit")
        print("  R     = reset episode")
        print("  S     = print current state")
        print("  A     = change agent")
        print("  V     = toggle verbose (auto-print every step)")
        print("="*50)

        # Reset environment and agent
        obs, info = self.env.reset()
        self.agent.reset()

        # Episode variables
        paused = False
        step = 0
        done = False
        total_reward = 0.0
        self.running = True
        self.verbose = False  # Reset verbose for each episode
        agent_error = False

        while self.running and not done:
            # Handle pygame events
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\nQuitting...")
                            self.running = False
                            break
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("PAUSED" if paused else "RESUMED")
                        elif event.key == pygame.K_r:
                            # Reset episode
                            obs, info = self.env.reset()
                            self.agent.reset()
                            step = 0
                            total_reward = 0.0
                            done = False
                            agent_error = False
                            print("\nEPISODE RESET")
                        elif event.key == pygame.K_s:
                            # Print current state
                            self.print_agent_state(obs, info, step=step)
                        elif event.key == pygame.K_a:
                            # Change agent
                            if self.change_agent_during_episode(obs, info):
                                agent_error = False
                                if paused:
                                    print("Agent changed! Press SPACE to resume.")
                                else:
                                    print("Agent changed! Continuing...")
                        elif event.key == pygame.K_v:
                            # Toggle verbose mode
                            self.verbose = not self.verbose
                            print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
            except pygame.error:
                # Pygame was quit, exit the episode
                self.running = False
                break

            if not self.running:
                break

            if not paused and not done and not agent_error:
                try:
                    # Get agent action
                    actions = self.agent.get_actions(obs, info)

                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(actions)
                    total_reward += reward
                    done = terminated or truncated

                    # Print state if verbose mode is on
                    if self.verbose:
                        self.print_agent_state(obs, info, actions, step=step)

                    # Print status periodically (less frequently if verbose)
                    print_interval = 100 if self.verbose else 50
                    if step % print_interval == 0:
                        action = actions[0] if isinstance(actions, np.ndarray) else actions
                        action_names = ['FORWARD', 'LEFT', 'RIGHT', 'STAY']
                        print(f"Step {step:4d} | Action: {action_names[action]:8s} | "
                              f"Reward: {total_reward:7.2f}", end='')

                        # Add task-specific info
                        if 'wall_coverage' in info:
                            print(f" | Wall: {info['wall_coverage']*100:.1f}%", end='')
                        if 'room_coverage' in info:
                            print(f" | Room: {info['room_coverage']*100:.1f}%", end='')
                        if 'boundary_found' in info and info['boundary_found']:
                            print(" | BOUNDARY!", end='')
                        if 'has_passed_through' in info and info['has_passed_through']:
                            print(" | PASSED!", end='')

                        print()

                    step += 1

                except Exception as e:
                    print(f"\n⚠️ AGENT ERROR: {e}")
                    print(f"Error occurred with {self.agent.__class__.__name__} at step {step}")
                    agent_error = True
                    paused = True
                    print("\nEpisode paused. Options:")
                    print("  A - Change to a different agent")
                    print("  R - Reset episode")
                    print("  ESC - Quit")

            # Render
            self.env.render()
            self.clock.tick(self.fps)

        # Episode complete
        if done:
            print("\n" + "="*50)
            print("EPISODE COMPLETE")
            print(f"Agent: {self.agent.__class__.__name__}")
            print(f"Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")

            if 'task_status' in info:
                status = ['IN_PROGRESS', 'SUCCESS', 'FAILURE'][info['task_status']]
                print(f"Status: {status}")

                if info['task_status'] == 1:
                    print("✓ Task completed successfully!")
                elif info['task_status'] == 2:
                    print("✗ Task failed")

            print("="*50)

            # Ask if user wants to continue with a different agent
            print("\nWhat would you like to do?")
            print("1. Try another agent on the same environment")
            print("2. Run same agent again")
            print("3. Change configuration")
            print("4. Quit")

            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                print("\nSelect a new agent:")
                self.select_agent(quick_select=True)
                self.agent.reset()
                return 'continue_new_agent'
            elif choice == '2':
                return 'continue_same'
            elif choice == '3':
                return 'change_config'
            else:
                return 'quit'

        return 'interrupted' if not self.running else 'continue_same'

    def run(self):
        """Main execution loop"""
        print("\n" + "="*50)
        print("SLAM ENVIRONMENT INTERACTIVE VIEWER")
        print("="*50)

        while True:
            # Configure everything
            self.select_map_option()
            self.select_sensor_config()

            # Ask for number of agents
            num_str = input("\nNumber of agents (1-5) [1]: ").strip()
            if num_str and num_str.isdigit():
                self.num_agents = min(5, max(1, int(num_str)))
            else:
                self.num_agents = 1
            print(f"Using {self.num_agents} agent(s)")

            # Select environment and agent
            self.select_environment()
            self.select_agent()

            # Run episodes
            while True:
                result = self.run_episode()

                if result == 'quit' or result == 'interrupted':
                    if self.env:
                        self.env.close()
                    if self.pygame_initialized:
                        pygame.quit()
                        self.pygame_initialized = False
                    print("\nGoodbye!")
                    return
                elif result == 'change_config':
                    # Close current environment
                    self.env.close()
                    # Quit pygame to reset video system
                    pygame.quit()
                    self.pygame_initialized = False
                    break
                elif result == 'continue_new_agent':
                    # Continue with the newly selected agent
                    continue
                else:  # continue_same
                    continue

        # Cleanup
        if self.env:
            self.env.close()
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        print("\nGoodbye!")


if __name__ == "__main__":
    viewer = InteractiveSLAMViewer()
    viewer.run()