"""
Interactive SLAM Environment Viewer with Simple Sensor Control and Mission System
Now includes LLM integration for natural language mission planning
"""

import pygame
import numpy as np
import time
import sys
import subprocess
import tempfile
import os
import re
from typing import Dict, Any, Optional, List, Tuple
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


class Mission:
    """Represents a single mission for an agent"""
    def __init__(self, agent_id: int, agent_type: str, task: str, goal: Optional[Tuple[int, int]] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.task = task
        self.goal = goal
        self.completed = False
        self.agent_instance = None


class InteractiveSLAMViewer:
    """Interactive viewer for SLAM environments with simple sensor control and mission system."""

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
        self.manual_goal = None  # Store manual goal position

        # Mission system
        self.missions: List[Mission] = []
        self.current_mission_idx = 0
        self.mission_mode = False
        self.active_agent_id = 0  # Track which agent is currently active
        self.reset_between_missions = False  # Whether to reset env between missions
        self.auto_advance_missions = True  # Whether to auto-advance on completion

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
                map_path = '/home/user/nadav/TheAgency/resources/planner/maps/house_map_19.txt'

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

    def get_llm_plan(self, command):
        """Query LLM for mission plan"""
        prompt = f"""You are controlling a drone with these agents:
Agent 1: doorway - enters/exits rooms through doors
Agent 2: navigate - moves to specific coordinates
Agent 3: room - scans the current room
Agent 4: wall - follows walls

Convert this natural language command into numbered steps.
Each step should say "Activate agent X" where X is 1-4.

Command: {command}

Response (list the steps):"""

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(prompt)
            temp_file = f.name

        try:
            # Call Ollama
            cmd = f"cat {temp_file} | ollama run llama3.1:8b"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            response = result.stdout.strip()
            os.unlink(temp_file)
            return response
        except Exception as e:
            print(f"LLM Error: {e}")
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return None

    def parse_llm_output(self, llm_output):
        """Parse LLM output into Mission objects"""
        missions = []

        # Look for "agent X" patterns in the output
        lines = llm_output.split('\n')

        for line in lines:
            # Match patterns like "agent 1", "Agent 2", "activate agent 3"
            match = re.search(r'agent\s+(\d)', line.lower())
            if match:
                agent_id = int(match.group(1))

                # Map agent numbers to task types
                agent_map = {
                    1: ('doorway', 'doorway'),
                    2: ('navigation', 'navigate'),
                    3: ('room', 'room'),
                    4: ('wall', 'wall')
                }

                if agent_id in agent_map:
                    agent_type, task = agent_map[agent_id]

                    # For navigation, try to extract coordinates if mentioned
                    goal = None
                    if agent_type == 'navigation':
                        # Look for coordinates in format (x,y) or "x,y" or "to X,Y"
                        coord_match = re.search(r'(\d+)\s*,\s*(\d+)', line)
                        if coord_match:
                            goal = (int(coord_match.group(1)), int(coord_match.group(2)))
                        else:
                            # Use current position as default
                            goal = self.get_current_position() if hasattr(self, 'env') and self.env else (10, 10)

                    mission = Mission(agent_id, agent_type, task, goal)
                    missions.append(mission)

        return missions

    def setup_llm_missions(self):
        """Setup missions using LLM to interpret natural language commands"""
        print("\n" + "="*50)
        print("LLM MISSION PLANNING")
        print("="*50)
        print("Describe what you want the drone to do in natural language.")
        print("Examples:")
        print("  - 'scan the kitchen'")
        print("  - 'explore all rooms'")
        print("  - 'go to the bathroom and scan it'")
        print()

        command = input("Your command: ").strip()
        if not command:
            print("No command provided. Switching to manual mission setup.")
            return self.setup_missions()

        print("\nQuerying LLM for mission plan...")

        # Get LLM plan
        llm_output = self.get_llm_plan(command)

        if not llm_output:
            print("Failed to get LLM response. Switching to manual setup.")
            return self.setup_missions()

        print("\nLLM Plan:")
        print("-"*40)
        print(llm_output)
        print("-"*40)

        # Parse LLM output to missions
        self.missions = self.parse_llm_output(llm_output)

        if not self.missions:
            print("\nCould not parse missions from LLM. Switching to manual setup.")
            return self.setup_missions()

        print(f"\nGenerated {len(self.missions)} missions:")
        for i, m in enumerate(self.missions, 1):
            agent_name_map = {
                'wall': 'Wall Following',
                'frontier': 'Frontier Explorer',
                'room': 'Room Scanner',
                'navigation': 'Navigator',
                'doorway': 'Door Handler'
            }
            print(f"  {i}. Agent {m.agent_id}: {agent_name_map.get(m.agent_type, m.agent_type)}")
            if m.goal:
                print(f"     Target: {m.goal}")

        # Ask about execution options
        print("\nExecution options:")
        auto_option = input("Auto-advance missions? (y/n) [y]: ").strip().lower()
        self.auto_advance_missions = (auto_option != 'n')

        reset_option = input("Reset between missions? (y/n) [n]: ").strip().lower()
        self.reset_between_missions = (reset_option == 'y')

    def select_operation_mode(self):
        """Select between single agent, mission mode, or LLM mode"""
        print("\n" + "="*50)
        print("OPERATION MODE")
        print("="*50)
        print("1. Single Agent Mode (classic)")
        print("2. Mission Mode (manual multi-agent tasks)")
        print("3. LLM Mission Mode (natural language commands)")

        choice = input("\nEnter your choice (1-3) [1]: ").strip()

        if choice == '3':
            self.mission_mode = True
            self.setup_llm_missions()
            return True
        elif choice == '2':
            self.mission_mode = True
            self.setup_missions()
            return True
        else:
            self.mission_mode = False
            return False

    def setup_missions(self):
        """Setup mission sequence"""
        print("\n" + "="*50)
        print("MISSION SETUP")
        print("="*50)
        print("Define your mission sequence. Each mission activates an agent with a specific task.")
        print("\nValid task types:")
        print("  - 'wall' or 'follow_walls' → Wall Following Agent")
        print("  - 'frontier' or 'explore' → Frontier Agent")
        print("  - 'room' → Room Frontier Agent")
        print("  - 'navigate' → A* Navigation Agent (will ask for coordinates)")
        print("  - 'doorway' or 'door' or 'enter' → Doorway Entry Agent")
        print("\nFormat: <agent_id> <task_type>")
        print("Example: '1 wall' or '2 navigate' or '3 explore'")
        print("\nEnter missions (press Enter on empty line to finish):")

        self.missions = []
        mission_count = 1

        while True:
            mission_str = input(f"Mission {mission_count}: ").strip()
            if not mission_str:
                break

            parts = mission_str.split(maxsplit=1)
            if len(parts) < 2:
                print("Invalid format. Use: <agent_id> <task_type>")
                continue

            try:
                agent_id = int(parts[0])
                task = parts[1].lower()

                # Map task to agent type with validation
                agent_type_map = {
                    'navigate': 'navigation',
                    'navigation': 'navigation',
                    'explore': 'frontier',
                    'frontier': 'frontier',
                    'room': 'room',
                    'room_frontier': 'room',
                    'follow_walls': 'wall',
                    'follow': 'wall',
                    'wall': 'wall',
                    'walls': 'wall',
                    'enter': 'doorway',
                    'exit': 'doorway',
                    'door': 'doorway',
                    'doorway': 'doorway'
                }

                agent_type = agent_type_map.get(task, None)

                if agent_type is None:
                    print(f"Unknown task type: '{task}'")
                    print("Valid tasks: wall, frontier/explore, room, navigate, doorway/door")
                    continue

                # Check if navigation task needs coordinates
                goal = None
                if agent_type == 'navigation':
                    current_pos = self.get_current_position() if hasattr(self, 'env') and self.env else (0, 0)
                    print(f"Current position: {current_pos}")
                    x_str = input(f"Target X coordinate [{current_pos[0]}]: ").strip()
                    y_str = input(f"Target Y coordinate [{current_pos[1]}]: ").strip()

                    x = int(x_str) if x_str else current_pos[0]
                    y = int(y_str) if y_str else current_pos[1]
                    goal = (x, y)

                # Store original task name for display
                mission = Mission(agent_id, agent_type, task, goal)
                self.missions.append(mission)

                # Display confirmation
                agent_name_map = {
                    'wall': 'Wall Following Agent',
                    'frontier': 'Frontier Agent',
                    'room': 'Room Frontier Agent',
                    'navigation': 'A* Navigation Agent',
                    'doorway': 'Doorway Entry Agent'
                }
                print(f"  Added: Agent {agent_id} using {agent_name_map[agent_type]}")
                if goal:
                    print(f"         Target: {goal}")

                mission_count += 1

            except ValueError:
                print("Invalid agent ID. Please enter a number.")
                continue

        if self.missions:
            print(f"\n{len(self.missions)} missions configured:")
            for i, m in enumerate(self.missions, 1):
                goal_str = f" to {m.goal}" if m.goal else ""
                agent_name_map = {
                    'wall': 'Wall Agent',
                    'frontier': 'Frontier',
                    'room': 'Room Explorer',
                    'navigation': 'Navigator',
                    'doorway': 'Doorway'
                }
                print(f"  {i}. Agent {m.agent_id} ({agent_name_map.get(m.agent_type, 'Unknown')}): {m.task}{goal_str}")

            # Ask about reset behavior
            reset_option = input("\nReset environment between missions? (y/n) [n]: ").strip().lower()
            self.reset_between_missions = (reset_option == 'y')

            # Ask about auto-advance
            auto_option = input("Auto-advance missions on completion? (y/n) [y]: ").strip().lower()
            self.auto_advance_missions = (auto_option != 'n')
        else:
            print("No missions configured. Switching to single agent mode.")
            self.mission_mode = False

    def get_current_position(self) -> Tuple[int, int]:
        """Get current agent position"""
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'agent_positions'):
            # Always use first agent in environment
            pos = self.env.env.agent_positions[0]
            return (pos[0], pos[1])
        return (0, 0)

    def create_agent_for_mission(self, mission: Mission):
        """Create appropriate agent instance for mission"""
        agent_type = mission.agent_type

        if agent_type == 'wall':
            return LogicalWallAgent(num_agents=self.num_agents)
        elif agent_type == 'frontier':
            return FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
        elif agent_type == 'room':
            return RoomFrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
        elif agent_type == 'navigation':
            agent = AStarNavigationAgent(num_agents=self.num_agents)
            self.manual_goal = mission.goal  # Set the goal for navigation
            return agent
        elif agent_type == 'doorway':
            return DoorwayEntryAgent(num_agents=self.num_agents)
        else:
            # Default to frontier agent if unknown type
            print(f"Warning: Unknown agent type '{agent_type}', defaulting to Frontier Agent")
            return FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )

    def switch_to_next_mission(self):
        """Switch to the next mission in sequence"""
        if not self.mission_mode or not self.missions:
            return False

        # Mark current mission as completed
        if self.current_mission_idx < len(self.missions):
            self.missions[self.current_mission_idx].completed = True

        # Move to next mission
        self.current_mission_idx += 1

        if self.current_mission_idx >= len(self.missions):
            print("\n✓ All missions completed!")
            return False

        # Activate next mission
        next_mission = self.missions[self.current_mission_idx]
        print(f"\n>>> Switching to Mission {self.current_mission_idx + 1}:")
        print(f"    Agent {next_mission.agent_id}: {next_mission.task}")
        if next_mission.goal:
            print(f"    Goal: {next_mission.goal}")

        # Create and set new agent (but we keep using single agent in environment)
        self.active_agent_id = next_mission.agent_id  # Store the logical agent ID
        self.agent = self.create_agent_for_mission(next_mission)
        self.agent.reset()

        return True

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
            self.agent = LogicalWallAgent(
                num_agents=self.num_agents,
            )

        elif choice == '2':
            print("Selected: Frontier Agent")
            self.agent = FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )

        elif choice == '3':
            print("Selected: Room Frontier Agent")
            self.agent = RoomFrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )

        elif choice == '4':
            print("Selected: A* Navigation Agent")
            self.agent = AStarNavigationAgent(
                num_agents=self.num_agents,
            )

        else:
            print("Selected: Doorway Entry Agent")
            self.agent = DoorwayEntryAgent(num_agents=self.num_agents)

    def print_agent_state(self, obs, info, action=None, step=0):
        """Print detailed agent state information"""
        print("\n" + "-"*40)
        print(f"Step {step}")

        # Print mission info if in mission mode
        if self.mission_mode and self.current_mission_idx < len(self.missions):
            mission = self.missions[self.current_mission_idx]
            print(f"Mission {self.current_mission_idx + 1}/{len(self.missions)}: Agent {mission.agent_id} - {mission.task}")

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
        current_pos = None
        # Use agent index 0 since we're simulating multiple agents with a single agent
        agent_idx = 0  # Always use first agent in environment
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'agent_positions'):
            pos = self.env.env.agent_positions[agent_idx]
            print(f"Position: ({pos[0]}, {pos[1]})")
            current_pos = pos
        elif 'positions' in obs:
            pos = obs['positions'][agent_idx]
            print(f"Position: ({pos[0]}, {pos[1]})")
            current_pos = pos

        # Print goal information if available
        if 'goal_position' in obs and obs['goal_position'][0] >= 0:
            goal_pos = obs['goal_position']
            print(f"Goal: ({goal_pos[0]}, {goal_pos[1]})")
            if current_pos is not None:
                distance = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
                print(f"Distance to goal: {distance}")
        elif self.manual_goal:
            goal_pos = self.manual_goal
            print(f"Goal (manual): {goal_pos}")
            if current_pos is not None:
                distance = abs(goal_pos[0] - current_pos[0]) + abs(goal_pos[1] - current_pos[1])
                print(f"Distance to goal: {distance}")

        # Print action if provided
        if action is not None:
            action_val = action[0] if isinstance(action, np.ndarray) else action
            action_names = ['FORWARD', 'LEFT', 'RIGHT', 'STAY']
            print(f"Action: {action_names[action_val]}")

        # Print task-specific info
        if 'wall_coverage' in info:
            print(f"Wall Coverage: {info['wall_coverage']*100:.1f}%")
        if 'room_coverage' in info:
            print(f"Room Coverage: {info['room_coverage']*100:.1f}%")
        if 'exploration_coverage' in info:
            print(f"Exploration: {info['exploration_coverage']*100:.1f}%")

        print("-"*40)

    def change_agent_during_episode(self, obs, info):
        """Allow changing agent during episode or switching to next mission"""
        print("\n" + "="*30)

        if self.mission_mode:
            print("MISSION CONTROL")
            print("="*30)
            if self.current_mission_idx < len(self.missions):
                mission = self.missions[self.current_mission_idx]
                print(f"Current Mission {self.current_mission_idx + 1}: Agent {mission.agent_id} - {mission.task}")
            print("-"*30)
            print("1. Complete current mission & go to next")
            print("2. Skip to specific mission")
            print("3. Switch to manual agent control")
            print("0. Cancel")

            choice = input("\nSelect option (0-3): ").strip()

            if choice == '1':
                if self.switch_to_next_mission():
                    return True
                else:
                    print("No more missions!")
                    return False
            elif choice == '2':
                mission_num = input(f"Enter mission number (1-{len(self.missions)}): ").strip()
                try:
                    idx = int(mission_num) - 1
                    if 0 <= idx < len(self.missions):
                        self.current_mission_idx = idx
                        mission = self.missions[idx]
                        self.active_agent_id = mission.agent_id  # Store logical agent ID
                        self.agent = self.create_agent_for_mission(mission)
                        self.agent.reset()
                        print(f"Switched to Mission {idx + 1}")
                        return True
                except ValueError:
                    pass
                print("Invalid mission number")
                return False
            elif choice == '3':
                self.mission_mode = False
                print("Switched to manual control")
                # Fall through to manual agent selection
            else:
                return False

        # Manual agent selection (original code)
        print("CHANGE AGENT")
        print("="*30)
        print("Current agent:", self.agent.__class__.__name__)

        # Print current position
        current_pos = None
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'agent_positions'):
            current_pos = self.env.env.agent_positions[0]  # Always use index 0
            print(f"Current drone position: ({current_pos[0]}, {current_pos[1]})")

        print("-"*30)
        print("1. Logical Wall Agent")
        print("2. Frontier Agent")
        print("3. Room Frontier Agent")
        print("4. A* Navigation Agent")
        print("5. Doorway Entry Agent")
        print("0. Cancel")

        choice = input("\nSelect new agent (0-5): ").strip()

        if choice == '1':
            self.manual_goal = None
            self.agent = LogicalWallAgent(num_agents=self.num_agents)
            print("Switched to: Logical Wall Agent")
        elif choice == '2':
            self.manual_goal = None
            self.agent = FrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
            print("Switched to: Frontier Agent")
        elif choice == '3':
            self.manual_goal = None
            self.agent = RoomFrontierAgent(
                num_agents=self.num_agents,
                camera_range=self.sensor_params['max_range']
            )
            print("Switched to: Room Frontier Agent")
        elif choice == '4':
            print("\nSwitched to: A* Navigation Agent")
            # Get navigation goal
            current_pos = self.get_current_position()
            print(f"Current position: {current_pos}")
            x_str = input(f"Target X [{current_pos[0]}]: ").strip()
            y_str = input(f"Target Y [{current_pos[1]}]: ").strip()

            x = int(x_str) if x_str else current_pos[0]
            y = int(y_str) if y_str else current_pos[1]
            self.manual_goal = (x, y)

            self.agent = AStarNavigationAgent(num_agents=self.num_agents)
        elif choice == '5':
            self.manual_goal = None
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

        if self.mission_mode:
            print(f"MISSION MODE: {len(self.missions)} missions configured")
            if self.missions and self.current_mission_idx < len(self.missions):
                # Start first mission
                mission = self.missions[self.current_mission_idx]
                self.active_agent_id = mission.agent_id  # Store logical agent ID
                self.agent = self.create_agent_for_mission(mission)
                print(f"Starting Mission 1: Agent {mission.agent_id} - {mission.task}")
        else:
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
        print("  A     = change agent/mission")
        print("  M     = complete mission (mission mode)")
        print("  V     = toggle verbose")
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
        self.verbose = False
        agent_error = False
        mission_completed = False

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
                            self.manual_goal = None
                            # Reset mission progress
                            if self.mission_mode:
                                self.current_mission_idx = 0
                                if self.missions:
                                    mission = self.missions[0]
                                    self.active_agent_id = mission.agent_id - 1
                                    self.agent = self.create_agent_for_mission(mission)
                                    self.agent.reset()
                            print("\nEPISODE RESET")
                        elif event.key == pygame.K_s:
                            # Print current state
                            self.print_agent_state(obs, info, step=step)
                        elif event.key == pygame.K_a:
                            # Change agent/mission
                            if self.change_agent_during_episode(obs, info):
                                agent_error = False
                                if paused:
                                    print("Agent changed! Press SPACE to resume.")
                                else:
                                    print("Agent changed! Continuing...")
                        elif event.key == pygame.K_m:
                            # Complete current mission (mission mode only)
                            if self.mission_mode:
                                print("\nCompleting current mission...")
                                if self.switch_to_next_mission():
                                    mission_completed = False
                                else:
                                    print("All missions completed!")
                                    done = True
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
                    # Inject manual goal into observations if set and using A* agent
                    if self.manual_goal and isinstance(self.agent, AStarNavigationAgent):
                        obs_with_goal = obs.copy()
                        obs_with_goal['goal_position'] = np.array(self.manual_goal)
                        actions = self.agent.get_actions(obs_with_goal, info)
                    else:
                        actions = self.agent.get_actions(obs, info)

                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(actions)
                    total_reward += reward
                    done = terminated or truncated

                    # Check for mission completion in mission mode
                    if self.mission_mode and not mission_completed:
                        # Check multiple completion conditions
                        completion_detected = False

                        # Check task_status in info
                        if 'task_status' in info and info['task_status'] == 1:
                            completion_detected = True

                        # Check agent execution state if available
                        elif hasattr(self.agent, 'execution_state'):
                            state = self.agent.execution_state
                            state_str = state.value if hasattr(state, 'value') else str(state)
                            if 'completed' in state_str.lower() or 'success' in state_str.lower():
                                completion_detected = True

                        # Check for specific task completions
                        elif 'wall_coverage' in info and info['wall_coverage'] > 0.95:
                            completion_detected = True
                        elif 'room_coverage' in info and info['room_coverage'] > 0.95:
                            completion_detected = True
                        elif 'has_passed_through' in info and info['has_passed_through']:
                            completion_detected = True

                        # Check if navigation agent reached goal
                        elif isinstance(self.agent, AStarNavigationAgent) and self.manual_goal:
                            current_pos = self.get_current_position()
                            if current_pos == self.manual_goal:
                                completion_detected = True

                        if completion_detected and self.auto_advance_missions:
                            mission_completed = True
                            print(f"\n✓ Mission {self.current_mission_idx + 1} completed!")

                            if self.switch_to_next_mission():
                                mission_completed = False

                                # Reset environment if requested
                                if self.reset_between_missions:
                                    obs, info = self.env.reset()
                                    step = 0  # Reset step counter
                                    print("Environment reset for new mission")
                            else:
                                print("All missions completed!")
                                done = True

                    # Print state if verbose mode is on
                    if self.verbose:
                        self.print_agent_state(obs, info, actions, step=step)

                    # Print status periodically
                    print_interval = 100 if self.verbose else 50
                    if step % print_interval == 0:
                        action = actions[0] if isinstance(actions, np.ndarray) else actions
                        action_names = ['FORWARD', 'LEFT', 'RIGHT', 'STAY']
                        print(f"Step {step:4d} | Action: {action_names[action]:8s} | "
                              f"Reward: {total_reward:7.2f}", end='')

                        # Add mission info if in mission mode
                        if self.mission_mode and self.current_mission_idx < len(self.missions):
                            print(f" | Mission {self.current_mission_idx + 1}/{len(self.missions)}", end='')

                        # Add task-specific info
                        if 'wall_coverage' in info:
                            print(f" | Wall: {info['wall_coverage']*100:.1f}%", end='')
                        if 'room_coverage' in info:
                            print(f" | Room: {info['room_coverage']*100:.1f}%", end='')

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

            if self.mission_mode:
                completed_count = sum(1 for m in self.missions if m.completed)
                print(f"Missions Completed: {completed_count}/{len(self.missions)}")
            else:
                print(f"Agent: {self.agent.__class__.__name__}")

            print(f"Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")

            if 'task_status' in info:
                status = ['IN_PROGRESS', 'SUCCESS', 'FAILURE'][info['task_status']]
                print(f"Status: {status}")

            print("="*50)

            # Ask if user wants to continue
            print("\nWhat would you like to do?")
            print("1. Run again with same configuration")
            print("2. Change configuration")
            print("3. Quit")

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                # Reset mission progress for new run
                if self.mission_mode:
                    self.current_mission_idx = 0
                    for m in self.missions:
                        m.completed = False
                return 'continue_same'
            elif choice == '2':
                return 'change_config'
            else:
                return 'quit'

        return 'interrupted' if not self.running else 'continue_same'

    def run(self):
        """Main execution loop"""
        print("\n" + "="*50)
        print("SLAM ENVIRONMENT INTERACTIVE VIEWER")
        print("with LLM Mission Planning")
        print("="*50)

        while True:
            # Configure everything
            self.select_map_option()
            self.select_sensor_config()

            # Ask for operation mode
            is_mission_mode = self.select_operation_mode()

            # Ask for number of agents
            num_str = input("\nNumber of agents (1-5) [1]: ").strip()
            if num_str and num_str.isdigit():
                self.num_agents = min(5, max(1, int(num_str)))
            else:
                self.num_agents = 1
            print(f"Using {self.num_agents} agent(s)")

            # Select environment
            self.select_environment()

            # Select initial agent if not in mission mode
            if not is_mission_mode:
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
                    # Clear state
                    self.manual_goal = None
                    self.missions = []
                    self.current_mission_idx = 0
                    self.mission_mode = False
                    break
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