#!/usr/bin/env python3
"""
mission_executor_agent.py - Super Agent that executes missions by orchestrating other agents

This agent:
1. Reads mission commands from the LLM output file
2. Parses instructions to determine which agent to activate
3. Manages agent switching and coordination
4. Translates room names to grid coordinates
5. Monitors agent completion status
"""

import numpy as np
import json
import re
import os
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from src.agents.base_agent import BaseSLAMAgent, AgentState
from src.agents.a_star_navigation_agent import AStarNavigationAgent
from src.agents.doorway_traversal_agent import DoorwayEntryAgent
from src.agents.wall_following_agent import WallFollowingAgent
from src.environments.base.constants import Action


class AgentType(Enum):
    """Types of agents that can be activated"""
    NAVIGATION = "NavigationAgent"
    DOOR = "DoorAgent"
    SCAN = "ScanAgent"
    WALL = "WallAgent"
    UNKNOWN = "Unknown"


@dataclass
class MissionStep:
    """Single step in the mission"""
    step_number: int
    agent_type: AgentType
    action_description: str
    target_location: Optional[str] = None
    completed: bool = False


class RoomScanner:
    """Simple scanning behavior - spiral or systematic coverage"""

    def __init__(self):
        self.scan_pattern = []
        self.scan_index = 0
        self.scan_steps = 0
        self.max_scan_steps = 30

    def get_scan_action(self, pos: Tuple[int, int], facing: int) -> int:
        """Execute a scanning pattern in the current room"""
        self.scan_steps += 1

        if self.scan_steps >= self.max_scan_steps:
            return Action.STAY

        # Simple scanning pattern: turn right every 3 forward moves
        if self.scan_steps % 4 == 0:
            return Action.TURN_RIGHT
        else:
            return Action.FORWARD

    def reset(self):
        self.scan_pattern = []
        self.scan_index = 0
        self.scan_steps = 0

    def is_complete(self) -> bool:
        return self.scan_steps >= self.max_scan_steps


class MissionExecutorAgent(BaseSLAMAgent):
    """
    Super agent that reads LLM mission instructions and orchestrates sub-agents.

    Mission flow:
    1. Load mission from agent_commands.txt
    2. Parse steps and identify agents needed
    3. Execute each step by activating appropriate agent
    4. Monitor completion and switch agents
    5. Update progress
    """

    def __init__(self, num_agents: int = 1):
        super().__init__(num_agents)

        # Mission management
        self.mission_steps: List[MissionStep] = []
        self.current_step_index = 0
        self.mission_loaded = False

        # Room mapping
        self.room_locations = {}
        self.load_room_data()

        # Sub-agents
        self.navigation_agent = AStarNavigationAgent(num_agents)
        self.door_agent = DoorwayEntryAgent(num_agents)
        self.wall_agent = WallFollowingAgent(num_agents)
        self.scanner = RoomScanner()

        # Current active agent
        self.active_agent = None
        self.active_agent_type = None

        # Goal tracking
        self.final_goal = None  # Store the final destination (e.g., door position)

        # Mission files
        self.mission_file = "agent_commands.txt"
        self.last_mission_time = 0

    def load_room_data(self):
        """Load room locations and boundaries from unified_rooms.json"""
        try:
            with open("unified_rooms.json", 'r') as f:
                data = json.load(f)

            self.room_data = {}  # Store full room data

            # Extract room data including boundaries and doors
            for room_name, room_info in data.get('rooms', {}).items():
                bbox = room_info.get('bbox', [])
                if len(bbox) >= 4:
                    # Calculate room center
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    self.room_locations[room_name.lower()] = (center_x, center_y)

                    # Store full room data
                    self.room_data[room_name.lower()] = {
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'doors': room_info.get('doors', []),
                        'camera': room_info.get('camera_position', [])
                    }

                # Also store camera position as alternative
                cam_pos = room_info.get('camera_position', [])
                if len(cam_pos) >= 2:
                    alt_name = f"{room_name.lower()}_camera"
                    self.room_locations[alt_name] = (cam_pos[0], cam_pos[1])

            print(f"Loaded {len(self.room_locations)} room locations")

        except Exception as e:
            print(f"Could not load room data: {e}")
            self.room_locations = {}
            self.room_data = {}

    def load_mission(self) -> bool:
        """Load and parse mission from agent_commands.txt"""
        try:
            # Check if mission file exists and is new
            if not os.path.exists(self.mission_file):
                return False

            file_time = os.path.getmtime(self.mission_file)
            if file_time <= self.last_mission_time:
                return False  # No new mission

            with open(self.mission_file, 'r') as f:
                content = f.read().strip()

            if not content:
                return False

            # Parse mission steps
            self.mission_steps = self.parse_mission_steps(content)
            self.current_step_index = 0
            self.mission_loaded = True
            self.last_mission_time = file_time

            print(f"Loaded mission with {len(self.mission_steps)} steps")
            for step in self.mission_steps:
                print(f"  Step {step.step_number}: {step.agent_type.value} - {step.action_description}")

            return True

        except Exception as e:
            print(f"Error loading mission: {e}")
            return False

    def parse_mission_steps(self, content: str) -> List[MissionStep]:
        """Parse LLM output into mission steps"""
        steps = []

        # Match numbered steps like "1. Activate NavigationAgent to navigate to hallway"
        pattern = r'(\d+)\.\s*Activate\s+(\w+)\s+to\s+(.+)'

        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                step_num = int(match.group(1))
                agent_name = match.group(2)
                action = match.group(3).strip()

                # Determine agent type
                agent_type = AgentType.UNKNOWN
                if 'navigation' in agent_name.lower():
                    agent_type = AgentType.NAVIGATION
                elif 'door' in agent_name.lower():
                    agent_type = AgentType.DOOR
                elif 'scan' in agent_name.lower():
                    agent_type = AgentType.SCAN
                elif 'wall' in agent_name.lower():
                    agent_type = AgentType.WALL

                # Extract target location if mentioned
                target = self.extract_target_location(action)

                steps.append(MissionStep(
                    step_number=step_num,
                    agent_type=agent_type,
                    action_description=action,
                    target_location=target
                ))

        return steps

    def extract_target_location(self, action: str) -> Optional[str]:
        """Extract room/location name from action description"""
        # Look for patterns like "to hallway", "to MAMAD", "from Open Space to hallway"
        patterns = [
            r'to\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'into\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'enter\s+(?:the\s+)?(\w+(?:\s+\w+)?)',
            r'through\s+(?:the\s+)?(\w+(?:\s+\w+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                return location

        return None

    def get_closest_room_tile(self, room_name: str, current_pos: Tuple[int, int],
                              observations: Dict) -> Optional[Tuple[int, int]]:
        """Get the closest accessible tile in a room to current position"""
        room_lower = room_name.lower()
        if room_lower not in self.room_data:
            # Fallback to center if no room data
            return self.get_room_coordinates(room_name)

        room_info = self.room_data[room_lower]
        bbox = room_info['bbox']
        if len(bbox) < 4:
            return room_info.get('center')

        # Get the observed map to check for accessible tiles
        global_map = observations.get('global_map')
        if global_map is None:
            return room_info.get('center')

        # Find all accessible tiles in the room
        accessible_tiles = []
        x1, y1, x2, y2 = bbox

        for y in range(max(0, y1), min(global_map.shape[0], y2)):
            for x in range(max(0, x1), min(global_map.shape[1], x2)):
                # Check if tile is accessible (free space or unknown)
                tile_value = global_map[y, x]
                if tile_value in [-1, 0, 2, 4]:  # Unknown, free_space, entry_point, door_open
                    accessible_tiles.append((x, y))

        if not accessible_tiles:
            # No accessible tiles found, use room center
            return room_info.get('center')

        # Find closest accessible tile to current position
        min_dist = float('inf')
        closest_tile = None

        for tile in accessible_tiles:
            dist = abs(tile[0] - current_pos[0]) + abs(tile[1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_tile = tile

        return closest_tile

    def get_door_position(self, room_name: str) -> Optional[Tuple[int, int]]:
        """Get the door position for a room"""
        room_lower = room_name.lower()

        # Handle variations in room names
        room_mapping = {
            "moshe": "moshe's office",
            "moshe's": "moshe's office",
            "inbal": "inbal's office",
            "inbal's": "inbal's office",
            "yaniv": "yaniv oren's office",
            "yaniv oren": "yaniv oren's office",
            "yaniv oren's": "yaniv oren's office",
            "mamad": "mamad",
            "open space": "open space",
            "open": "open space",
            "hallway": "hallway",
            "hall": "hallway"
        }

        # Map to full room name if needed
        if room_lower in room_mapping:
            room_lower = room_mapping[room_lower]

        if room_lower not in self.room_data:
            # Try to find partial match
            for room_key in self.room_data.keys():
                if room_lower in room_key or room_key in room_lower:
                    room_lower = room_key
                    break
            else:
                return None

        room_info = self.room_data[room_lower]
        doors = room_info.get('doors', [])

        if doors and len(doors) >= 2:
            # Doors are stored as [x, y] coordinates in the JSON
            door_x = doors[0]
            door_y = doors[1]
            print(f"Found door for {room_name} at position ({door_x}, {door_y})")
            return (door_x, door_y)

        print(f"No door found for room: {room_name}")
        return None

    def get_room_coordinates(self, room_name: str) -> Optional[Tuple[int, int]]:
        """Get grid coordinates for a room name (returns center by default)"""
        if not room_name:
            return None

        # Try exact match first
        room_lower = room_name.lower()
        if room_lower in self.room_locations:
            return self.room_locations[room_lower]

        # Try partial match
        for room_key, coords in self.room_locations.items():
            if room_lower in room_key or room_key in room_lower:
                return coords

        return None

    def is_near_door(self, current_pos: Tuple[int, int], room_name: str,
                     max_distance: int = 2) -> bool:
        """Check if current position is near a door"""
        door_pos = self.get_door_position(room_name)
        if not door_pos:
            return True  # If no door info, assume we're ready

        dist = abs(current_pos[0] - door_pos[0]) + abs(current_pos[1] - door_pos[1])
        return dist <= max_distance

    def get_actions(self, observations: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        """Execute mission by coordinating sub-agents"""
        try:
            # Load mission if not loaded or if there's a new one
            if not self.mission_loaded or os.path.exists(self.mission_file):
                if self.load_mission():
                    print("New mission loaded!")

            # If no mission, just explore
            if not self.mission_steps:
                return self.wall_agent.get_actions(observations, info)

            # Check if mission is complete
            if self.current_step_index >= len(self.mission_steps):
                self.execution_state = AgentState.COMPLETED
                print("Mission complete!")
                return np.array([Action.STAY])

            # Get current step
            current_step = self.mission_steps[self.current_step_index]
            current_pos = tuple(observations['positions'][0])

            # Special handling for door agent - check proximity first
            if current_step.agent_type == AgentType.DOOR and not self.active_agent:
                # Check if we need to navigate to door first
                if current_step.target_location:
                    if not self.is_near_door(current_pos, current_step.target_location):
                        # Create a temporary navigation step
                        print(f"Need to navigate to door first for room: {current_step.target_location}")
                        self.navigation_agent.reset()
                        self.active_agent = self.navigation_agent
                        self.active_agent_type = AgentType.NAVIGATION

                        # Set goal to door position
                        door_coords = self.get_door_position(current_step.target_location)
                        if door_coords:
                            observations['goal_position'] = np.array(door_coords)
                            self.final_goal = door_coords  # Store final destination
                            print(f"Navigating to door at {door_coords}")
                        else:
                            # Try to get close to room entrance
                            coords = self.get_closest_room_tile(current_step.target_location, current_pos, observations)
                            if coords:
                                observations['goal_position'] = np.array(coords)
                                self.final_goal = coords  # Store final destination
                    else:
                        # We're close enough, activate door agent
                        self.switch_agent(current_step, observations)

            # Check if we need to switch agents (for non-door agents or if not set)
            elif self.active_agent_type != current_step.agent_type and not self.active_agent:
                self.switch_agent(current_step, observations)

            # Execute current agent
            action = self.execute_active_agent(observations, info, current_step)

            # Check if current agent is done
            if self.is_agent_complete():
                # Special case: if we were navigating to a door, now activate the door agent
                if self.active_agent_type == AgentType.NAVIGATION and current_step.agent_type == AgentType.DOOR:
                    if self.is_near_door(current_pos, current_step.target_location):
                        print("Reached door position, now activating DoorAgent")
                        self.door_agent.reset()
                        self.active_agent = self.door_agent
                        self.active_agent_type = AgentType.DOOR
                        # Don't advance step yet, let door agent complete
                    else:
                        # Still not close enough, keep navigating
                        print("Navigation complete but not near door yet, continuing...")
                        self.active_agent = None
                        self.active_agent_type = None
                else:
                    # Normal completion
                    print(f"Step {current_step.step_number} complete: {current_step.action_description}")
                    current_step.completed = True
                    self.current_step_index += 1
                    self.active_agent = None
                    self.active_agent_type = None
                    self.final_goal = None  # Clear final goal when step completes

            return action

        except Exception as e:
            print(f"Mission executor error: {e}")
            self.set_error(str(e))
            return np.array([Action.STAY])

    def switch_agent(self, step: MissionStep, observations: Dict):
        """Switch to appropriate agent for current step"""
        current_pos = tuple(observations['positions'][0])
        print(f"Activating {step.agent_type.value} for: {step.action_description}")

        if step.agent_type == AgentType.NAVIGATION:
            self.navigation_agent.reset()
            self.active_agent = self.navigation_agent

            # Set goal position based on target room
            if step.target_location:
                # First, check if the target room has a door
                door_coords = self.get_door_position(step.target_location)

                if door_coords:
                    # Room has a door - navigate to door entrance
                    print(f"Room '{step.target_location}' has door at {door_coords}, navigating there")
                    observations['goal_position'] = np.array(door_coords)
                else:
                    # No door - navigate to closest tile in room
                    coords = self.get_closest_room_tile(step.target_location, current_pos, observations)
                    if coords:
                        print(f"Room '{step.target_location}' has no door, navigating to closest tile at {coords}")
                        observations['goal_position'] = np.array(coords)
                    else:
                        # Fallback to room center
                        coords = self.get_room_coordinates(step.target_location)
                        if coords:
                            print(f"Using room center for '{step.target_location}' at {coords}")
                            observations['goal_position'] = np.array(coords)

        elif step.agent_type == AgentType.DOOR:
            # Check if we're close enough to a door
            if step.target_location:
                door_pos = self.get_door_position(step.target_location)
                if door_pos and not self.is_near_door(current_pos, step.target_location, max_distance=3):
                    # Need to navigate to door first
                    print(f"Not near door yet (need to be within 3 tiles of {door_pos}), navigating there first...")
                    self.navigation_agent.reset()
                    self.active_agent = self.navigation_agent
                    self.active_agent_type = AgentType.NAVIGATION  # Stay in navigation mode
                    observations['goal_position'] = np.array(door_pos)
                    self.final_goal = door_pos  # Store final destination
                    return  # Don't switch to door agent yet

            # We're close enough, activate door agent
            print(f"Close enough to door, activating DoorAgent")
            self.door_agent.reset()
            self.active_agent = self.door_agent

        elif step.agent_type == AgentType.WALL:
            self.wall_agent.reset()
            self.active_agent = self.wall_agent

        elif step.agent_type == AgentType.SCAN:
            self.scanner.reset()
            self.active_agent = self.scanner

        self.active_agent_type = step.agent_type

    def execute_active_agent(self, observations: Dict, info: Dict, step: MissionStep) -> np.ndarray:
        """Execute the currently active agent"""
        current_pos = tuple(observations['positions'][0])

        if self.active_agent_type == AgentType.SCAN:
            # Use scanner behavior
            facing = observations['facings'][0]
            action = self.scanner.get_scan_action(current_pos, facing)
            return np.array([action])

        elif self.active_agent:
            # Use sub-agent
            if self.active_agent_type == AgentType.NAVIGATION and step.target_location:
                # For navigation, always use closest tile in target room
                coords = self.get_closest_room_tile(step.target_location, current_pos, observations)
                if coords:
                    observations['goal_position'] = np.array(coords)
                    self.final_goal = coords  # Store final destination
                else:
                    # Fallback to room center
                    coords = self.get_room_coordinates(step.target_location)
                    if coords:
                        observations['goal_position'] = np.array(coords)
                        self.final_goal = coords  # Store final destination

            return self.active_agent.get_actions(observations, info)

        return np.array([Action.STAY])

    def is_agent_complete(self) -> bool:
        """Check if current agent has completed its task"""
        if self.active_agent_type == AgentType.SCAN:
            return self.scanner.is_complete()

        elif self.active_agent and hasattr(self.active_agent, 'execution_state'):
            return self.active_agent.execution_state == AgentState.COMPLETED

        return False

    def reset(self):
        """Reset all state"""
        super().reset()
        self.mission_steps = []
        self.current_step_index = 0
        self.mission_loaded = False
        self.active_agent = None
        self.active_agent_type = None

        # Reset all sub-agents
        self.navigation_agent.reset()
        self.door_agent.reset()
        self.wall_agent.reset()
        self.scanner.reset()

    def get_metrics(self) -> Dict[str, Any]:
        """Get mission execution metrics"""
        metrics = super().get_metrics()

        completed_steps = sum(1 for step in self.mission_steps if step.completed)

        metrics.update({
            'mission_loaded': self.mission_loaded,
            'total_steps': len(self.mission_steps),
            'completed_steps': completed_steps,
            'current_step': self.current_step_index,
            'active_agent': self.active_agent_type.value if self.active_agent_type else None,
            'current_action': self.mission_steps[self.current_step_index].action_description
            if self.current_step_index < len(self.mission_steps) else None
        })

        return metrics