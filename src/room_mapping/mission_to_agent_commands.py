#!/usr/bin/env python3
"""
mission_to_agent_commands.py - Monitors for new missions and converts them to agent commands
Watches for missions from web_mission_server.py and generates step-by-step agent execution plans
"""

import subprocess
import tempfile
import os
import json
import time
import sys
from pathlib import Path

# File paths for mission exchange
MISSION_FILE = "current_mission.txt"
AGENT_COMMANDS_FILE = "agent_commands.txt"

PROMPT = """You are a drone controller that converts navigation missions into step-by-step agent commands.

CURRENT HOUSE MAP:
{house_json}

NAVIGATION MISSION TO EXECUTE:
{mission_instruction}

YOUR SPECIALIZED AGENTS:
- DoorAgent: Opens doors, enters rooms, and exits rooms (ONLY for rooms with doors!)
- NavigationAgent: Moves drone to specific rooms or locations  
- ScanAgent: Scans the room and makes sure not to leave it.
- WallAgent: Follows walls and detects wall boundaries for navigation

CRITICAL OUTPUT RULES:
1. OUTPUT ONLY NUMBERED STEPS - NO EXPLANATIONS, NO NOTES, NO COMMENTARY
2. Each step must be: "Activate [AgentName] to [action]"
3. DO NOT repeat navigation after reaching destination
4. DO NOT add verification steps - scanning IS the verification
5. Once you scan a room, you're done with that room (just exit if it has doors)
6. NEVER describe what you'll find - the scan will discover that

DOOR RULES:
- Check JSON: If "doors": [] → NO DOORS, never use DoorAgent
- Check JSON: If "doors": ["door_name"] → HAS DOORS, must use DoorAgent
- Open spaces and hallways typically have no doors

CORRECT SEQUENCE PATTERN:
For room WITH doors:
1. Move to room entrance (NavigationAgent)
2. Enter room (DoorAgent)
3. Scan room (ScanAgent)
4. Exit room (DoorAgent)

For room WITHOUT doors:
1. Move to room (NavigationAgent)
2. Scan room (ScanAgent)

WRONG PATTERNS TO AVOID:
Adding extra movement after already at destination
Describing what will be found during scan
Adding "verification" steps after scanning
Repeating the mission instructions in the steps
Using DoorAgent on rooms with empty doors array

EXAMPLE (for illustration only — actual behavior depends entirely on the JSON house map and room structure):

Example Input:
"From Open Space, walk into hallway. MAMAD is at end on right. Enter MAMAD to find refrigerator"

Example JSON:
Open Space: "doors": []
hallway: "doors": []
MAMAD: "doors": ["door1"]

Correct Output Sequence:
1. Activate NavigationAgent to navigate from Open Space to hallway
2. Activate NavigationAgent to navigate through the hallway to the MAMAD entrance
3. Activate DoorAgent to open and enter the MAMAD
4. Activate ScanAgent to scan the MAMAD interior to find the refrigerator
5. Activate DoorAgent to exit the MAMAD

Note: This is only an example. The actual sequence must always depend on the real JSON data, the presence or absence of doors, and the relative positions of rooms in the specific house map.


Generate ONLY the numbered steps:"""


def load_house_data():
    """Load and process house data with room positions"""
    try:
        with open("data/unified_rooms.json", 'r') as f:
            house_data = json.load(f)

        # Create a more detailed summary for the LLM
        simplified = {"rooms": {}}
        for room_name, room_info in house_data.get('rooms', {}).items():
            # Get camera position to understand room location
            pos = room_info.get('camera_position', [0, 0, 0])

            # Extract key objects (first 8 for context)
            objects = [obj['type'] for obj in room_info.get('objects', [])[:8]]

            simplified["rooms"][room_name] = {
                "camera_position": pos,
                "location_description": f"x={pos[0]:.1f}, z={pos[2]:.1f}" if len(pos) >= 3 else "unknown",
                "contains_objects": objects,
                "connected_via_doors": room_info.get('doors', [])
            }

        return simplified
    except Exception as e:
        print(f"Warning: Could not load unified_rooms.json: {e}")
        # Return empty structure if file doesn't exist
        return {"rooms": {}}


def ask_ollama(house_json, mission_instruction):
    """Send prompt to Ollama and get response"""
    full_prompt = PROMPT.format(
        house_json=house_json,
        mission_instruction=mission_instruction
    )

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(full_prompt)
        temp_file = f.name

    try:
        # Send file content to Ollama
        cmd = f"cat {temp_file} | ollama run llama3.2:3b"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        response = result.stdout.strip()

        # Clean up temp file
        os.unlink(temp_file)

        return response
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return f"Error: {e}"


def monitor_missions():
    """Monitor for new missions and generate agent commands"""
    print("=" * 70)
    print("MISSION TO AGENT COMMAND MONITOR")
    print("Watching for missions from web server...")
    print("=" * 70)

    last_mission = None
    last_modified = 0

    while True:
        try:
            # Check if mission file exists and has changed
            if os.path.exists(MISSION_FILE):
                current_modified = os.path.getmtime(MISSION_FILE)

                if current_modified > last_modified:
                    # Small delay to ensure file is fully written
                    time.sleep(0.2)

                    # Read the new mission
                    with open(MISSION_FILE, 'r') as f:
                        mission = f.read().strip()

                    if mission and mission != last_mission:
                        print(f"\n[{time.strftime('%H:%M:%S')}] New mission received!")
                        print("-" * 70)
                        print("Mission:", mission[:200] + "..." if len(mission) > 200 else mission)
                        print("-" * 70)

                        # Clear old agent commands immediately
                        # if os.path.exists(AGENT_COMMANDS_FILE):
                            # os.remove(AGENT_COMMANDS_FILE)

                        # Load current house data
                        house_data = load_house_data()
                        if not house_data or not house_data.get('rooms'):
                            print(" No room data available - using general navigation")
                            house_data = {"rooms": {}}

                        # Create JSON representation
                        house_json = json.dumps(house_data, indent=2)

                        # Generate agent commands
                        print("🤖 Generating agent commands...")
                        agent_commands = ask_ollama(house_json, mission)

                        # Save agent commands to file
                        with open(AGENT_COMMANDS_FILE, 'w') as f:
                            f.write(agent_commands)

                        print("\n Agent Execution Plan:")
                        print("=" * 70)
                        print(agent_commands)
                        print("=" * 70)
                        print(f"✓ Commands saved to {AGENT_COMMANDS_FILE}")

                        last_mission = mission
                        last_modified = current_modified

            # Check every 0.5 seconds
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nShutting down monitor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


def main():
    # Clean up any old files at startup
    for file in [AGENT_COMMANDS_FILE, MISSION_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f" Cleaned old {file}")

    # Check if unified_rooms.json exists
    if not os.path.exists("data/unified_rooms.json"):
        print(" Warning: unified_rooms.json not found.")
        print("  The agent planner will work but without room awareness.")
        print("  Run pixel_room_mapper.py first for best results.")
        print()
    else:
        house_data = load_house_data()
        if house_data and house_data.get('rooms'):
            num_rooms = len(house_data.get('rooms', {}))
            print(f" Loaded house map with {num_rooms} rooms")

    # Start monitoring
    monitor_missions()


if __name__ == "__main__":
    main()