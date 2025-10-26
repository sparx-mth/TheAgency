#!/usr/bin/env python3
"""
llm_mission_processor.py - Dedicated LLM processor for mission generation
Monitors for task requests and generates mission instructions using LLM
Can run on a separate computer from the web server
"""

import json
import subprocess
import tempfile
import os
import time

# File paths for communication
TASK_REQUEST_FILE = "task_request.json"
MISSION_RESPONSE_FILE = "mission_response.txt"
MISSION_FILE = "current_mission.txt"

# Your existing PROMPT - copy it here
PROMPT = """You are a mission planner for an autonomous drone that navigates houses. The drone needs clear navigation instructions based on a house map and user requests.

CRITICAL JSON READING RULES:
- You MUST read the JSON map EXACTLY as provided - every object listed is real, nothing else exists
- NEVER claim to see objects that are not explicitly in the "objects" field of each room
- If you say "I see [object]" - that object MUST exist in the JSON with that exact name
- DO NOT hallucinate, imagine, or invent ANY objects not in the JSON data
- Before saying any object exists, verify it's actually in the JSON "objects" lists
- Room names may have variations: "Inbal's room" = "Inbal's Office", "Moshe's room" = "Moshe's Office", etc.
- Check the "doors" array: empty array [] means no doors (open access), coordinates [x,y] indicate door position

CONTEXT:
- The drone's starting position is at grid coordinates [27,34] which is INSIDE the Open Space room
- The drone is ALREADY IN the Open Space - no navigation needed for objects in Open Space
- Open Space and hallways have NO DOORS (empty doors array) - you can walk directly into/through them
- You must understand the house layout using bbox coordinates [x1,y1,x2,y2] to determine room positions
- Use bbox coordinates to determine if a room is at the beginning, middle, or end of a hallway
- Avoid confusing compass directions - use relative terms like "straight ahead", "to your left", "to your right"

HOUSE MAP (JSON format showing all rooms and their objects):
{house_json}

USER TASK: {user_task}

YOUR JOB:
Generate human-friendly navigation instructions that describe:
1. How to navigate from room to room to reach the target room
2. Where exactly to find the object within that room (near what other objects the other objects must be in the json)

SYNONYM HANDLING AND ROOM MATCHING:
- Room name variations: "Inbal's room" = "Inbal's Office" etc.
- Object synonyms: "couch" = "sofa", "weapon" = "gun", "monitor" = "screen", etc.
- Always check if the requested item or room might be listed under a similar or related name

NAVIGATION INSTRUCTION FORMAT:

1. If object or its SYNONYM EXISTS in map:
   "I see that the [object/synonym] is in [room name]. From your position in the Open Space, walk into the hallway. [Room name] is [at the beginning/middle/end] of the hallway on your [left/right]. Once you enter [room name], you will find the [object] [specific location: next to/near/beside which other objects in that room]."

2. If object DOESN'T EXIST in any form:
   "I don't see [object] or anything similar in the map. Based on typical house layouts, this would usually be found in a [typical room type]. The available rooms are: [list actual rooms]. You should explore [suggest most likely room based on room names] by [describe how to get there]."

3. For room navigation:
   If room EXISTS: "I see [room name/variation]. From your position in the Open Space, walk into the hallway. [Room name] is [at the beginning/middle/end] of the hallway on your [left/right]. [Additional landmark info if helpful]."

CRITICAL NAVIGATION RULES:
- The drone STARTS IN Open Space - never tell it to navigate to Open Space from Open Space
- For objects IN Open Space: say "right here where you are" or "in your current location"
- For "find all [object]" requests: list EVERY instance of that object in ALL rooms, not just one
- Describe navigation using simple, human-friendly terms: "straight ahead", "on your left/right", "at the beginning/middle/end"
- NEVER say "exit through the door" for Open Space or hallway - they have no doors
- Use bbox coordinates to accurately describe room positions:
  * Lower y-values = beginning of hallway (closer to entrance)
  * Higher y-values = end of hallway (further from entrance)
- Always specify which room contains the object
- Describe object location relative to other objects in that same room
- ONLY mention rooms and objects that actually exist in the JSON

IMPORTANT: Output ONLY the navigation instruction as a single flowing paragraph. Do NOT include any introduction, explanation, clarification, or numbered lists. Just write the complete instruction directly in natural, human-friendly language.

Generate the navigation instruction:"""


def load_house_data():
    """Load and format house data clearly for LLM"""
    try:
        with open("data/unified_rooms.json", 'r') as f:
            house_data = json.load(f)

        # Create clear structure showing rooms and their objects
        simplified = {
            "available_rooms": list(house_data.get('rooms', {}).keys()),
            "rooms": {}
        }

        for room_name, room_info in house_data.get('rooms', {}).items():
            # Extract just the object types (names) for each room
            object_types = []
            for obj in room_info.get('objects', []):
                if 'type' in obj:
                    object_types.append(obj['type'])

            # Make it clear what's in each room
            simplified["rooms"][room_name] = {
                "bbox": room_info.get('bbox'),
                "objects": object_types,
                "object_count": len(object_types),
                "doors": room_info.get('doors', [])
            }

        # Add summary for clarity
        simplified["summary"] = {
            "total_rooms": len(simplified["rooms"]),
            "total_objects": sum(room["object_count"] for room in simplified["rooms"].values())
        }

        return simplified
    except Exception as e:
        print(f"Error loading house data: {e}")
        return None


def ask_ollama(house_json, user_task):
    """Send prompt to Ollama and get response"""
    full_prompt = PROMPT.format(house_json=house_json, user_task=user_task)

    # Write prompt to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(full_prompt)
        temp_file = f.name

    try:
        cmd = f"cat {temp_file} | ollama run llama3.1:8b"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        response = result.stdout.strip()
        os.unlink(temp_file)
        return response
    except Exception as e:
        os.unlink(temp_file)
        return f"Error: {e}"


def clean_ollama_cache():
    """Clean Ollama cache and stop running models (run once at startup)."""
    cleanup_cmds = [
        "ollama stop all || true",
        "rm -rf ~/.ollama/cache/* || true",
        "rm -rf ~/.ollama/data/tmp/* || true"
    ]
    for cmd in cleanup_cmds:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Ollama cache cleaned before startup.")


def process_loop():
    """Main processing loop"""
    print("=" * 60)
    print("LLM MISSION PROCESSOR")
    print("=" * 60)
    print("Monitoring for task requests...")
    print(f"Reading from: {TASK_REQUEST_FILE}")
    print(f"Writing to: {MISSION_RESPONSE_FILE}")
    print("-" * 60)

    last_modified = 0

    while True:
        try:
            # Check if task request file exists and has changed
            if os.path.exists(TASK_REQUEST_FILE):
                current_modified = os.path.getmtime(TASK_REQUEST_FILE)

                if current_modified > last_modified:
                    # Read the task request
                    with open(TASK_REQUEST_FILE, 'r') as f:
                        request_data = json.load(f)

                    task = request_data.get('task', '')
                    timestamp = request_data.get('timestamp', '')

                    print(f"\n[{time.strftime('%H:%M:%S')}] New task received: {task}")

                    # Load house data
                    house_data = load_house_data()
                    if not house_data:
                        response = "No house data available. Please run pixel_room_mapper.py first."
                    else:
                        house_json = json.dumps(house_data, indent=2)
                        # Generate mission using LLM
                        print("🤖 Generating mission...")
                        response = ask_ollama(house_json, task)
                        print(f" Mission generated: {response[:100]}...")

                    # Save response
                    with open(MISSION_RESPONSE_FILE, 'w') as f:
                        f.write(response)

                    # Also save to current_mission.txt for agent processor
                    with open(MISSION_FILE, 'w') as f:
                        f.write(response)

                    last_modified = current_modified

            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nShutting down LLM processor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


def main():
    clean_ollama_cache()

    # Clean up old files
    for f in [MISSION_RESPONSE_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned old {f}")

    # Initial load to check house data
    house_data = load_house_data()
    if house_data:
        print(f"Loaded house data: {house_data['summary']['total_rooms']} rooms")
    else:
        print("WARNING: No house data found. Will wait for data...")

    process_loop()


if __name__ == "__main__":
    main()