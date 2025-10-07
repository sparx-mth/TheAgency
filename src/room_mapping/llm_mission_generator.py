#!/usr/bin/env python3
"""
mission_generator.py - Simple LLM-based mission generator for drone navigation
"""

import json
import subprocess
import tempfile
import os

PROMPT = """You are a mission planner for an autonomous drone that navigates houses. The drone needs clear navigation instructions based on a house map and user requests.

CONTEXT:
- The drone starts in the bedroom unless told otherwise
- The drone can navigate to coordinates, enter/exit rooms, and scan areas
- You have knowledge of typical house layouts (bathrooms have toiletries, kitchens have appliances, studies have computers, etc.)

HOUSE MAP (JSON format showing rooms, objects, and their grid positions):
{house_json}

USER TASK: {user_task}

YOUR JOB:
Generate ONE SHORT SENTENCE or BRIEF PARAGRAPH (max 2-3 sentences) with navigation instructions.

RESPONSE FORMAT:

1. If object EXISTS in map:
   "I see the [object] is in the [room] at (x,y), so the task is: navigate to [room] at (x,y) and retrieve it."

2. If object DOESN'T EXIST but you know where it typically is:
   "I don't see [object] in the map, but it's typically in the [room], so the task is: navigate to [room] to search for it."

3. For room navigation:
   "I see the [room] at (x,y), so the task is: navigate to (x,y) and enter."
   OR
   "I haven't found the [room] yet, so the task is: explore unexplored doors to find it."

GUIDELINES:
- Keep it SHORT - one clear instruction
- Use real-world knowledge: refrigerator-kitchen, toilet paper-bathroom, car keys-On the table, and more
- Include positions (x,y) when available
- Be direct and specific

Generate the brief navigation instruction:"""


def ask_ollama(house_json, user_task):
    """Send prompt to Ollama and get response"""
    full_prompt = PROMPT.format(house_json=house_json, user_task=user_task)

    # Write prompt to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(full_prompt)
        temp_file = f.name

    try:
        cmd = f"cat {temp_file} | ollama run llama3.1:8b"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        response = result.stdout.strip()
        os.unlink(temp_file)
        return response
    except Exception as e:
        os.unlink(temp_file)
        return f"Error: {e}"


def main():
    print("Mission Generator for Drone Navigation")
    print("-" * 40)

    # Load house structure
    try:
        with open("unified_rooms.json", 'r') as f:
            house_data = json.load(f)
    except FileNotFoundError:
        print("ERROR: unified_rooms.json not found.")
        print("Run room_unifier.py first to generate the house structure.")
        return

    # Simplify house data for prompt
    simplified = {"rooms": {}}
    for room_name, room_info in house_data.get('rooms', {}).items():
        simplified["rooms"][room_name] = {
            "position": room_info.get('camera_position'),
            "objects": [
                {"type": obj['type'], "location": obj['location']}
                for obj in room_info.get('objects', [])
            ],
            "doors": room_info.get('doors', [])
        }

    house_json = json.dumps(simplified, indent=2)

    # Interactive loop
    while True:
        user_task = input("\nEnter task (or 'quit'): ").strip()

        if user_task.lower() == 'quit':
            break

        if not user_task:
            continue

        print("\nGenerating mission...")
        print("=" * 60)
        response = ask_ollama(house_json, user_task)
        print(response)
        print("=" * 60)


if __name__ == "__main__":

    example = """
    1. "Find the refrigerator"

    2. "Get to the weapon"

    3. "living room"

    4. "Find the shampoo"

    5. "Locate all chairs"

    6. "Go to the garage"

    7. "find the computer"

    8. "kitchen"

    9. "toilet paper"

    10. "car keys"
    """

    main()