#!/usr/bin/env python3
"""
web_mission_server.py - Flask server for Mission Generator Web GUI
Connects the web interface to the existing LLM mission generator
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import subprocess
import tempfile
import os
import threading
import time

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Global variable to store latest house data
latest_house_data = None
data_lock = threading.Lock()

# Import the PROMPT from the original file
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
- Offices and closed rooms have doors at specific grid coordinates shown in the "doors" field
- You must understand the house layout using bbox coordinates [x1,y1,x2,y2] to determine room positions
- Use bbox coordinates to determine if a room is at the beginning, middle, or end of a hallway
- Avoid confusing compass directions - use relative terms like "straight ahead", "to your left", "to your right"

HOUSE MAP (JSON format showing all rooms and their objects):
{house_json}

USER TASK: {user_task}

YOUR JOB:
Generate human-friendly navigation instructions that describe:
1. How to navigate from room to room to reach the target room
2. Where exactly to find the object within that room (near what other objects)

SYNONYM HANDLING AND ROOM MATCHING:
- Use your understanding of language to recognize variations in room and object names
- Room name variations: "Inbal's room" = "Inbal's Office", "Yaniv's room" = "Yaniv Oren's Office", etc.
- Object synonyms: "couch" = "sofa", "weapon" = "gun", "monitor" = "screen", etc.
- Always check if the requested item or room might be listed under a similar or related name
- Be flexible with room names - if someone asks for "Inbal's room" and you see "Inbal's Office", that's a match

NAVIGATION INSTRUCTION FORMAT:

1. If object or its SYNONYM EXISTS in map:
   "I see that the [object/synonym] is in [room name]. From your position in the Open Space, walk into the hallway. [Room name] is [at the beginning/middle/end] of the hallway on your [left/right]. Once you enter [room name], you will find the [object] [specific location: next to/near/beside which other objects in that room]."

   Example: "I see that the weapon is in the MAMAD. From your position in the Open Space, walk into the hallway. The MAMAD is at the end of the hallway on your right. Once you enter the MAMAD, you will find the weapon next to the refrigerator, placed on the chair beside it."

2. If object DOESN'T EXIST but a related synonym does:
   "I don't see [requested object] specifically, but there is a [synonym] in the [room name] which is the same thing. From your position in the Open Space, walk into the hallway. [Room name] is [position description]. Once you enter the [room name], you will find the [synonym] [specific location in room]."

3. If object DOESN'T EXIST in any form:
   "I don't see [object] or anything similar in the map. Based on typical house layouts, this would usually be found in a [typical room type]. The available rooms are: [list actual rooms]. You should explore [suggest most likely room based on room names] by [describe how to get there]."

4. For room navigation:
   If room EXISTS: "I see [room name/variation]. From your position in the Open Space, walk into the hallway. [Room name] is [at the beginning/middle/end] of the hallway on your [left/right]. [Additional landmark info if helpful]."

   Example for object in Open Space: "I see that there is a chair right here in the Open Space where you are currently located. The chair is next to the table and monitor."

   Example for finding ALL chairs: "I found 5 chairs in the house: In the Open Space (your current location): 1 chair next to the table and monitor. In Inbal's Office: 1 chair. To get there, walk into the hallway, Inbal's Office is at the beginning of the hallway on your left. In Yaniv Oren's Office: 1 chair. Continue down the hallway, it's the next room on your left. In Moshe's Office: 1 chair. It's toward the end of the hallway on your left. In the storage room: 1 chair next to the weapon."

   Example for other rooms: "I see that the weapon is in the MAMAD. From your position in the Open Space, walk into the hallway. The MAMAD is at the end of the hallway on your right. Once you enter the MAMAD, you will find the weapon next to the refrigerator, placed on the chair beside it."

   If room DOESN'T EXIST: "I don't see a [requested room] in this house. The available rooms are: [list actual room names]. You may want to check [suggest most similar room] which can be reached from the Open Space by [describe path using simple directions]."

CRITICAL NAVIGATION RULES:
- The drone STARTS IN Open Space - never tell it to navigate to Open Space from Open Space
- For objects IN Open Space: say "right here where you are" or "in your current location"
- For "find all [object]" requests: list EVERY instance of that object in ALL rooms, not just one
- Describe navigation using simple, human-friendly terms: "straight ahead", "on your left/right", "at the beginning/middle/end"
- NEVER say "exit through the door" for Open Space or hallway - they have no doors
- Use bbox coordinates to accurately describe room positions:
  * Lower y-values = beginning of hallway (closer to entrance)
  * Higher y-values = end of hallway (further from entrance)
- Be flexible with room names: "Inbal's room" matches "Inbal's Office", "Moshe's room" matches "Moshe's Office"
- Always specify which room contains the object
- Describe object location relative to other objects in that same room
- ONLY mention rooms and objects that actually exist in the JSON
- Avoid confusing directional terms like "north/south/east/west" - use "left/right/straight ahead" instead

PATH DESCRIPTION GUIDELINES:
- Always start navigation from the Open Space (grid position [27,34]) unless otherwise specified
- Remember: Open Space and hallways have NO DOORS (empty doors [] array) - say "walk into the hallway" not "exit through the door"
- Use simple, relative directions that humans understand:
  * "walk straight into the hallway"
  * "on your left/right"
  * "at the beginning/middle/end of the hallway"
  * "the first/second/third room on your left"
- Use bbox coordinates to determine room positions:
  * Compare y-values: lower y = beginning of hallway, higher y = toward the end
  * Compare x-values: lower x = left side, higher x = right side
  * Example: If a room bbox is [0,23,18,34], it's toward the end of the hallway on the left
- Describe the path naturally: "from the Open Space, walk into the hallway, then [position of target room]"
- Reference other rooms as landmarks: "you'll pass Inbal's Office first, then Yaniv's Office"

OBJECT LOCATION GUIDELINES:
- Always specify which other objects are near the target
- Use spatial relationships: "next to", "beside", "near", "between", "in the corner by"
- If multiple instances of an object exist, distinguish them: "the chair near the table" vs "the chair by the window"
- Be specific about placement: "on the table", "under the desk", "against the wall"

IMPORTANT: Output ONLY the navigation instruction as a single flowing paragraph. Do NOT include any introduction, explanation, clarification, or numbered lists. Just write the complete instruction directly in natural, human-friendly language.

Generate the navigation instruction:"""

def load_house_data():
    """Load and format house data clearly for LLM"""
    try:
        with open("unified_rooms.json", 'r') as f:
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
                "bbox": room_info.get('bbox'),  # Keep bbox for position reference
                "objects": object_types,  # List of object names in this room
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


def background_updater():
    """Background thread that continuously updates house data"""
    global latest_house_data

    while True:
        new_data = load_house_data()
        if new_data:
            with data_lock:
                latest_house_data = new_data
        time.sleep(1)  # Update every second


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


@app.route('/')
def serve_index():
    """Serve the HTML GUI from index.html file"""
    return send_from_directory('.', 'index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current house status"""
    with data_lock:
        current_data = latest_house_data

    if current_data:
        return jsonify({
            'rooms': current_data['summary']['total_rooms'],
            'objects': current_data['summary']['total_objects']
        })

    return jsonify({'rooms': 0, 'objects': 0})


@app.route('/api/generate', methods=['POST'])
def generate_mission():
    """Generate mission instructions"""
    try:
        data = request.get_json()
        task = data.get('task', '')

        print(f"Received task: {task}")

        if not task:
            return jsonify({'response': 'Please provide a task'}), 400

        # Get latest house data
        with data_lock:
            current_data = latest_house_data

        if not current_data:
            return jsonify({'response': 'No house data available. Please run pixel_room_mapper.py first.'}), 400

        house_json = json.dumps(current_data, indent=2)

        print(f"Calling Ollama with task: {task}")

        # Generate response using Ollama
        response = ask_ollama(house_json, task)

        print("\n=== DEBUG: JSON being sent to LLM ===")
        print(house_json)
        print("\n=== DEBUG: Task ===")
        print(task)
        print("=" * 30)

        print(f"Ollama response: {response[:100]}...")

        return jsonify({'response': response})

    except Exception as e:
        print(f"Error in generate_mission: {e}")
        return jsonify({'response': f'Error: {str(e)}'}), 500


def main():
    global latest_house_data

    print("Mission Generator Web Server")
    print("-" * 40)

    # Check if index.html exists
    if not os.path.exists('index.html'):
        print("ERROR: index.html not found in the current directory.")
        print("Make sure index.html is in the same folder as this script.")
        return

    # Initial load
    latest_house_data = load_house_data()
    if not latest_house_data:
        print("WARNING: unified_rooms.json not found.")
        print("Run pixel_room_mapper.py first to generate the house structure.")
    else:
        print("Loaded house data successfully")
        print(
            f"Found {latest_house_data['summary']['total_rooms']} rooms with {latest_house_data['summary']['total_objects']} objects")
        print("\nRooms detected:")
        for room_name, room_data in latest_house_data['rooms'].items():
            print(f"  - {room_name}: {room_data['object_count']} objects")
            if room_data['objects']:
                print(f"    Objects: {', '.join(room_data['objects'])}")

    # Start background updater thread
    updater = threading.Thread(target=background_updater, daemon=True)
    updater.start()
    print("\nAuto-reload enabled (updates every second)")

    print("-" * 40)
    print("Starting web server on http://localhost:8080")
    print("Open your browser and navigate to the URL above")
    print("-" * 40)

    # Run Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    main()