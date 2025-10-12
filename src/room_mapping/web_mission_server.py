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
- NEVER claim to see objects that are not explicitly in the "type" field of the JSON
- If you say "I see [object]" - that object MUST exist in the JSON with that exact type
- DO NOT hallucinate, imagine, or invent ANY objects not in the JSON data
- Before saying any object exists, verify it's actually in the JSON "type" fields

CONTEXT:
- The drone starts at the bottom center of the house (main entrance area)
- The drone can navigate using compass directions (north/south/east/west) and relative positions
- You have knowledge of typical house layouts (bathrooms have toiletries, kitchens have appliances, studies have computers, etc.)
- North is towards the top of the house, South towards the bottom, East to the right, West to the left

HOUSE MAP (JSON format showing rooms, objects, and their grid positions):
{house_json}

USER TASK: {user_task}

YOUR JOB:
Generate navigation instructions based STRICTLY on what exists in the map. NEVER invent rooms or objects that aren't listed. Check the "type" field of each object carefully.

SYNONYM HANDLING:
Use your understanding of language to recognize when objects are the same thing with different names. For example, if someone asks for a "gun" and the map has a "weapon", these refer to the same object. Similarly, "couch" and "sofa" are the same furniture piece. Always check if the requested item might be listed under a synonym or related term before deciding it doesn't exist.

RESPONSE FORMAT:

1. If object or its SYNONYM EXISTS in map:
   "I see the [object/synonym] is in the [room]. To get there: head [direction] from your current position, then [continue direction/turn] to reach the [room]. The [object] is located [near/next to/by] [landmark/other object]."

2. If object DOESN'T EXIST but a related synonym does:
   "I don't see [requested object] specifically, but there is a [synonym] in the [room] which is the same thing. To get there: head [direction] from your current position, then [continue direction/turn] to reach the [room]. The [synonym] is located [position]."

3. If object DOESN'T EXIST in any form:
   "I don't see [object] or anything similar in the map, but it's typically found in the [typical room]. Since there's no [typical room] in this house, the task is: explore the house systematically by heading [direction] and checking each room."
   OR if the typical room exists:
   "I don't see [object] or anything similar in the map, but it's typically in the [room]. To search for it: move [direction] to the [room] and check near [typical locations]."

4. For room navigation:
   If room EXISTS: "I see the [room]. To navigate there: go [direction] from your starting point, then [continue/turn] to enter the [room]."
   If room DOESN'T EXIST: "I don't see a [room] in this house. The task is: explore the available rooms by heading [direction] to check if any room serves as a [room]."

CRITICAL RULES:
- ONLY mention rooms that are explicitly listed in the map
- ONLY mention objects whose "type" field is explicitly listed in the map
- Use your knowledge of synonyms - if someone asks for something, check if it exists under a different but equivalent name
- If something isn't in the map, clearly state "I don't see [X] in the map"
- When suggesting where something typically is, check if that room actually exists first
- Always check for synonymous terms before saying something doesn't exist

GUIDELINES:
- Always describe the path using compass directions (north, south, east, west, northeast, etc.)
- Reference nearby objects or landmarks when describing the target's location
- Keep instructions clear and sequential (first do this, then do that)
- Use real-world knowledge: refrigerator-kitchen, toilet paper-bathroom, car keys-table, etc.
- Mention relative positions like "next to", "near", "by", "in the corner", "along the wall"

IMPORTANT: Output ONLY the navigation instruction as a single paragraph. Do NOT include any introduction, explanation, clarification, or numbered lists. Just write the complete instruction directly.

Generate the navigation instruction with directions:"""

def load_house_data():
    """Load and simplify house data"""
    try:
        with open("unified_rooms.json", 'r') as f:
            house_data = json.load(f)

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
        total_objects = sum(len(room.get('objects', [])) for room in current_data['rooms'].values())
        return jsonify({
            'rooms': len(current_data['rooms']),
            'objects': total_objects
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
            return jsonify({'response': 'No house data available. Please run room_unifier.py first.'}), 400

        house_json = json.dumps(current_data, indent=2)

        print(f"Calling Ollama with task: {task}")

        # Generate response using Ollama
        response = ask_ollama(house_json, task)

        print("\n=== DEBUG: JSON being sent ===")
        print(house_json)
        print("\n=== DEBUG: Task ===")
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
        print("Run room_unifier.py first to generate the house structure.")
    else:
        print("Loaded house data successfully")
        total_objects = sum(len(room.get('objects', [])) for room in latest_house_data['rooms'].values())
        print(f"Found {len(latest_house_data['rooms'])} rooms with {total_objects} objects")

    # Start background updater thread
    updater = threading.Thread(target=background_updater, daemon=True)
    updater.start()
    print("Auto-reload enabled (updates every second)")

    print("-" * 40)
    print("Starting web server on http://localhost:8080")
    print("Open your browser and navigate to the URL above")
    print("-" * 40)

    # Run Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    main()