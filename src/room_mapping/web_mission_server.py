#!/usr/bin/env python3
"""
web_mission_server.py - Flask server for Mission Generator Web GUI
Connects the web interface to the existing LLM mission generator
Updated to work with new pipeline structure
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
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


def check_agent_commands():
    """Check if agent commands are available"""
    agent_file = "data/agent_commands.txt"
    if os.path.exists(agent_file):
        try:
            with open(agent_file, 'r') as f:
                commands = f.read().strip()
            # Clear the file after reading
            os.remove(agent_file)
            return commands
        except:
            return None
    return None


def wait_for_pipeline_results(task, timeout=30):
    """Wait for and aggregate results from the new pipeline"""
    start_time = time.time()

    result_files = {
        'object_location': 'data/object_location.json',
        'planned_path': 'data/planned_path.json',
        'inroom_description': 'data/inroom_description.json',
        'route_narration': 'data/route_narration.json'
    }

    results = {}
    mission_response = []

    while time.time() - start_time < timeout:
        for key, filepath in result_files.items():
            if key not in results and os.path.exists(filepath):
                try:
                    if os.path.getmtime(filepath) > start_time:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        results[key] = data
                        print(f"Found {key} result")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        # --- NEW: if object_location exists, check immediately ---
        if 'object_location' in results:
            obj_data = results['object_location']
            found_room = obj_data.get('room')
            found_object = obj_data.get('object')

            # Normalize 'none' strings
            if isinstance(found_room, str) and found_room.lower() == 'none':
                found_room = None
            if isinstance(found_object, str) and found_object.lower() == 'none':
                found_object = None

            # If both are missing -> stop immediately
            if not found_room and not found_object:
                return "Unable to find the requested object. Please try searching for something else.", ""

            # If object found but room missing -> invalid
            if found_object and not found_room:
                return "Invalid mission data: object found without room. Please try again.", ""

        # Only proceed with full mission if both object_location and route_narration ready
        if 'object_location' in results and 'route_narration' in results:
            obj_data = results['object_location']
            found_room = obj_data.get('room')
            found_object = obj_data.get('object')

            if isinstance(found_room, str) and found_room.lower() == 'none':
                found_room = None
            if isinstance(found_object, str) and found_object.lower() == 'none':
                found_object = None

            # CASE 1: both found
            if found_room and found_object:
                mission_response.append(f"Target identified: {found_object} in {found_room}")

            # CASE 2: room found only
            elif found_room and not found_object:
                mission_response.append(f"Object not found, but room '{found_room}' identified. Navigating to the room for further inspection.")
                results.pop('inroom_description', None)

            # CASE 3: object without room (invalid)
            elif found_object and not found_room:
                return "Invalid mission data: object found without room. Please try again.", ""

            # CASE 4: nothing found
            else:
                return "Unable to find the requested object. Please try searching for something else.", ""

            # Add route narration
            if 'route_narration' in results:
                narration = results['route_narration'].get('narration', '')
                if narration:
                    mission_response.append("\nNavigation instructions:")
                    mission_response.append(narration)

            # Add in-room description
            if 'inroom_description' in results:
                desc = results['inroom_description'].get('description', '')
                if desc:
                    mission_response.append("\nIn-room guidance:")
                    mission_response.append(desc)

            agent_commands = check_agent_commands()
            return '\n'.join(mission_response), agent_commands

        time.sleep(0.5)

    if mission_response:
        return '\n'.join(mission_response), check_agent_commands()
    else:
        return 'Timeout: Mission processing incomplete. Make sure all pipeline components are running.', ''


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
    """Generate mission instructions using the new pipeline"""
    try:
        data = request.get_json()
        task = data.get('task', '')

        print(f"Received task: {task}")

        if not task:
            return jsonify({'response': 'Please provide a task', 'agent_commands': ''}), 400

        request_data = {
            'task': task,
            'requested': task,
            'timestamp': time.time()
        }

        with open("data/object_request.json", 'w') as f:
            json.dump(request_data, f)
        print("Task request written to data/object_request.json")

        # Wait for pipeline results
        response, agent_commands = wait_for_pipeline_results(task)

        # --- NEW: if object not found or invalid, clear commands ---
        if response.startswith("Unable to find") or response.startswith("Invalid mission"):
            return jsonify({'response': response, 'agent_commands': ''})

        return jsonify({'response': response, 'agent_commands': agent_commands or ''})

    except Exception as e:
        print(f"Error in generate_mission: {e}")
        return jsonify({'response': f'Error: {str(e)}', 'agent_commands': ''}), 500



@app.route('/SPARX.jpg')
def serve_logo():
    """Serve the SPARX logo image"""
    import os
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparx_logo.png')

    if os.path.exists(logo_path):
        return send_file(logo_path,
                         mimetype='image/jpeg',
                         as_attachment=False,
                         max_age=0)
    else:
        return 'Logo not found', 404


@app.route('/api/map', methods=['GET'])
def get_map():
    """Serve the current map image"""
    import os

    # Use absolute path
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'current_map.png')

    if os.path.exists(map_path):
        # Fix: Use proper path and add as_attachment=False
        return send_file(map_path,
                         mimetype='image/png',
                         as_attachment=False,
                         max_age=0)
    else:
        return 'Map not found', 404


@app.route('/api/map_status', methods=['GET'])
def get_map_status():
    """Check if map exists and when it was last updated"""
    map_path = 'data/current_map.png'  # Changed from 'data/current_map.png'

    if os.path.exists(map_path):
        return jsonify({
            'available': True,
            'last_updated': os.path.getmtime(map_path)
        })
    else:
        return jsonify({
            'available': False,
            'last_updated': None
        })


@app.route('/api/check_agent_commands', methods=['GET'])
def api_check_agent_commands():
    """API endpoint to check for agent commands"""
    commands = check_agent_commands()
    return jsonify({'agent_commands': commands or ''})


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