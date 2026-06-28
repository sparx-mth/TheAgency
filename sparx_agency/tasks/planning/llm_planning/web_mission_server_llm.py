#!/usr/bin/env python3
"""
web_mission_server.py - Flask server for Mission Generator Web GUI
Connects the web interface to the existing LLM mission generator
Saves mission output to file for agent command processor
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import subprocess
import sys
import tempfile
import os
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from house_config import get_config

cfg = get_config()

app = Flask(__name__, static_folder=str(SCRIPT_DIR))
CORS(app)  # Enable CORS for all routes

# Global variable to store latest house data
latest_house_data = None
data_lock = threading.Lock()


def load_house_data():
    """Load and format house data clearly for LLM"""
    try:
        with open(cfg.unified_rooms_json, 'r') as f:
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


def background_updater():
    """Background thread that continuously updates house data"""
    global latest_house_data

    while True:
        new_data = load_house_data()
        if new_data:
            with data_lock:
                latest_house_data = new_data
        time.sleep(cfg.data_update_interval)


def check_agent_commands(min_timestamp=None):
    """Return agent commands only if the file is newer than min_timestamp."""
    if os.path.exists(cfg.agent_commands_file):
        try:
            if (min_timestamp is None) or (os.path.getmtime(cfg.agent_commands_file) > float(min_timestamp)):
                with open(cfg.agent_commands_file, 'r') as f:
                    return f.read().strip()
        except:
            return None
    return None


@app.route('/')
def serve_index():
    """Serve the HTML GUI from index.html file"""
    return send_from_directory(str(SCRIPT_DIR), 'index.html')


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
            return jsonify({'response': 'Please provide a task', 'agent_commands': ''}), 400

        # Write task request for LLM processor
        request_data = {
            'task': task,
            'timestamp': time.time()
        }
        with open(cfg.task_request_file, 'w') as f:
            json.dump(request_data, f)
        print(f"Task request written to {cfg.task_request_file}")

        # Wait for both mission response and its matching agent commands
        timeout = cfg.timeout
        start_time = time.time()

        mission_text = None
        agent_commands = None

        while time.time() - start_time < timeout:
            # 1) Mission text produced by llm_mission_processor.py
            if os.path.exists(cfg.mission_response_file):
                if os.path.getmtime(cfg.mission_response_file) > request_data['timestamp']:
                    with open(cfg.mission_response_file, 'r') as f:
                        mission_text = f.read().strip()

            # 2) Agent commands produced by mission_to_agent_commands.py
            agent_commands = check_agent_commands(min_timestamp=request_data['timestamp'])

            if mission_text and agent_commands:
                return jsonify({
                    'response': mission_text,
                    'agent_commands': agent_commands
                })

            time.sleep(cfg.web_poll_interval)

        # If we timeout, return whatever we have
        return jsonify({
            'response': mission_text or 'Timeout waiting for LLM processor. Make sure llm_mission_processor.py is running.',
            'agent_commands': agent_commands or ''
        }), (200 if mission_text else 504)

    except Exception as e:
        print(f"Error in generate_mission: {e}")
        return jsonify({'response': f'Error: {str(e)}', 'agent_commands': ''}), 500


@app.route('/SPARX.jpg')
def serve_logo():
    """Serve the SPARX logo image"""
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
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            cfg.current_map_png)

    if os.path.exists(map_path):
        return send_file(map_path,
                         mimetype='image/png',
                         as_attachment=False,
                         max_age=0)
    else:
        return 'Map not found', 404


@app.route('/api/map_status', methods=['GET'])
def get_map_status():
    """Check if map exists and when it was last updated"""
    if os.path.exists(cfg.current_map_png):
        return jsonify({
            'available': True,
            'last_updated': os.path.getmtime(cfg.current_map_png)
        })
    else:
        return jsonify({
            'available': False,
            'last_updated': None
        })

@app.route('/api/check_agent_commands', methods=['GET'])
def api_check_agent_commands():
    """API endpoint to check for agent commands (optionally since=timestamp)"""
    since = request.args.get('since', None)
    commands = check_agent_commands(min_timestamp=since)
    return jsonify({'agent_commands': commands or ''})


def main():
    global latest_house_data

    print("Mission Generator Web Server")
    print("-" * 40)

    # Check if index.html exists
    if not (SCRIPT_DIR / 'index.html').exists():
        print(f"ERROR: index.html not found in {SCRIPT_DIR}")
        return

    # Initial load
    latest_house_data = load_house_data()
    if not latest_house_data:
        print(f"WARNING: {cfg.unified_rooms_json} not found.")
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
    print(f"\nAuto-reload enabled (updates every {cfg.data_update_interval}s)")

    print("-" * 40)
    print(f"Starting web server on http://localhost:{cfg.web_port}")
    print("Open your browser and navigate to the URL above")
    print("-" * 40)

    # Run Flask server
    app.run(host=cfg.web_host, port=cfg.web_port, debug=False)


if __name__ == "__main__":
    main()