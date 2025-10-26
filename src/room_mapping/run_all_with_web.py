#!/usr/bin/env python3
"""
run_all_with_web.py
-------------------
Full clean startup and pipeline runner for the house mapping system.

Launch order:
1. receiver_owl.py
2. pixel_room_mapper.py
3. render_house.py
4. web_mission_server.py
5. llm_object_finder.py
6. path_planner.py
7. object_location_describer.py
8. route_narrator.py
9. mission_to_agent_commands.py
"""

import subprocess
import time
import sys
import os
import glob
import webbrowser
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
ROOM_MAPPING_DIR = BASE_PATH / "room_mapping"
DATA_DIR = ROOM_MAPPING_DIR / "data"
INGEST_DIR = ROOM_MAPPING_DIR / "ingest_out"
DATA_DIR.mkdir(exist_ok=True)
INGEST_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------
# Cleanup section
# ----------------------------------------------------------
def clean_directory(folder, keep_files=None):
    """Remove all files from a folder except the ones listed in keep_files."""
    if not folder.exists():
        return
    keep_files = keep_files or []
    deleted = 0
    for file in folder.glob("*"):
        if file.name in keep_files:
            continue
        try:
            file.unlink()
            deleted += 1
        except Exception:
            pass
    print(f" Cleaned {deleted} files from {folder.name}")


# ----------------------------------------------------------
# Process management
# ----------------------------------------------------------
def start_process(script_name, label, delay=1.0):
    """Start a Python subprocess and return (label, process)."""
    print(f"[+] Starting {label} ({script_name}) ...")
    p = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    time.sleep(delay)
    return (label, p)


def wait_for_json_files(folder, timeout=120):
    """Wait until any JSON files appear (for receiver_owl output)."""
    print("\nWaiting for first detection JSONs...")
    start = time.time()
    while True:
        json_files = glob.glob(str(folder / "*.json"))
        if json_files:
            print(f" Found {len(json_files)} JSON files in {folder}")
            return True
        if time.time() - start > timeout:
            print("Timeout: no JSON files found within 2 minutes.")
            return False
        print(".", end="", flush=True)
        time.sleep(1)


# ----------------------------------------------------------
# Main launcher
# ----------------------------------------------------------
def main():
    print("=" * 60)
    print("HOUSE MAPPING SYSTEM – FULL PIPELINE LAUNCHER")
    print("=" * 60)

    # --- 1. Cleanup phase ---
    print("\n[0/9] Cleaning temporary data and JSONs...")
    clean_directory(DATA_DIR)
    clean_directory(INGEST_DIR, keep_files=["office.json", "office_map.txt"])
    print(" Cleanup complete.\n")

    processes = []

    try:
        # 1. Receiver OWL
        processes.append(start_process("receiver_owl.py", "Receiver OWL", 2.0))

        # Wait for JSONs to appear
        wait_for_json_files(INGEST_DIR, timeout=120)

        # 2. Pixel Room Mapper
        processes.append(start_process("pixel_room_mapper.py", "Pixel Room Mapper", 2.0))

        # 3. Renderer
        processes.append(start_process("render_house.py", "Renderer", 1.0))

        # 4. Web Mission Server
        processes.append(start_process("web_mission_server.py", "Web Mission Server", 2.0))

        # 5. LLM Object Finder
        processes.append(start_process("llm_object_finder.py", "LLM Object Finder", 1.0))

        # 6. Path Planner
        processes.append(start_process("path_planner.py", "Path Planner", 1.0))

        # 7. Object Location Describer
        processes.append(start_process("object_location_describer.py", "Object Location Describer", 1.0))

        # 8. Route Narrator
        processes.append(start_process("route_narrator.py", "Route Narrator", 1.0))

        # 9. Mission → Agent Commands
        processes.append(start_process("route_to_agents.py", "Agent Command Monitor", 1.0))

        # Open Web GUI
        print("\n Opening Web GUI at http://localhost:8080 ...")
        webbrowser.open("http://localhost:8080")

        print("\n" + "=" * 60)
        print("ALL COMPONENTS RUNNING")
        print("=" * 60)
        print(" Web GUI: http://localhost:8080")
        print("Press Ctrl+C to stop all components.\n")

        # Monitor loop
        while True:
            time.sleep(2)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n {name} stopped unexpectedly (exit code {proc.returncode})")
                    print(f"Restarting {name}...")
                    new_proc = subprocess.Popen(
                        [sys.executable, f"{name.lower().replace(' ', '_')}.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    processes = [(n, p if n != name else new_proc) for n, p in processes]

    except KeyboardInterrupt:
        print("\n Shutting down all components...")

    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("\nAll components stopped.")
        print("=" * 60)


if __name__ == "__main__":
    required = [
        "receiver_owl.py",
        "pixel_room_mapper.py",
        "render_house.py",
        "web_mission_server.py",
        "llm_object_finder.py",
        "path_planner.py",
        "object_location_describer.py",
        "route_narrator.py",
        "mission_to_agent_commands.py"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    main()
