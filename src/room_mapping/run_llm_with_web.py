#!/usr/bin/env python3
"""
run_all_with_web.py - Run all house mapping components with Web GUI
Now includes the mission to agent command monitor
"""

import subprocess
import time
import sys
import os
import glob
import webbrowser
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent)


def main():
    print("=" * 60)
    print("HOUSE MAPPING SYSTEM WITH DUAL LLM PIPELINE")
    print("=" * 60)

    # Ask about cleaning directory
    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    json_files = []
    if os.path.exists(bbox_dir):
        # Look for ALL .json files, not just *_dets.json
        json_files = glob.glob(os.path.join(bbox_dir, "*.json"))

    if json_files:
        print(f"\nFound {len(json_files)} existing JSON files in:")
        print(f"  {bbox_dir}")
        print("\nFiles found:")
        for f in json_files[:5]:  # Show first 5 files
            print(f"  - {os.path.basename(f)}")
        if len(json_files) > 5:
            print(f"  ... and {len(json_files) - 5} more")

        response = input("\nClean directory before starting? (y/n): ").strip().lower()

        if response == 'y':
            print("Cleaning directory...")
            for f in json_files:
                try:
                    os.remove(f)
                    print(f"  Removed: {os.path.basename(f)}")
                except:
                    pass

            # Also clean output files
            for f in ["unified_rooms.json", "house_map.txt", "current_mission.txt", "agent_commands.txt"]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed: {f}")

            print("Directory cleaned!\n")
        else:
            print("Keeping existing files.\n")

    processes = []

    try:
        # 1. Start receiver_owl first
        print("\n[1/7] Starting Receiver OWL (processes incoming images)...")
        receiver = subprocess.Popen(
            [sys.executable, "receiver_owl.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Receiver OWL", receiver))

        # Wait for first JSON files to appear
        print("\nWaiting for first detection files...")
        bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")

        wait_count = 0
        while True:
            if os.path.exists(bbox_dir):
                # Look for ALL .json files
                json_files = glob.glob(os.path.join(bbox_dir, "*.json"))
                if json_files:
                    print(f"\n✓ Found {len(json_files)} JSON files!")
                    print("Starting remaining components...")
                    break

            print(".", end="", flush=True)
            time.sleep(1)
            wait_count += 1
            if wait_count % 60 == 0:
                print("\n", end="", flush=True)

            # Add timeout check
            if wait_count > 600:  # 600 seconds timeout
                print("\n No JSON files found after 30 seconds.")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response == 'y':
                    print("Continuing with setup...")
                    break
                else:
                    print("Exiting...")
                    sys.exit(1)

            # Check if receiver is still running
            if receiver.poll() is not None:
                print("\nError: Receiver OWL stopped unexpectedly")
                sys.exit(1)

        # 2. Now start the rest of the components
        print("\n[2/7] Starting Room Unifier (monitors for new scans)...")
        unifier = subprocess.Popen(
            [sys.executable, "pixel_room_mapper.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Room Unifier", unifier))
        time.sleep(2)  # Let it initialize

        print("[3/7] Starting LLM Mission Processor...")
        llm_processor = subprocess.Popen(
            [sys.executable, "llm_mission_processor.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("LLM Processor", llm_processor))
        time.sleep(2)  # Let it initialize

        # 3. Start renderer (auto-refreshes the visualization)
        print("[4/7] Starting House Renderer (auto-refresh enabled)...")
        renderer = subprocess.Popen(
            [sys.executable, "render_house.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Renderer", renderer))
        time.sleep(1)

        # 4. Start Web Mission Server
        print("[5/7] Starting Web Mission Server...")
        web_server = subprocess.Popen(
            [sys.executable, "web_mission_server_llm.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Web Server", web_server))
        time.sleep(2)  # Give server time to start

        # 5. Start Mission to Agent Monitor (Second LLM)
        print("[6/7] Starting Agent Command Monitor (Second LLM)...")
        agent_monitor = subprocess.Popen(
            [sys.executable, "mission_to_agent_commands.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Agent Monitor", agent_monitor))
        time.sleep(1)

        # 6. Open browser
        print("[7/7] Opening web browser...")
        webbrowser.open("http://localhost:8080")

        print("\n" + "=" * 60)
        print("ALL SYSTEMS RUNNING!")
        print("=" * 60)
        print("\n🌐 Web GUI is available at: http://localhost:8080")
        print("🚁 You can now use the web interface to send navigation tasks")
        print("🤖 Dual LLM Pipeline Active:")
        print("   1. Mission Generator LLM creates navigation instructions")
        print("   2. Agent Command LLM converts to step-by-step commands")
        print("\nPress Ctrl+C to stop all components")
        print("=" * 60)

        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if any critical process has died
            for name, proc in processes:
                if proc.poll() is not None:  # Process terminated
                    print(f"\n Warning: {name} has stopped")
                    if name == "Web Server":
                        print("Restarting Web Server...")
                        web_server = subprocess.Popen(
                            [sys.executable, "web_mission_server_llm.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        # Update the process in the list
                        for i, (n, p) in enumerate(processes):
                            if n == "Web Server":
                                processes[i] = ("Web Server", web_server)
                                break
                    elif name == "Agent Monitor":
                        print("Restarting Agent Monitor...")
                        agent_monitor = subprocess.Popen(
                            [sys.executable, "mission_to_agent_commands.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        # Update the process in the list
                        for i, (n, p) in enumerate(processes):
                            if n == "Agent Monitor":
                                processes[i] = ("Agent Monitor", agent_monitor)
                                break

    except KeyboardInterrupt:
        print("\n\nShutting down all components...")

    finally:
        # Clean shutdown
        for name, proc in processes:
            if proc.poll() is None:  # Still running
                print(f"Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

        print("\nAll components stopped.")
        print("=" * 60)


if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        "receiver_owl.py",
        "pixel_room_mapper.py",
        "render_house.py",
        "web_mission_server_llm.py",
        "mission_to_agent_commands.py",
        "llm_mission_processor.py",
        "index.html"
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nMake sure you have created all necessary files")
        sys.exit(1)

    main()