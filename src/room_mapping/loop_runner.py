#!/usr/bin/env python3
"""
run_all.py - Run all house mapping components
"""

import subprocess
import time
import sys
import os
import glob


def main():
    print("=" * 60)
    print("HOUSE MAPPING SYSTEM - STARTING ALL COMPONENTS")
    print("=" * 60)

    # Ask about cleaning directory
    bbox_dir = "/home/user/PycharmProjects/TheAgency/src/room_mapping/ingest_out"
    json_files = []
    if os.path.exists(bbox_dir):
        import glob
        json_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))

    if json_files:
        print(f"\nFound {len(json_files)} existing detection files in:")
        print(f"  {bbox_dir}")
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
            for f in ["unified_rooms.json", "house_map.txt"]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed: {f}")

            print("Directory cleaned!\n")
        else:
            print("Keeping existing files.\n")

    processes = []

    try:
        # 1. Start receiver_owl first
        print("\n[1/4] Starting Receiver OWL (processes incoming images)...")
        receiver = subprocess.Popen(
            [sys.executable, "receiver_owl.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Receiver OWL", receiver))

        # Wait for first JSON files to appear
        print("\nWaiting for first detection files...")
        bbox_dir = "/home/user/PycharmProjects/TheAgency/src/room_mapping/ingest_out"

        while True:
            if os.path.exists(bbox_dir):
                json_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))
                if json_files:
                    print(f"✓ Found {len(json_files)} detection files!")
                    print("Starting remaining components...")
                    break

            print(".", end="", flush=True)
            time.sleep(1)

            # Check if receiver is still running
            if receiver.poll() is not None:
                print("\nError: Receiver OWL stopped unexpectedly")
                sys.exit(1)

        # 2. Now start the rest of the components
        print("\n[2/4] Starting Room Unifier (monitors for new scans)...")
        unifier = subprocess.Popen(
            [sys.executable, "room_unifier.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Room Unifier", unifier))
        time.sleep(2)  # Let it initialize

        # 3. Start renderer (auto-refreshes the visualization)
        print("[3/4] Starting House Renderer (auto-refresh enabled)...")
        renderer = subprocess.Popen(
            [sys.executable, "render_house.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(("Renderer", renderer))
        time.sleep(1)

        # 4. Start LLM mission generator (interactive)
        print("[4/4] Starting Mission Generator (interactive mode)...")
        print("\n" + "=" * 60)
        print("ALL SYSTEMS RUNNING!")
        print("=" * 60)
        print("\nSwitching to Mission Generator interactive mode...")
        print("(Other components running in background)\n")

        # Run mission generator in foreground for interaction
        mission = subprocess.Popen(
            [sys.executable, "llm_mission_generator.py"],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        processes.append(("Mission Generator", mission))

        # Wait for mission generator to finish (user quits)
        mission.wait()

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
        "room_unifier.py",
        "render_house.py",
        "llm_mission_generator.py",
        "tile_definitions.py"
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    main()