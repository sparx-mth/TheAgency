#!/usr/bin/env python3
"""
run_all_with_web.py - Run all house mapping components with Web GUI
Now includes the mission to agent command monitor.
All CLI --args are forwarded to every subprocess via config.py.
"""

import subprocess
import time
import sys
import os
import glob
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house_config import get_config

cfg = get_config()

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = str(SCRIPT_DIR.parent)

# CLI args to forward to subprocesses (everything after script name)
_EXTRA_ARGS = sys.argv[1:]


def _spawn(script_name):
    """Launch a subprocess with the same CLI overrides."""
    return subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / script_name)] + _EXTRA_ARGS,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    print("=" * 60)
    print("HOUSE MAPPING SYSTEM WITH DUAL LLM PIPELINE")
    print("=" * 60)

    # Ask about cleaning directory
    bbox_dir = os.path.join(BASE_PATH, "room_mapping", cfg.ingest_out_dir)
    json_files = []
    if os.path.exists(bbox_dir):
        json_files = glob.glob(os.path.join(bbox_dir, "*.json"))

    if json_files:
        print(f"\nFound {len(json_files)} existing JSON files in:")
        print(f"  {bbox_dir}")
        print("\nFiles found:")
        for f in json_files[:5]:
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
            for f in [cfg.mission_file, cfg.agent_commands_file,
                       cfg.task_request_file, cfg.mission_response_file]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed: {f}")

            # Clean data folder
            if os.path.exists(cfg.data_dir):
                for f in glob.glob(os.path.join(cfg.data_dir, "*")):
                    try:
                        os.remove(f)
                        print(f"  Removed: {f}")
                    except:
                        pass

            print("Directory cleaned!\n")
        else:
            print("Keeping existing files.\n")

    processes = []

    try:
        # 1. Start receiver_owl first
        print("\n[1/7] Starting Receiver OWL (processes incoming images)...")
        receiver = _spawn("receiver_owl.py")
        processes.append(("Receiver OWL", receiver))

        # Check for existing detection files
        if os.path.exists(bbox_dir):
            json_files = glob.glob(os.path.join(bbox_dir, "*.json"))
            if json_files:
                print(f"\nFound {len(json_files)} existing JSON files!")
            else:
                print("\nNo JSON files yet - components will process them as they arrive.")
        else:
            print(f"\nDirectory {bbox_dir} not found yet - will be created by receiver.")

        # 2. Start the rest of the components
        print("\n[2/7] Starting Room Unifier (monitors for new scans)...")
        unifier = _spawn("pixel_room_mapper.py")
        processes.append(("Room Unifier", unifier))
        time.sleep(2)

        print("[3/7] Starting LLM Mission Processor...")
        llm_processor = _spawn("llm_mission_processor.py")
        processes.append(("LLM Processor", llm_processor))
        time.sleep(2)

        # 3. Start renderer
        print("[4/7] Starting House Renderer (auto-refresh enabled)...")
        renderer = _spawn("render_house.py")
        processes.append(("Renderer", renderer))
        time.sleep(1)

        # 4. Start Web Mission Server
        print("[5/7] Starting Web Mission Server...")
        web_server = _spawn("web_mission_server_llm.py")
        processes.append(("Web Server", web_server))
        time.sleep(2)

        # 5. Start Mission to Agent Monitor (Second LLM)
        print("[6/7] Starting Agent Command Monitor (Second LLM)...")
        agent_monitor = _spawn("mission_to_agent_commands.py")
        processes.append(("Agent Monitor", agent_monitor))
        time.sleep(1)

        # 6. Open browser
        print("[7/7] Opening web browser...")
        webbrowser.open(f"http://localhost:{cfg.web_port}")

        print("\n" + "=" * 60)
        print("ALL SYSTEMS RUNNING!")
        print("=" * 60)
        print(f"\nWeb GUI is available at: http://localhost:{cfg.web_port}")
        print("You can now use the web interface to send navigation tasks")
        print("Dual LLM Pipeline Active:")
        print(f"   1. Mission Generator LLM ({cfg.mission_model})")
        print(f"   2. Agent Command LLM ({cfg.agent_model})")
        print("\nPress Ctrl+C to stop all components")
        print("=" * 60)

        # Keep running until interrupted
        while True:
            time.sleep(1)

            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n Warning: {name} has stopped (exit code: {proc.returncode})")
                    if name == "Web Server":
                        print("Restarting Web Server...")
                        new_proc = _spawn("web_mission_server_llm.py")
                        for i, (n, p) in enumerate(processes):
                            if n == "Web Server":
                                processes[i] = ("Web Server", new_proc)
                                break
                    elif name == "Agent Monitor":
                        print("Restarting Agent Monitor...")
                        new_proc = _spawn("mission_to_agent_commands.py")
                        for i, (n, p) in enumerate(processes):
                            if n == "Agent Monitor":
                                processes[i] = ("Agent Monitor", new_proc)
                                break

    except KeyboardInterrupt:
        print("\n\nShutting down all components...")

    finally:
        for name, proc in processes:
            if proc.poll() is None:
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
        "index.html",
        "house_config.py"
    ]

    missing = [f for f in required_files if not (SCRIPT_DIR / f).exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nMake sure you have created all necessary files")
        sys.exit(1)

    main()