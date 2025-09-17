#!/usr/bin/env python3

import subprocess
import tempfile
import os
import re


def get_llm_plan(command):
    """Get mission plan from LLM"""
    prompt = """You are a drone controller with 4 agents:
1: doorway (enter/exit rooms)
2: navigate (go to coordinates)
3: room (scan current room)
4: wall (follow walls)

Convert this command to numbered steps using agent numbers:
Command: """ + command + "\nSteps:"

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(prompt)
        tmp = f.name

    try:
        cmd = f"cat {tmp} | ollama run llama3.1:8b"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        os.unlink(tmp)
        return result.stdout.strip()
    except:
        os.unlink(tmp)
        return None


def parse_missions(llm_output):
    """Convert LLM output to mission format"""
    missions = []

    # Find lines with "agent X" or "activate agent X"
    for line in llm_output.split('\n'):
        # Look for patterns like "agent 1", "agent 2", etc
        match = re.search(r'agent\s+(\d)', line.lower())
        if match:
            agent_num = match.group(1)

            # Map to task types
            task_map = {
                '1': 'doorway',
                '2': 'navigate',
                '3': 'room',
                '4': 'wall'
            }

            if agent_num in task_map:
                missions.append(f"{agent_num} {task_map[agent_num]}")

    return missions


def write_mission_file(missions):
    """Write missions to temp file for visualizer"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for m in missions:
            f.write(m + '\n')
        return f.name


def main():
    print("LLM Mission Runner")
    print("-" * 40)

    # Get command from user
    command = input("What do you want the drone to do? ")

    print("\nGetting plan from LLM...")
    plan = get_llm_plan(command)

    if not plan:
        print("Failed to get plan")
        return

    print("\n" + "=" * 40)
    print("LLM PLAN:")
    print("=" * 40)
    print(plan)
    print("=" * 40)

    # Parse missions
    missions = parse_missions(plan)

    if not missions:
        print("No missions found in plan")
        return

    print("\nMISSIONS TO RUN:")
    for m in missions:
        print(f"  {m}")

    # Write to file
    mission_file = write_mission_file(missions)
    print(f"\nMissions saved to: {mission_file}")

    # Run visualizer with missions
    run = input("\nRun missions now? (y/n): ")
    if run.lower() == 'y':
        # You can pipe missions to visualizer or modify it to read from file
        print("\nTo run these missions:")
        print(f"1. Start visualize_agents.py")
        print(f"2. Choose option 2 (Mission Mode)")
        print(f"3. Enter these missions:")
        for m in missions:
            print(f"   {m}")


if __name__ == "__main__":
    main()