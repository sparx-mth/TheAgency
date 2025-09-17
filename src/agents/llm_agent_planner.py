#!/usr/bin/env python3

import subprocess
import tempfile
import os

PROMPT = """You are a drone controller with exactly 4 specialized agents:

Agent 1: Room Entry/Exit - Opens doors, enters rooms, exits rooms
Agent 2: Navigation - Moves to specific coordinates or named locations  
Agent 3: Room Scanner - Scans and analyzes the current room (cannot move or leave)
Agent 4: Wall Follower - Follows walls, detects wall boundaries

IMPORTANT RULES:
- To scan a room, you MUST first use Agent 1 to enter it, then Agent 3 to scan, then Agent 1 to exit
- Agent 3 can ONLY scan the room it's currently in, it cannot move
- Always enter a room before scanning it
- Always exit a room after scanning it
- Navigation between rooms requires Agent 2

EXAMPLES:
User: "Scan the kitchen"
Response:
1. Activate agent 2 to navigate to kitchen door
2. Activate agent 1 to enter the kitchen
3. Activate agent 3 to scan the kitchen
4. Activate agent 1 to exit the kitchen

User: "Go to room A and check the walls"
Response:
1. Activate agent 2 to navigate to room A
2. Activate agent 1 to enter room A
3. Activate agent 4 to follow and detect walls
4. Activate agent 1 to exit room A

Now convert this command into numbered steps (only output the steps, nothing else):
User command: """


def ask_ollama(user_input):
    """Send prompt to Ollama and get response"""
    full_prompt = PROMPT + user_input + "\nResponse:"

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(full_prompt)
        temp_file = f.name

    try:
        # Send file content to Ollama
        cmd = f"cat {temp_file} | ollama run llama3.1:8b"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        response = result.stdout.strip()

        # Clean up temp file
        os.unlink(temp_file)

        return response
    except Exception as e:
        os.unlink(temp_file)
        return f"Error: {e}"


def main():
    print("Simple LLM Agent Controller")
    print("-" * 40)

    while True:
        # Get user input
        user_input = input("\nEnter command (or 'quit'): ")

        if user_input.lower() == 'quit':
            break

        # Get response from Ollama
        print("\nProcessing...")
        response = ask_ollama(user_input)

        # Display ONLY the response (the steps)
        print("\n" + "=" * 40)
        print("EXECUTION PLAN:")
        print("=" * 40)
        print(response)
        print("=" * 40)


if __name__ == "__main__":
    example = """
    1. "Scan the kitchen."
    
    2. "Go to the bedroom and scan it."
    
    3. "Map the living room and the dining room."
    
    4. "Scan the bathroom, then the hallway."
    
    5. "Navigate to the office, scan it, then exit."
    
    6. "Go to the garage and scan it."
    
    7. "Scan the kitchen, bathroom, and bedroom."
    
    8. "Explore and scan room A and room B."
    
    9. "Map the pantry, then go to the dining room and scan it."
    
    10. "Go to the laundry room, scan it, then scan the hallway."
    
    11. "Scan the living room first, then the kitchen, then the bedroom."
    
    12. "Navigate through all rooms: kitchen, living room, and bathroom, scanning each one."
    
    13. "Go to room C, scan it, then go to room D and scan it."
    
    14. "Scan every room in the house starting with the hallway and ending with the garage."
    
    15. "Enter the dining room, scan it, exit, then go scan the bedroom."
    
    16. "Map the living room, pantry, and office in that order."
    
    17. "Go to the study and scan it, then scan the laundry room."
    
    18. "Navigate to room X, scan it, then scan room Y, then return to hallway."
    
    19. "Scan the kitchen, then immediately map the bathroom and bedroom."
    
    20 "Explore the garage, scan it, then map the dining room and the pantry."
    """

    main()