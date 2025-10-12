#!/usr/bin/env python3
"""
pipeline.py - Complete pipeline to process scans and generate missions
"""

import json
import subprocess
import numpy as np
from tile_definitions import TileType

# ============= CONFIGURATION =============
HOUSE_WIDTH_M = 10.0
HOUSE_HEIGHT_M = 10.0
GRID_RESOLUTION = 0.2
ENTRY_X_M = HOUSE_WIDTH_M / 2
ENTRY_Y_M = HOUSE_HEIGHT_M - 1.0


# ============= STEP 1: ROOM UNIFICATION =============
def process_scans():
    """Process scans by running room_unifier.py script"""
    print("=" * 60)
    print("STEP 1: Processing scans and unifying rooms...")
    print("=" * 60)

    # Run room_unifier.py
    print("\nRunning room_unifier.py...")
    try:
        subprocess.run(["python3", "room_unifier.py"], check=True)
    except Exception as e:
        print(f"Error running room_unifier.py: {e}")
        return False

    # Add entry point
    print("\nAdding drone entry point...")
    entry_grid_x = int(ENTRY_X_M / GRID_RESOLUTION)
    entry_grid_y = int(ENTRY_Y_M / GRID_RESOLUTION)

    # Update JSON
    with open("unified_rooms.json", 'r') as f:
        data = json.load(f)
    data["drone_entry_point"] = {
        "position_m": (ENTRY_X_M, ENTRY_Y_M),
        "position_grid": (entry_grid_x, entry_grid_y)
    }
    with open("unified_rooms.json", 'w') as f:
        json.dump(data, f, indent=2)

    # Mark entry point in grid
    grid = np.loadtxt("house_map.txt", dtype=np.int8)
    if 0 <= entry_grid_y < grid.shape[0] and 0 <= entry_grid_x < grid.shape[1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y, x = entry_grid_y + dy, entry_grid_x + dx
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    if grid[y, x] == TileType.FREE_SPACE:
                        grid[y, x] = TileType.ENTRY_POINT
    np.savetxt("house_map.txt", grid, fmt='%d')

    print(f"✓ Entry point marked at ({ENTRY_X_M:.1f}, {ENTRY_Y_M:.1f})m")
    return True


# ============= STEP 2: RENDER HOUSE =============
def render_house():
    """Always render the house map without asking"""
    print("\n" + "=" * 60)
    print("STEP 2: House Visualization")
    print("=" * 60)
    subprocess.Popen(["python3", "render_house.py", "--cell-size", "20"])
    print("Renderer launched in separate window")


# ============= STEP 3: MISSION GENERATION =============
def run_mission_generator():
    """Run the mission generator"""
    print("\n" + "=" * 60)
    print("STEP 3: Mission Generator")
    print("=" * 60)
    print("Starting interactive mission generator...")
    print("Note: Drone starts at the ENTRY POINT (bottom center)")

    subprocess.run(["python3", "llm_mission_generator.py"])


# ============= MAIN PIPELINE =============
def run_pipeline():
    """Run complete pipeline"""
    print("\n" + "=" * 60)
    print(" DRONE NAVIGATION PIPELINE")
    print("=" * 60)

    # Step 1: Process scans
    if not process_scans():
        print("Pipeline aborted.")
        return

    # Step 2: Always visualize
    render_house()

    # Step 3: Run mission generator
    run_mission_generator()

    print("\nPipeline complete!")


if __name__ == "__main__":
    run_pipeline()
