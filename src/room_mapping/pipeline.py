#!/usr/bin/env python3
"""
pipeline.py - Complete pipeline to process scans and generate missions
"""

import json
import subprocess
import os
import sys
from room_unifier import RoomUnifier
from tile_definitions import OBJECT_TO_TILE, OBJECT_HEIGHTS, TileType, OBJECT_SIZES

# ============= CONFIGURATION =============
SCANS_FILE = "/home/user/PycharmProjects/TheAgency/src/room_mapping/images/scans.json"
HOUSE_WIDTH_M = 10.0
HOUSE_HEIGHT_M = 10.0
GRID_RESOLUTION = 0.2
CAMERA_X_M = 5.0  # Center of house
CAMERA_Y_M = 5.0
CAMERA_HEIGHT_M = 1.5
CAMERA_FOV_H = 70
CAMERA_FOV_V = 50
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Drone entry point (bottom center of house)
ENTRY_X_M = HOUSE_WIDTH_M / 2  # Center X
ENTRY_Y_M = HOUSE_HEIGHT_M - 1.0  # Near bottom (1m from edge)


# ============= STEP 1: ROOM UNIFICATION =============
def process_scans():
    """Process scans and create unified room structure"""
    print("=" * 60)
    print("STEP 1: Processing scans and unifying rooms...")
    print("=" * 60)

    # Load scans
    with open(SCANS_FILE, 'r') as f:
        scans = json.load(f)

    # Create unifier
    unifier = RoomUnifier(
        house_width_m=HOUSE_WIDTH_M,
        house_height_m=HOUSE_HEIGHT_M,
        grid_resolution=GRID_RESOLUTION
    )

    # Process each scan
    for i, scan in enumerate(scans):
        # Get position from pose if available
        if 'pose' in scan:
            pose_x = scan['pose']['x'] + CAMERA_X_M
            pose_y = scan['pose']['y'] + CAMERA_Y_M
            pose_z = scan['pose'].get('z', CAMERA_HEIGHT_M)
        else:
            pose_x = CAMERA_X_M
            pose_y = CAMERA_Y_M
            pose_z = CAMERA_HEIGHT_M

        print(f"\nProcessing scan {i + 1}/{len(scans)}")

        # Add scan
        unifier.add_scan(
            scan_data=scan,
            camera_x_m=pose_x,
            camera_y_m=pose_y,
            camera_height_m=pose_z,
            camera_fov_h=CAMERA_FOV_H,
            camera_fov_v=CAMERA_FOV_V,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            room_name="main_room"
        )

    # Add entry point to the structure
    entry_grid_x = int(ENTRY_X_M / GRID_RESOLUTION)
    entry_grid_y = int(ENTRY_Y_M / GRID_RESOLUTION)

    # Save results with entry point info
    unifier.save()

    # Load the saved JSON and add entry point
    with open("unified_rooms.json", 'r') as f:
        data = json.load(f)

    data["drone_entry_point"] = {
        "position_m": (ENTRY_X_M, ENTRY_Y_M),
        "position_grid": (entry_grid_x, entry_grid_y),
        "description": "Drone starting position at bottom center of house"
    }

    with open("unified_rooms.json", 'w') as f:
        json.dump(data, f, indent=2)

    # Mark entry point in the grid map
    import numpy as np
    grid = np.loadtxt("house_map.txt", dtype=np.int8)

    # Mark entry point with special tile type
    if 0 <= entry_grid_y < grid.shape[0] and 0 <= entry_grid_x < grid.shape[1]:
        grid[entry_grid_y, entry_grid_x] = TileType.ENTRY_POINT
        # Also mark surrounding cells for visibility (3x3 area)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y, x = entry_grid_y + dy, entry_grid_x + dx
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    if grid[y, x] == TileType.FREE_SPACE:  # Only mark if free
                        grid[y, x] = TileType.ENTRY_POINT

    np.savetxt("house_map.txt", grid, fmt='%d')

    print(f"\n✓ Processed {len(scans)} scans")
    print(f"✓ Found {len(unifier.all_objects)} objects")
    print(f"✓ Marked entry point at ({ENTRY_X_M:.1f}, {ENTRY_Y_M:.1f})m")
    print(f"✓ Saved to unified_rooms.json and house_map.txt")

    return unifier


# ============= STEP 2: RENDER HOUSE =============
def render_house():
    """Optional: Render the house map"""
    print("\n" + "=" * 60)
    print("STEP 2: House Visualization")
    print("=" * 60)

    response = input("Do you want to visualize the house map? (y/n): ").strip().lower()

    if response == 'y':
        try:
            print("\nLaunching house renderer...")
            print("Controls: ESC to exit, R to reload, +/- to zoom")
            print("CYAN color = Drone entry point (bottom center)")
            # Run renderer in subprocess so pipeline can continue after
            subprocess.Popen(["python3", "render_house.py", "--cell-size", "20"])
            print("Renderer launched in separate window")
        except FileNotFoundError:
            print("Error: render_house.py not found")
        except Exception as e:
            print(f"Error launching renderer: {e}")
    else:
        print("Skipping visualization")


# ============= STEP 3: MISSION GENERATION =============
def run_mission_generator():
    """Run the mission generator"""
    print("\n" + "=" * 60)
    print("STEP 3: Mission Generator")
    print("=" * 60)
    print("Starting interactive mission generator...\n")
    print("Note: Drone starts at the ENTRY POINT (bottom center)")

    try:
        subprocess.run(["python3", "llm_mission_generator.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running mission generator: {e}")
    except FileNotFoundError:
        print("Error: llm_mission_generator.py not found in current directory")


# ============= MAIN PIPELINE =============
def run_pipeline():
    """Run complete pipeline"""
    print("\n" + "=" * 60)
    print(" DRONE NAVIGATION PIPELINE")
    print("=" * 60)

    # Step 1: Process scans
    unifier = process_scans()

    # Step 2: Optional visualization
    render_house()

    # Step 3: Run mission generator
    run_mission_generator()

    print("\nPipeline complete!")


if __name__ == "__main__":
    run_pipeline()