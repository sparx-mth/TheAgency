#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_sdf_world(filename):
    """Parse SDF file and extract objects with their positions and sizes."""
    tree = ET.parse(filename)
    root = tree.getroot()

    objects = []

    # Find all models in the world
    for model in root.findall('.//model'):
        model_name = model.get('name')

        # Skip ground planes - they're not obstacles
        if 'ground_plane' in model_name.lower():
            continue

        # Get model pose
        model_pose = [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw
        pose_elem = model.find('pose')
        if pose_elem is not None and pose_elem.text:
            pose_values = [float(x) for x in pose_elem.text.split()]
            model_pose[:len(pose_values)] = pose_values

        # Process each link in the model
        for link in model.findall('.//link'):
            # Get link pose if exists
            link_pose = [0, 0, 0, 0, 0, 0]
            link_pose_elem = link.find('pose')
            if link_pose_elem is not None and link_pose_elem.text:
                link_pose_values = [float(x) for x in link_pose_elem.text.split()]
                link_pose[:len(link_pose_values)] = link_pose_values

            # Find all collision geometries in the link
            for collision in link.findall('.//collision'):
                # Get collision pose relative to link
                collision_pose = [0, 0, 0, 0, 0, 0]
                pose_elem = collision.find('pose')
                if pose_elem is not None and pose_elem.text:
                    pose_values = [float(x) for x in pose_elem.text.split()]
                    collision_pose[:len(pose_values)] = pose_values

                # Get geometry
                geometry = collision.find('geometry')
                if geometry is not None:
                    # Handle box geometry
                    box = geometry.find('box')
                    if box is not None:
                        size = box.find('size')
                        if size is not None:
                            dimensions = [float(x) for x in size.text.split()]

                            # Combine all poses (model + link + collision)
                            # Simplified rotation handling
                            cos_yaw = math.cos(model_pose[5])
                            sin_yaw = math.sin(model_pose[5])

                            # Rotate link pose by model yaw
                            rotated_link_x = link_pose[0] * cos_yaw - link_pose[1] * sin_yaw
                            rotated_link_y = link_pose[0] * sin_yaw + link_pose[1] * cos_yaw

                            # Rotate collision pose by combined yaw
                            total_yaw = model_pose[5] + link_pose[5]
                            cos_total = math.cos(total_yaw)
                            sin_total = math.sin(total_yaw)
                            rotated_col_x = collision_pose[0] * cos_total - collision_pose[1] * sin_total
                            rotated_col_y = collision_pose[0] * sin_total + collision_pose[1] * cos_total

                            world_x = model_pose[0] + rotated_link_x + rotated_col_x
                            world_y = model_pose[1] + rotated_link_y + rotated_col_y

                            objects.append({
                                'name': model_name,
                                'x': world_x,
                                'y': world_y,
                                'width': dimensions[0],
                                'height': dimensions[1],
                                'yaw': model_pose[5] + link_pose[5] + collision_pose[5]
                            })

                    # Handle cylinder geometry
                    cylinder = geometry.find('cylinder')
                    if cylinder is not None:
                        radius_elem = cylinder.find('radius')
                        if radius_elem is not None:
                            radius = float(radius_elem.text)
                            # Use diameter as both width and height for cylinders
                            objects.append({
                                'name': model_name,
                                'x': model_pose[0] + link_pose[0] + collision_pose[0],
                                'y': model_pose[1] + link_pose[1] + collision_pose[1],
                                'width': radius * 2,
                                'height': radius * 2,
                                'yaw': model_pose[5]
                            })

                    # Handle mesh and other geometries
                    elif geometry.find('mesh') is not None:
                        # Use a default bounding box for meshes
                        objects.append({
                            'name': model_name,
                            'x': model_pose[0] + link_pose[0],
                            'y': model_pose[1] + link_pose[1],
                            'width': 1.0,  # Default size for meshes
                            'height': 1.0,
                            'yaw': model_pose[5]
                        })

    return objects


def create_occupancy_grid(objects, grid_width, grid_height, world_size=22):
    """
    Create occupancy grid from objects.

    Args:
        objects: List of objects with positions and sizes
        grid_width: Number of cells in x direction
        grid_height: Number of cells in y direction
        world_size: Size of the world in meters (assumes square world)
    """
    # Initialize grid with zeros (free space)
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Cell size in world coordinates
    cell_width = world_size / grid_width
    cell_height = world_size / grid_height

    # World origin offset (center of world is at 0,0)
    origin_offset = world_size / 2

    for obj in objects:
        # Handle rotation for accurate bounding box
        yaw = obj['yaw']
        cos_yaw = abs(math.cos(yaw))
        sin_yaw = abs(math.sin(yaw))

        # Calculate rotated bounding box dimensions
        rotated_width = obj['width'] * cos_yaw + obj['height'] * sin_yaw
        rotated_height = obj['width'] * sin_yaw + obj['height'] * cos_yaw

        half_width = rotated_width / 2
        half_height = rotated_height / 2

        # Get object corners in world coordinates
        min_x = obj['x'] - half_width
        max_x = obj['x'] + half_width
        min_y = obj['y'] - half_height
        max_y = obj['y'] + half_height

        # Convert to grid coordinates
        min_i = int((min_x + origin_offset) / cell_width)
        max_i = int((max_x + origin_offset) / cell_width)
        min_j = int((min_y + origin_offset) / cell_height)
        max_j = int((max_y + origin_offset) / cell_height)

        # Clamp to grid bounds
        min_i = max(0, min_i)
        max_i = min(grid_width - 1, max_i)
        min_j = max(0, min_j)
        max_j = min(grid_height - 1, max_j)

        # Mark cells as occupied
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                grid[j, i] = 1  # Note: grid[row, col] where row is y, col is x

    return grid


def display_map(grid):
    """Display the occupancy grid as an image with clear grid lines."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display grid (0=white for free, 1=black for occupied)
    im = ax.imshow(grid, cmap='gray_r', origin='lower', interpolation='nearest')

    # Get the dimensions
    rows, cols = grid.shape

    # Set ticks at every cell boundary for grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

    # Set major ticks for labels - only at multiples of 8 for a 32x32 grid
    major_interval = 8
    ax.set_xticks(np.arange(0, cols + 1, major_interval))
    ax.set_yticks(np.arange(0, rows + 1, major_interval))

    # Set the actual labels to show only at major ticks
    ax.set_xticklabels(np.arange(0, cols + 1, major_interval))
    ax.set_yticklabels(np.arange(0, rows + 1, major_interval))

    # Add grid lines at every cell boundary
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.6)
    ax.grid(which='major', color='black', linestyle='-', linewidth=1.0, alpha=0.8)

    # Remove tick marks
    ax.tick_params(which='both', length=0)

    # Add labels
    ax.set_xlabel('X (cells)', fontsize=12)
    ax.set_ylabel('Y (cells)', fontsize=12)
    ax.set_title(f'2D Occupancy Grid Map ({cols}×{rows} = {cols * rows} cells)', fontsize=14)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='0=Free (white), 1=Occupied (black)', shrink=0.8)

    # Make the plot square
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the conversion."""
    # File path
    sdf_file = '/home/nadavc/PycharmProjects/TheAgency_workspace/gz-sim-environment/ros2_ws/src/vehicle_packages/gz_sim_worlds/worlds/clearpath_playpen.sdf'  # Change this to your file path

    # Grid dimensions
    grid_width = 64  # Number of cells in x direction
    grid_height = 64  # Number of cells in y direction

    # World size (adjust based on your world)
    world_size = 22  # Your world is about 22x22 meters

    print(f"Parsing SDF file: {sdf_file}")
    objects = parse_sdf_world(sdf_file)
    print(f"Found {len(objects)} objects")

    # Optional: Print object details for debugging
    print("\nObject details:")
    for obj in objects[:10]:  # Print first 10 objects
        print(f"  {obj['name']}: pos=({obj['x']:.2f}, {obj['y']:.2f}), size=({obj['width']:.2f}x{obj['height']:.2f})")

    print(f"\nCreating {grid_width}x{grid_height} occupancy grid...")
    grid = create_occupancy_grid(objects, grid_width, grid_height, world_size)

    print("\nOccupancy Grid Statistics:")
    print(f"  Free cells: {np.sum(grid == 0)}")
    print(f"  Occupied cells: {np.sum(grid == 1)}")

    print("\nDisplaying map...")
    display_map(grid)

    # Save grid to file
    np.savetxt('occupancy_grid.txt', grid, fmt='%d')
    print("\nGrid saved to 'occupancy_grid.txt'")


if __name__ == "__main__":
    main()