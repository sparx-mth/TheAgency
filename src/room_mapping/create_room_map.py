"""
create_room_map.py

Creates room map from camera scans using perspective geometry for accurate distance estimation.
"""

import numpy as np
import math
from typing import List, Dict
from tile_definitions import (
    TileType, OBJECT_TO_TILE, OBJECT_SIZES, OBJECT_HEIGHTS, TILE_NAMES
)


def estimate_distance_from_bbox(bbox: List[int], object_class: str,
                               camera_height: float, camera_fov_v: float,
                               frame_height: int) -> float:
    """
    Estimate object distance using perspective projection and known object heights.

    Based on the pinhole camera model:
    distance = (real_height * focal_length) / pixel_height

    Args:
        bbox: [x1, y1, x2, y2] in pixels
        object_class: Type of object (for height lookup)
        camera_height: Height of camera from ground in meters
        camera_fov_v: Vertical field of view in degrees
        frame_height: Frame height in pixels

    Returns:
        Estimated distance in meters
    """
    x1, y1, x2, y2 = bbox
    bbox_height_px = y2 - y1

    # Get real-world height of object
    real_object_height = OBJECT_HEIGHTS.get(object_class.lower(), 1.0)

    # Calculate focal length in pixels
    focal_length_px = (frame_height / 2) / math.tan(math.radians(camera_fov_v / 2))

    # Basic distance from perspective projection
    if bbox_height_px > 0:
        distance = (real_object_height * focal_length_px) / bbox_height_px
    else:
        distance = 5.0  # Default fallback

    # Adjust for camera height and object vertical position
    bbox_center_y = (y1 + y2) / 2
    vertical_position = bbox_center_y / frame_height  # 0 = top, 1 = bottom

    # Apply ground plane constraint
    if vertical_position > 0.5:  # Object in lower half of frame
        angle_to_base = math.atan2(camera_height, distance)
        ground_distance = camera_height / math.tan(angle_to_base)
        weight = (vertical_position - 0.5) * 2  # 0 to 1 for lower half
        distance = distance * (1 - weight * 0.3)  # Slight adjustment

    return distance


def create_map_from_scans(scans: List[Dict],
                          room_width_m: float,
                          room_height_m: float,
                          camera_height_m: float = 1.5,
                          camera_fov_h: float = 60,
                          camera_fov_v: float = 45,
                          frame_width: int = 640,
                          frame_height: int = 480,
                          grid_resolution: float = 0.25) -> np.ndarray:
    """
    Convert camera scans into a 2D grid map using proper distance estimation.

    Args:
        scans: List of scans with 'yaw' (in radians) and 'bboxes'
        room_width_m: Room width in meters
        room_height_m: Room depth in meters
        camera_height_m: Camera height from ground (typical: 1.5m standing, 1.0m sitting)
        camera_fov_h: Horizontal field of view in degrees
        camera_fov_v: Vertical field of view in degrees
        frame_width: Camera frame width in pixels
        frame_height: Camera frame height in pixels
        grid_resolution: Meters per tile (0.25 = 25cm)

    Returns:
        2D numpy array representing the room
    """
    # Calculate grid dimensions
    grid_width = int(room_width_m / grid_resolution)
    grid_height = int(room_height_m / grid_resolution)

    # Initialize map with walls
    room_map = np.zeros((grid_height, grid_width), dtype=np.int8)
    room_map[0, :] = TileType.WALL
    room_map[-1, :] = TileType.WALL
    room_map[:, 0] = TileType.WALL
    room_map[:, -1] = TileType.WALL

    # Camera at center
    cam_x = grid_width // 2
    cam_y = grid_height // 2
    fov_h_rad = math.radians(camera_fov_h)

    print(f"Camera config: height={camera_height_m}m, FOV={camera_fov_h}°x{camera_fov_v}°")
    print(f"Frame size: {frame_width}x{frame_height}px")

    all_detections = []

    # Process each scan
    for scan in scans:
        # Changed: accept 'yaw' in radians directly (no conversion needed)
        angle_rad = scan.get('yaw', scan.get('angle', 0))  # Support both 'yaw' and 'angle' fields

        for bbox_data in scan['bboxes']:
            obj_class = bbox_data['class'].lower()
            tile_type = OBJECT_TO_TILE.get(obj_class)
            if not tile_type:
                continue

            bbox = bbox_data['bbox']
            x1, y1, x2, y2 = bbox

            # Estimate distance using perspective geometry
            distance_m = estimate_distance_from_bbox(
                bbox, obj_class, camera_height_m, camera_fov_v, frame_height
            )

            # Limit to room dimensions
            max_distance = math.sqrt((room_width_m/2)**2 + (room_height_m/2)**2)
            distance_m = min(distance_m, max_distance)

            # Calculate horizontal angle to object
            bbox_center_x = (x1 + x2) / 2
            angle_offset = ((bbox_center_x / frame_width) - 0.5) * fov_h_rad
            object_angle = angle_rad + angle_offset

            # Calculate grid position
            grid_dist = distance_m / grid_resolution
            obj_x = int(cam_x + grid_dist * math.sin(object_angle))
            obj_y = int(cam_y - grid_dist * math.cos(object_angle))

            # Get object size for placement
            obj_dims = OBJECT_SIZES.get(tile_type, (0.5, 0.5, 0.5))
            obj_width_tiles = max(1, int(obj_dims[0] / grid_resolution))
            obj_depth_tiles = max(1, int(obj_dims[1] / grid_resolution))

            # Store detection info (convert to degrees for display)
            all_detections.append({
                'class': obj_class,
                'distance': distance_m,
                'angle': math.degrees(object_angle),  # Convert to degrees for display
                'grid_pos': (obj_x, obj_y),
                'confidence': bbox_data.get('confidence', 1.0)
            })

            # Place object on map
            for dy in range(obj_depth_tiles):
                for dx in range(obj_width_tiles):
                    y = obj_y - obj_depth_tiles//2 + dy
                    x = obj_x - obj_width_tiles//2 + dx
                    if 1 <= x < grid_width-1 and 1 <= y < grid_height-1:
                        if room_map[y, x] == TileType.FREE_SPACE:
                            room_map[y, x] = tile_type

    # Add entry point at camera
    room_map[cam_y, cam_x] = TileType.ENTRY_POINT

    # Print detection summary
    print(f"\nDetected {len(all_detections)} objects:")
    for det in all_detections[:5]:  # Show first 5
        print(f"  {det['class']}: {det['distance']:.1f}m away at {det['angle']:.0f}°")

    return room_map


def save_map(room_map: np.ndarray, filename: str = "room_map.txt"):
    """Save the map to a text file."""
    np.savetxt(filename, room_map, fmt='%d')
    print(f"\nMap saved to {filename}")
    print(f"Map size: {room_map.shape[1]}x{room_map.shape[0]} tiles")

    # Count objects by type using TILE_NAMES from definitions
    object_counts = {}
    for tile_type, tile_name in TILE_NAMES.items():
        if tile_type >= TileType.CHAIR:  # Object tile types start from CHAIR
            count = np.sum(room_map == tile_type)
            if count > 0:
                object_counts[tile_name] = count

    if object_counts:
        print("Objects placed:")
        for name, count in object_counts.items():
            print(f"  {name}: {count} tiles")


# Example usage with realistic camera parameters
if __name__ == "__main__":
    import json
    import os

    # Load scans from JSON file
    json_path = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/images/scans.json"

    try:
        with open(json_path, 'r') as f:
            scans = json.load(f)
        print(f"Loaded {len(scans)} scans from {json_path}")
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Using example scans.")
        # Fallback example scans if JSON file doesn't exist (now using radians)
        scans = [
            {
                'yaw': 0,  # North (0 radians)
                'bboxes': [
                    {'class': 'tv', 'bbox': [280, 200, 380, 280], 'confidence': 0.95},
                    {'class': 'table', 'bbox': [200, 300, 340, 420], 'confidence': 0.88},
                    {'class': 'chair', 'bbox': [400, 320, 480, 440], 'confidence': 0.82},
                ]
            },
            {
                'yaw': 1.5708,  # East (π/2 radians ≈ 90°)
                'bboxes': [
                    {'class': 'couch', 'bbox': [150, 250, 450, 400], 'confidence': 0.91},
                    {'class': 'plant', 'bbox': [500, 280, 560, 380], 'confidence': 0.75},
                ]
            },
        ]
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        print("Using example scans instead.")
        scans = []

    # Create map with camera parameters
    room_map = create_map_from_scans(
        scans,
        room_width_m=5.0,
        room_height_m=5.0,
        camera_height_m=0.1,  # Standing height
        camera_fov_h=60,  # Typical webcam horizontal FOV
        camera_fov_v=45,  # Typical webcam vertical FOV
        frame_width=1280,
        frame_height=720,
        grid_resolution=0.1    )

    save_map(room_map, "room_map.txt")