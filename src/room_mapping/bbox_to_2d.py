#!/usr/bin/env python3
"""
room_unifier.py - Block 3: Room Scanner and Unifier

Processes BBOX data from multiple camera positions, calculates real distances,
and unifies them into a single global house structure.
Dynamic tile assignment with pixel-based size calculation.
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple, Optional
from tile_definitions import OBJECT_TO_TILE, TileType


class RoomUnifier:
    """Processes and unifies room scans using camera geometry."""

    def __init__(self,
                 house_width_m: float = 2.5,
                 house_height_m: float = 2.5,
                 grid_resolution: float = 0.1):
        """
        Initialize the room unifier with house parameters.

        Args:
            house_width_m: Total house width in meters
            house_height_m: Total house height/depth in meters
            grid_resolution: Meters per grid cell
        """
        self.house_width_m = house_width_m
        self.house_height_m = house_height_m
        self.grid_resolution = grid_resolution

        # Calculate grid dimensions
        self.grid_width = int(house_width_m / grid_resolution)
        self.grid_height = int(house_height_m / grid_resolution)

        # Storage for rooms and objects
        self.rooms = {}
        self.all_objects = []

        # Dynamic tile mapping
        self.dynamic_tile_counter = 100  # Start from 100 for dynamic tiles
        self.dynamic_tile_map = {}  # Map object_class -> dynamic tile type

    def pixels_to_meters(self,
                         bbox_width_px: int,
                         bbox_height_px: int,
                         frame_width: int,
                         frame_height: int,
                         camera_fov_h: float,
                         camera_fov_v: float,
                         distance_m: float = 1.0) -> Tuple[float, float]:
        """
        Convert pixel dimensions to real-world meters at a given distance.

        Args:
            bbox_width_px: Width of bounding box in pixels
            bbox_height_px: Height of bounding box in pixels
            frame_width: Total frame width in pixels
            frame_height: Total frame height in pixels
            camera_fov_h: Horizontal field of view in degrees
            camera_fov_v: Vertical field of view in degrees
            distance_m: Distance to object in meters (default 1.0)

        Returns:
            (width_m, height_m) in real-world meters
        """
        # Calculate angular size of each pixel
        h_angle_per_pixel = camera_fov_h / frame_width  # degrees per pixel
        v_angle_per_pixel = camera_fov_v / frame_height  # degrees per pixel

        # Calculate angular size of bounding box
        bbox_h_angle = bbox_width_px * h_angle_per_pixel  # degrees
        bbox_v_angle = bbox_height_px * v_angle_per_pixel  # degrees

        # Convert to radians
        bbox_h_angle_rad = math.radians(bbox_h_angle)
        bbox_v_angle_rad = math.radians(bbox_v_angle)

        # Calculate real-world size at the given distance
        # Using basic trigonometry: size = 2 * distance * tan(angle/2)
        width_m = 2 * distance_m * math.tan(bbox_h_angle_rad / 2)
        height_m = 2 * distance_m * math.tan(bbox_v_angle_rad / 2)

        return width_m, height_m

    def get_or_create_tile_type(self, object_class: str) -> int:
        """
        Get tile type for an object class, creating a new dynamic type if needed.

        Args:
            object_class: Type/class of the object

        Returns:
            Tile type integer
        """
        # First check if it's in the predefined tiles
        tile_type = OBJECT_TO_TILE.get(object_class)
        if tile_type is not None:
            return tile_type

        # Check if we already created a dynamic tile for this class
        if object_class in self.dynamic_tile_map:
            return self.dynamic_tile_map[object_class]

        # Create a new dynamic tile type
        new_tile_type = self.dynamic_tile_counter
        self.dynamic_tile_map[object_class] = new_tile_type
        self.dynamic_tile_counter += 1

        return new_tile_type

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert meters to grid coordinates."""
        grid_x = int(x_m / self.grid_resolution)
        grid_y = int(y_m / self.grid_resolution)
        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        return grid_x, grid_y

    def add_scan(self,
                 scan_data: Dict,
                 camera_x_m: float,
                 camera_y_m: float,
                 camera_height_m: float = 0.5,
                 camera_fov_h: float = 70,
                 camera_fov_v: float = 50,
                 frame_width: int = 1280,
                 frame_height: int = 720,
                 room_name: Optional[str] = None,
                 room_dims_m: Optional[Tuple[float, float]] = (2.5, 2.5),
                 yaw: float = 0.0,
                 fixed_distance_m: float = 1.0):
        """
        Add a scan from a specific camera position.

        Args:
            scan_data: Detection data with 'detections'
            camera_x_m: Camera X position in meters within the house
            camera_y_m: Camera Y position in meters within the house
            camera_height_m: Camera height from ground in meters
            camera_fov_h: Horizontal field of view in degrees
            camera_fov_v: Vertical field of view in degrees
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
            room_name: Optional name for the room being scanned
            room_dims_m: Optional (width, height) of room in meters
            yaw: Camera angle in radians
            fixed_distance_m: Fixed distance to all objects (default 1.0m)
        """
        # Convert camera position to grid
        cam_grid_x, cam_grid_y = self.meters_to_grid(camera_x_m, camera_y_m)

        # Get detections from new format
        detections = scan_data.get('detections', [])

        # Calculate FOV in radians for this scan
        fov_h_rad = math.radians(camera_fov_h)

        # Process detections
        detected_objects = []
        detected_doors = []

        for detection in detections:
            # Extract object class from label (remove 'a ' or 'an ' prefix)
            label = detection.get('label', '').lower()
            obj_class = label.replace('a ', '').replace('an ', '').strip()

            bbox = detection['bbox']
            confidence = detection.get('score', 1.0)

            # Get or create tile type dynamically
            tile_type = self.get_or_create_tile_type(obj_class)

            # Calculate bounding box dimensions in pixels
            bbox_width_px = bbox[2] - bbox[0]
            bbox_height_px = bbox[3] - bbox[1]

            # Calculate real-world size using pixel dimensions
            obj_width_m, obj_height_m = self.pixels_to_meters(
                bbox_width_px, bbox_height_px,
                frame_width, frame_height,
                camera_fov_h, camera_fov_v,
                fixed_distance_m
            )

            # Use fixed distance for all objects
            distance_m = fixed_distance_m

            # Calculate horizontal angle offset
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            angle_offset = ((bbox_center_x / frame_width) - 0.5) * fov_h_rad
            object_angle = yaw + math.pi + angle_offset

            # Calculate object position in meters
            obj_x_m = camera_x_m - distance_m * math.sin(object_angle)
            obj_y_m = camera_y_m + distance_m * math.cos(object_angle)

            # Convert to grid coordinates
            obj_grid_x, obj_grid_y = self.meters_to_grid(obj_x_m, obj_y_m)

            # Create object info with calculated dimensions
            obj_info = {
                "type": obj_class,
                "tile_type": tile_type,
                "location": (obj_grid_x, obj_grid_y),
                "location_m": (round(obj_x_m, 2), round(obj_y_m, 2)),
                "distance_m": round(distance_m, 2),
                "size_m": {
                    "width": round(obj_width_m, 3),
                    "height": round(obj_height_m, 3),
                    "depth": round(obj_width_m, 3)  # Assume depth = width for simplicity
                },
                "bbox_pixels": {
                    "width": bbox_width_px,
                    "height": bbox_height_px
                },
                "confidence": round(confidence, 2),
                "from_camera": (cam_grid_x, cam_grid_y)
            }

            # Store as door or object
            if 'door' in obj_class:
                obj_info["status"] = "open" if "open" in obj_class else "closed"
                obj_info["connects_to"] = "unknown"
                detected_doors.append(obj_info)
            else:
                detected_objects.append(obj_info)

            # Add to global object list
            self.all_objects.append(obj_info)

        # If room name provided, store room info
        if room_name:
            room_info = {
                "name": room_name,
                "camera_position": (cam_grid_x, cam_grid_y),
                "camera_position_m": (camera_x_m, camera_y_m),
                "camera_params": {
                    "height_m": camera_height_m,
                    "fov_h": camera_fov_h,
                    "fov_v": camera_fov_v,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "fixed_distance_m": fixed_distance_m
                },
                "dimensions_m": room_dims_m,
                "objects": detected_objects,
                "doors": detected_doors,
                "scan_yaw": round(yaw, 3),
                "dynamic_tiles": self.dynamic_tile_map.copy()  # Store dynamic tile mapping
            }

            # If this room already exists, merge the objects
            if room_name in self.rooms:
                self.rooms[room_name]["objects"].extend(detected_objects)
                self.rooms[room_name]["doors"].extend(detected_doors)
                # Update dynamic tiles
                self.rooms[room_name]["dynamic_tiles"].update(self.dynamic_tile_map)
            else:
                self.rooms[room_name] = room_info

    def get_unified_structure(self) -> Dict:
        """Return the unified room structure."""
        return self.rooms

    def create_grid_map(self, show_camera_positions: bool = True) -> np.ndarray:
        """
        Create a 2D grid map of the entire house.

        Args:
            show_camera_positions: Whether to mark camera positions

        Returns:
            2D numpy array with tile types
        """
        # Initialize grid
        grid = np.full((self.grid_height, self.grid_width),
                       TileType.FREE_SPACE, dtype=np.int16)  # int16 for dynamic tiles

        # Place all detected objects with their calculated sizes
        for obj in self.all_objects:
            cx, cy = obj["location"]  # Center position
            tile_type = obj.get("tile_type", TileType.UNKNOWN)

            # Doors get special treatment
            if "status" in obj:  # It's a door
                if obj["status"] == "open":
                    tile_type = TileType.DOOR_OPEN
                else:
                    tile_type = TileType.DOOR_CLOSED
                # Doors are thin, just place as single tile
                if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
                    grid[cy, cx] = tile_type
            else:
                # Get object dimensions from calculated size
                if "size_m" in obj:
                    obj_width_m = obj["size_m"]["width"]
                    obj_depth_m = obj["size_m"]["depth"]
                else:
                    # Fallback to default small size
                    obj_width_m = 0.3
                    obj_depth_m = 0.3

                # Convert to grid cells
                obj_width_cells = max(1, int(obj_width_m / self.grid_resolution))
                obj_depth_cells = max(1, int(obj_depth_m / self.grid_resolution))

                # Place object cells around center
                for dy in range(obj_depth_cells):
                    for dx in range(obj_width_cells):
                        x = cx - obj_width_cells // 2 + dx
                        y = cy - obj_depth_cells // 2 + dy
                        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                            # Only place if cell is free
                            if grid[y, x] == TileType.FREE_SPACE:
                                grid[y, x] = tile_type

        # Optionally mark camera positions
        if show_camera_positions:
            for room_name, room_info in self.rooms.items():
                cam_x, cam_y = room_info["camera_position"]
                if 0 <= cam_x < self.grid_width and 0 <= cam_y < self.grid_height:
                    grid[cam_y, cam_x] = TileType.ENTRY_POINT

        # Draw room boundaries if dimensions are known
        for room_name, room_info in self.rooms.items():
            if room_info["dimensions_m"]:
                width_m, height_m = room_info["dimensions_m"]
                cam_x_m, cam_y_m = room_info["camera_position_m"]

                # Calculate room corners (assuming camera is in center)
                left = int((cam_x_m - width_m / 2) / self.grid_resolution)
                right = int((cam_x_m + width_m / 2) / self.grid_resolution)
                top = int((cam_y_m - height_m / 2) / self.grid_resolution)
                bottom = int((cam_y_m + height_m / 2) / self.grid_resolution)

                # Draw walls if within bounds
                for x in range(max(0, left), min(self.grid_width, right + 1)):
                    if 0 <= top < self.grid_height:
                        grid[top, x] = TileType.WALL
                    if 0 <= bottom < self.grid_height:
                        grid[bottom, x] = TileType.WALL

                for y in range(max(0, top), min(self.grid_height, bottom + 1)):
                    if 0 <= left < self.grid_width:
                        grid[y, left] = TileType.WALL
                    if 0 <= right < self.grid_width:
                        grid[y, right] = TileType.WALL

        return grid

    def save(self, json_file: str = "unified_rooms.json",
             map_file: str = "house_map.txt"):
        """Save the unified structure and grid map."""
        # Prepare JSON-serializable structure
        output = {
            "house_dimensions_m": {
                "width": self.house_width_m,
                "height": self.house_height_m
            },
            "grid_resolution": self.grid_resolution,
            "rooms": self.rooms,
            "total_objects": len(self.all_objects),
            "dynamic_tile_mapping": self.dynamic_tile_map
        }

        # Save JSON structure
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Save grid map
        grid = self.create_grid_map()
        np.savetxt(map_file, grid, fmt='%d')


def parse_pose_from_filename(filename: str) -> Dict:
    """
    Parse pose and yaw from filename like 'x0000y0200z1500yaw4712389_dets.json'

    Returns:
        Dict with x, y, z (in cm) and yaw (in radians)
    """
    import re

    # Extract base filename without path and extension
    base = os.path.basename(filename).replace('_dets.json', '')

    # Parse x, y, z values (they're in mm, convert to meters)
    x_match = re.search(r'x(-?\d+)', base)
    y_match = re.search(r'y(-?\d+)', base)
    z_match = re.search(r'z(-?\d+)', base)
    yaw_match = re.search(r'yaw(\d+)', base)

    pose = {}
    if x_match:
        pose['x'] = int(x_match.group(1)) / 1000.0  # Convert mm to m
    if y_match:
        pose['y'] = int(y_match.group(1)) / 1000.0  # Convert mm to m
    if z_match:
        pose['z'] = int(z_match.group(1)) / 1000.0  # Convert mm to m
    if yaw_match:
        # Yaw appears to be in units of 0.000001 radians
        pose['yaw'] = int(yaw_match.group(1)) / 1000000.0

    return pose


def process_files():
    """Process all detection files and return count."""

    # Directory containing the JSON files
    bbox_dir = "/home/user/PycharmProjects/TheAgency/src/room_mapping/ingest_out"

    # Find all detection JSON files
    json_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))

    if not json_files:
        return 0

    # Create unifier for a 2.5x2.5 meter room with 10cm resolution
    unifier = RoomUnifier(
        house_width_m=2.5,
        house_height_m=2.5,
        grid_resolution=0.1  # 10cm per grid cell for better precision
    )

    # Set camera position in the center of the room
    default_camera_x_m = 1.25  # Center X (2.5/2)
    default_camera_y_m = 1.25  # Center Y (2.5/2)

    # Camera parameters
    camera_height_m = 0.5  # Camera height from ground
    camera_fov_h = 70  # Horizontal field of view in degrees
    camera_fov_v = 50  # Vertical field of view in degrees
    fixed_distance_m = 1.0  # All objects at 1 meter distance

    # Process each JSON file
    for json_file in sorted(json_files):
        # Load the detection data
        try:
            with open(json_file, 'r') as f:
                scan_data = json.load(f)
        except:
            continue

        # Parse pose from filename
        pose = parse_pose_from_filename(json_file)

        # Calculate camera position in room coordinates
        camera_x_m = default_camera_x_m + pose.get('x', 0)
        camera_y_m = default_camera_y_m + pose.get('y', 0)
        camera_z_m = pose.get('z', camera_height_m)
        yaw = pose.get('yaw', 0)

        # Get image dimensions if available
        if 'image' in scan_data:
            frame_width = scan_data['image'].get('width', 1280)
            frame_height = scan_data['image'].get('height', 720)
        else:
            frame_width = 1280
            frame_height = 720

        # Add scan to unifier with fixed distance
        unifier.add_scan(
            scan_data=scan_data,
            camera_x_m=camera_x_m,
            camera_y_m=camera_y_m,
            camera_height_m=camera_z_m,
            camera_fov_h=camera_fov_h,
            camera_fov_v=camera_fov_v,
            frame_width=frame_width,
            frame_height=frame_height,
            room_name="main_room",
            room_dims_m=(2.5, 2.5),  # Room dimensions
            yaw=yaw,
            fixed_distance_m=fixed_distance_m  # Fixed 1 meter distance
        )

    # Save results
    unifier.save()

    return len(json_files)


def main():
    """Monitor directory and process files when changes detected."""

    print("Room Unifier - Dynamic Tile & Pixel-Based Size Calculation")
    print("=" * 50)
    print("Configuration:")
    print("- Room size: 2.5m x 2.5m")
    print("- Fixed object distance: 1.0m")
    print("- Resolution: 1280x720")
    print("- Grid resolution: 10cm per cell")
    print("=" * 50)
    print("Monitoring for new detection files...")
    print("Press Ctrl+C to stop\n")

    # Directory to monitor
    bbox_dir = "/home/user/PycharmProjects/TheAgency/src/room_mapping/ingest_out"

    # Track processed files
    last_file_count = 0
    check_interval = 2  # Check every 2 seconds

    try:
        while True:
            # Get current file count
            current_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))
            current_count = len(current_files)

            # Check if there are new files
            if current_count != last_file_count:
                if current_count > 0:
                    print(
                        f"\n[{time.strftime('%H:%M:%S')}] Detected {current_count} files (change from {last_file_count})")
                    print("Processing...")

                    # Process all files
                    processed = process_files()

                    if processed > 0:
                        print(f"Processed {processed} files")
                        print(f"Updated unified_rooms.json and house_map.txt")

                        # Load and display dynamic tile info
                        try:
                            with open("unified_rooms.json", 'r') as f:
                                data = json.load(f)
                                if "dynamic_tile_mapping" in data and data["dynamic_tile_mapping"]:
                                    print("\nDynamic tiles created:")
                                    for obj_class, tile_id in data["dynamic_tile_mapping"].items():
                                        print(f"  - {obj_class}: Tile #{tile_id}")
                        except:
                            pass

                    last_file_count = current_count
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No detection files found")
                    last_file_count = 0

            # Wait before next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print("Final outputs: unified_rooms.json, house_map.txt")


if __name__ == "__main__":
    main()