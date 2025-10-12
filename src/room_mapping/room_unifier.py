#!/usr/bin/env python3
"""
room_unifier.py - Block 3: Room Scanner and Unifier

Processes BBOX data from multiple camera positions, calculates real distances,
and unifies them into a single global house structure.
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple, Optional
from tile_definitions import OBJECT_TO_TILE, OBJECT_HEIGHTS, TileType, OBJECT_SIZES


class RoomUnifier:
    """Processes and unifies room scans using camera geometry."""

    def __init__(self,
                 house_width_m: float = 20.0,
                 house_height_m: float = 20.0,
                 grid_resolution: float = 0.2):
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
        self.all_objects = []  # Store all objects with their positions

    def estimate_distance(self, bbox: List[int], object_class: str,
                          camera_height_m: float, camera_fov_v: float,
                          frame_height: int) -> float:
        """
        Estimate object distance using perspective projection.

        Args:
            bbox: [x1, y1, x2, y2] in pixels
            object_class: Type of object (for height lookup)
            camera_height_m: Camera height from ground in meters
            camera_fov_v: Vertical field of view in degrees
            frame_height: Camera frame height in pixels

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height_px = y2 - y1

        # Calculate focal length for this camera
        focal_length_px = (frame_height / 2) / math.tan(math.radians(camera_fov_v / 2))

        # Get real-world height of object
        real_object_height = OBJECT_HEIGHTS.get(object_class.lower(), 1.0)

        # Basic distance from perspective projection
        if bbox_height_px > 0:
            distance = (real_object_height * focal_length_px) / bbox_height_px
        else:
            distance = 5.0  # Default fallback

        # Adjust for camera height and vertical position
        bbox_center_y = (y1 + y2) / 2
        vertical_position = bbox_center_y / frame_height

        # Apply ground plane constraint for objects in lower frame
        if vertical_position > 0.5:
            angle_to_base = math.atan2(camera_height_m, distance)
            ground_distance = camera_height_m / math.tan(angle_to_base)
            weight = (vertical_position - 0.5) * 2
            distance = distance * (1 - weight * 0.3)

        return distance

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
                 camera_height_m: float = 1.5,
                 camera_fov_h: float = 60,
                 camera_fov_v: float = 45,
                 frame_width: int = 1280,
                 frame_height: int = 720,
                 room_name: Optional[str] = None,
                 room_dims_m: Optional[Tuple[float, float]] = None,
                 yaw: float = 0.0):
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

            # Skip unknown objects
            tile_type = OBJECT_TO_TILE.get(obj_class)
            if not tile_type and 'door' not in obj_class:
                continue

            # Estimate real distance in meters using this scan's camera params
            distance_m = self.estimate_distance(bbox, obj_class, camera_height_m,
                                                camera_fov_v, frame_height)

            # Calculate horizontal angle offset
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            angle_offset = ((bbox_center_x / frame_width) - 0.5) * fov_h_rad
            object_angle = yaw + math.pi + angle_offset

            # Calculate object position in meters
            obj_x_m = camera_x_m - distance_m * math.sin(object_angle)
            obj_y_m = camera_y_m + distance_m * math.cos(object_angle)

            # Convert to grid coordinates
            obj_grid_x, obj_grid_y = self.meters_to_grid(obj_x_m, obj_y_m)

            # Create object info
            obj_info = {
                "type": obj_class,
                "tile_type": tile_type if tile_type else TileType.UNKNOWN,
                "location": (obj_grid_x, obj_grid_y),
                "location_m": (round(obj_x_m, 2), round(obj_y_m, 2)),
                "distance_m": round(distance_m, 2),
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
                    "frame_height": frame_height
                },
                "dimensions_m": room_dims_m,  # Can be None
                "objects": detected_objects,
                "doors": detected_doors,
                "scan_yaw": round(yaw, 3)
            }

            # If this room already exists, merge the objects
            if room_name in self.rooms:
                self.rooms[room_name]["objects"].extend(detected_objects)
                self.rooms[room_name]["doors"].extend(detected_doors)
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
        from tile_definitions import OBJECT_SIZES

        # Initialize grid
        grid = np.full((self.grid_height, self.grid_width),
                       TileType.FREE_SPACE, dtype=np.int8)

        # Place all detected objects with proper sizes
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
                # Get object dimensions from OBJECT_SIZES
                obj_dims = OBJECT_SIZES.get(tile_type, (0.5, 0.5, 0.5))
                obj_width_m = obj_dims[0]  # Width in meters
                obj_depth_m = obj_dims[1]  # Depth in meters

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
            "total_objects": len(self.all_objects)
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

    # Create unifier for a 10x10 meter house (adjust as needed)
    unifier = RoomUnifier(
        house_width_m=10.0,
        house_height_m=10.0,
        grid_resolution=0.2  # 20cm per grid cell
    )

    # Set default camera position in the center of the house
    default_camera_x_m = 5.0  # Center X
    default_camera_y_m = 5.0  # Center Y

    # Camera parameters based on typical values
    camera_height_m = 0.3  # Camera height from ground
    camera_fov_h = 70  # Horizontal field of view in degrees
    camera_fov_v = 50  # Vertical field of view in degrees

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

        # Calculate camera position in house coordinates
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

        # Add scan to unifier
        unifier.add_scan(
            scan_data=scan_data,
            camera_x_m=camera_x_m,
            camera_y_m=camera_y_m,
            camera_height_m=camera_z_m,
            camera_fov_h=camera_fov_h,
            camera_fov_v=camera_fov_v,
            frame_width=frame_width,
            frame_height=frame_height,
            room_name="main_room",  # All from the same room
            yaw=yaw
        )

    # Save results
    unifier.save()

    return len(json_files)


def main():
    """Monitor directory and process files when changes detected."""

    print("Room Unifier - Continuous Monitor")
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