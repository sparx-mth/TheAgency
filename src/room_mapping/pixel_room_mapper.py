#!/usr/bin/env python3
"""
pixel_room_mapper.py - Simplified room mapper using pixel-based size estimation

Maps objects in a room using only pixel sizes and fixed assumptions:
- All objects are 1 meter from camera
- Room is 2.5m x 2.5m
- Camera is in center of room
- Uses dynamic tile generation
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple


class DynamicTileManager:
    """Manages dynamic tile types and colors."""

    def __init__(self):
        # Reserved tile types
        self.FREE_SPACE = 0
        self.WALL = 1
        self.CAMERA = 2

        # Dynamic tile registry
        self.tile_registry = {
            'free_space': self.FREE_SPACE,
            'wall': self.WALL,
            'camera': self.CAMERA
        }

        # Color registry (RGB)
        self.color_registry = {
            self.FREE_SPACE: (200, 200, 200),  # Light gray
            self.WALL: (100, 100, 100),  # Dark gray
            self.CAMERA: (0, 255, 255),  # Cyan
        }

        self.next_tile_id = 3  # Start after reserved types

    def get_tile_type(self, object_class: str) -> int:
        """Get or create tile type for object class."""
        # Normalize object class
        obj_key = object_class.lower().strip()

        if obj_key not in self.tile_registry:
            # Create new tile type
            self.tile_registry[obj_key] = self.next_tile_id

            # Generate a color based on hash for consistency
            hash_val = hash(obj_key)
            r = (hash_val & 0xFF0000) >> 16
            g = (hash_val & 0x00FF00) >> 8
            b = hash_val & 0x0000FF

            # Ensure colors are visible (not too dark)
            r = max(50, r)
            g = max(50, g)
            b = max(50, b)

            self.color_registry[self.next_tile_id] = (r, g, b)
            self.next_tile_id += 1

        return self.tile_registry[obj_key]

    def get_all_tiles(self) -> Dict:
        """Return all registered tiles."""
        return self.tile_registry.copy()


class PixelRoomMapper:
    """Simple room mapper using pixel-based size estimation."""

    def __init__(self,
                 room_width_m: float = 2.5,
                 room_height_m: float = 2.5,
                 grid_resolution: float = 0.05,  # 5cm per cell for better resolution
                 camera_fov_h: float = 100,  # Horizontal FOV in degrees
                 camera_fov_v: float = 50):  # Vertical FOV in degrees
        """Initialize the room mapper."""

        self.room_width_m = room_width_m
        self.room_height_m = room_height_m
        self.grid_resolution = grid_resolution

        # Camera parameters
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        # Camera position (center of room)
        self.camera_x_m = room_width_m / 2
        self.camera_y_m = room_height_m / 2

        # Grid dimensions
        self.grid_width = int(room_width_m / grid_resolution)
        self.grid_height = int(room_height_m / grid_resolution)

        # Tile manager
        self.tiles = DynamicTileManager()

        # Storage
        self.all_objects = []
        self.rooms = {}

        # Fixed distance assumption
        self.FIXED_DISTANCE = 1.0  # All objects 1 meter from camera

    def estimate_object_size_from_pixels(self,
                                         bbox: List[int],
                                         frame_width: int,
                                         frame_height: int) -> Tuple[float, float]:
        """
        Estimate object size based on pixel dimensions at fixed distance.

        Args:
            bbox: [x1, y1, x2, y2] in pixels
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            (width_m, height_m) estimated size in meters
        """
        # Calculate object pixel dimensions
        h_object_pixels = bbox[2] - bbox[0]  # width in pixels
        v_object_pixels = bbox[3] - bbox[1]  # height in pixels

        # Calculate radians per pixel
        h_radians_per_pixel = self.camera_fov_h / frame_width
        v_radians_per_pixel = self.camera_fov_v / frame_height

        # Calculate angular size
        h_object_radians = h_object_pixels * h_radians_per_pixel
        v_object_radians = v_object_pixels * v_radians_per_pixel

        # Calculate physical size at fixed distance
        # Using formula: size = 2 * distance * tan(angle/2)
        h_object_meters = 2 * self.FIXED_DISTANCE * math.tan(h_object_radians / 2)
        v_object_meters = 2 * self.FIXED_DISTANCE * math.tan(v_object_radians / 2)

        return h_object_meters, v_object_meters

    def calculate_object_position(self,
                                  bbox: List[int],
                                  yaw: float,
                                  frame_width: int) -> Tuple[float, float]:
        """
        Calculate object position based on bbox center and yaw.

        Args:
            bbox: [x1, y1, x2, y2] in pixels
            yaw: Camera yaw in radians
            frame_width: Frame width in pixels

        Returns:
            (x_m, y_m) position in room coordinates
        """
        # Calculate precise horizontal angle offset from center
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        # Invert angle offset for correct left/right mapping
        angle_offset = -((bbox_center_x / frame_width) - 0.5) * self.camera_fov_h

        # Direct angle calculation (no pi addition needed)
        object_angle = yaw + angle_offset

        # Calculate position at fixed distance
        # Use standard coordinate system: yaw=0 points up (north)
        obj_x_m = self.camera_x_m - self.FIXED_DISTANCE * math.sin(object_angle)
        obj_y_m = self.camera_y_m - self.FIXED_DISTANCE * math.cos(object_angle)

        return obj_x_m, obj_y_m

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert meters to grid coordinates."""
        grid_x = int(x_m / self.grid_resolution)
        grid_y = int(y_m / self.grid_resolution)

        # Clamp to bounds
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))

        return grid_x, grid_y

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        """
        Add a scan to the room map.

        Args:
            scan_data: Detection data with 'detections'
            yaw: Camera yaw in radians
        """
        # Get frame dimensions
        if 'image' in scan_data:
            frame_width = scan_data['image'].get('width', 1280)
            frame_height = scan_data['image'].get('height', 720)
        else:
            frame_width = 1280
            frame_height = 720

        # Process detections
        detections = scan_data.get('detections', [])

        for detection in detections:
            # Get object class
            label = detection.get('label', '').lower()
            obj_class = label.replace('a ', '').replace('an ', '').strip()

            if not obj_class:
                continue

            bbox = detection['bbox']
            confidence = detection.get('score', 1.0)

            # Get or create tile type
            tile_type = self.tiles.get_tile_type(obj_class)

            # Estimate size from pixels
            width_m, height_m = self.estimate_object_size_from_pixels(
                bbox, frame_width, frame_height
            )

            # Calculate position
            obj_x_m, obj_y_m = self.calculate_object_position(
                bbox, yaw, frame_width
            )

            # Convert to grid
            obj_grid_x, obj_grid_y = self.meters_to_grid(obj_x_m, obj_y_m)

            # Store object
            obj_info = {
                "type": obj_class,
                "tile_type": tile_type,
                "location": (obj_grid_x, obj_grid_y),
                "location_m": (round(obj_x_m, 2), round(obj_y_m, 2)),
                "size_m": (round(width_m, 2), round(height_m, 2)),
                "confidence": round(confidence, 2),
                "yaw": round(yaw, 3)
            }

            self.all_objects.append(obj_info)

    def create_grid_map(self) -> np.ndarray:
        """Create 2D grid map of the room."""
        # Initialize grid with free space
        grid = np.full((self.grid_height, self.grid_width),
                       self.tiles.FREE_SPACE, dtype=np.int8)

        # Draw room walls
        # Top and bottom walls
        for x in range(self.grid_width):
            grid[0, x] = self.tiles.WALL
            grid[self.grid_height - 1, x] = self.tiles.WALL

        # Left and right walls
        for y in range(self.grid_height):
            grid[y, 0] = self.tiles.WALL
            grid[y, self.grid_width - 1] = self.tiles.WALL

        # Place camera in center
        cam_x, cam_y = self.meters_to_grid(self.camera_x_m, self.camera_y_m)
        grid[cam_y, cam_x] = self.tiles.CAMERA

        # Place detected objects
        for obj in self.all_objects:
            cx, cy = obj["location"]
            tile_type = obj["tile_type"]
            width_m, height_m = obj["size_m"]

            # Convert size to grid cells
            width_cells = max(1, int(width_m / self.grid_resolution))
            height_cells = max(1, int(height_m / self.grid_resolution))

            # Place object (centered on position)
            for dy in range(height_cells):
                for dx in range(width_cells):
                    x = cx - width_cells // 2 + dx
                    y = cy - height_cells // 2 + dy

                    if 0 < x < self.grid_width - 1 and 0 < y < self.grid_height - 1:
                        # Don't overwrite walls or camera
                        if grid[y, x] not in [self.tiles.WALL, self.tiles.CAMERA]:
                            grid[y, x] = tile_type

        return grid

    def save(self, json_file: str = "unified_rooms.json",
             map_file: str = "house_map.txt"):
        """Save the room structure and grid map."""

        # Create room info
        self.rooms["main_room"] = {
            "name": "main_room",
            "camera_position": self.meters_to_grid(self.camera_x_m, self.camera_y_m),
            "camera_position_m": (self.camera_x_m, self.camera_y_m),
            "dimensions_m": (self.room_width_m, self.room_height_m),
            "objects": self.all_objects,
            "doors": []  # No door detection in simple version
        }

        # Prepare output
        output = {
            "house_dimensions_m": {
                "width": self.room_width_m,
                "height": self.room_height_m
            },
            "grid_resolution": self.grid_resolution,
            "rooms": self.rooms,
            "total_objects": len(self.all_objects),
            "tile_registry": self.tiles.get_all_tiles()
        }

        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)

        # Save grid map
        grid = self.create_grid_map()
        np.savetxt(map_file, grid, fmt='%d')

        print(f"Saved {len(self.all_objects)} objects with {len(self.tiles.get_all_tiles())} tile types")


def parse_yaw_from_filename(filename: str) -> float:
    """Extract yaw from filename."""
    import re
    base = os.path.basename(filename)
    yaw_match = re.search(r'yaw(\d+)', base)

    if yaw_match:
        # Convert from units of 0.000001 radians
        return int(yaw_match.group(1)) / 1000000.0
    return 0.0


def process_files():
    """Process all detection files."""

    # Directory with detection files
    bbox_dir = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/ingest_out"
    json_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))

    if not json_files:
        return 0

    # Create mapper
    mapper = PixelRoomMapper(
        room_width_m=2.5,
        room_height_m=2.5,
        grid_resolution=0.05,  # 5cm cells
        camera_fov_h=100,  # From your notebook
        camera_fov_v=50  # From your notebook
    )

    # Process each file
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                scan_data = json.load(f)

            # Get yaw from filename
            yaw = parse_yaw_from_filename(json_file)

            # Add scan
            mapper.add_scan(scan_data, yaw)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Save results
    mapper.save()
    return len(json_files)


def main():
    """Monitor and process detection files."""

    print("Pixel-Based Room Mapper - Continuous Monitor")
    print("=" * 50)
    print("Configuration:")
    print("  - Room size: 2.5m x 2.5m")
    print("  - Camera: Center of room")
    print("  - Distance: 1m (fixed)")
    print("  - FOV: 100° horizontal, 50° vertical")
    print("  - Dynamic tile generation")
    print("\nMonitoring for detection files...")
    print("Press Ctrl+C to stop\n")

    bbox_dir = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/ingest_out"
    last_file_count = 0
    check_interval = 2

    try:
        while True:
            current_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))
            current_count = len(current_files)

            if current_count != last_file_count:
                if current_count > 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {current_count} files")
                    print("Processing...")

                    processed = process_files()

                    if processed > 0:
                        print(f"✓ Processed {processed} files")
                        print(f"✓ Updated unified_rooms.json and house_map.txt")

                    last_file_count = current_count
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No detection files found")
                    last_file_count = 0

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        print("Outputs: unified_rooms.json, house_map.txt")


if __name__ == "__main__":
    main()