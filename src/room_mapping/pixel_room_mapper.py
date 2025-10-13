#!/usr/bin/env python3
"""
pixel_room_mapper.py - Dual-mode room mapper using pixel-based size estimation
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple, Optional


class DynamicTileManager:
    """Manages dynamic tile types and colors."""

    def __init__(self):
        # Reserved tile types
        self.FREE_SPACE = 0
        self.WALL = 1
        self.CAMERA = 2
        self.DOOR = 3  # Reserved for doors

        # Dynamic tile registry
        self.tile_registry = {
            'free_space': self.FREE_SPACE,
            'wall': self.WALL,
            'camera': self.CAMERA,
            'door': self.DOOR
        }

        # Color registry (RGB)
        self.color_registry = {
            self.FREE_SPACE: (200, 200, 200),  # Light gray
            self.WALL: (100, 100, 100),  # Dark gray
            self.CAMERA: (0, 255, 255),  # Cyan
            self.DOOR: (139, 69, 19),  # Brown
        }

        self.next_tile_id = 4  # Start after reserved types (including door)

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
    """Dual-mode room mapper using pixel-based size estimation."""

    def __init__(self,
                 mode: str = "standalone",
                 room_width_m: float = 2.5,
                 room_height_m: float = 2.5,
                 grid_resolution: float = 0.05,
                 existing_map_file: Optional[str] = None,
                 room_bbox: Optional[Tuple[int, int, int, int]] = None,
                 camera_fov_h: float = 100,
                 camera_fov_v: float = 50):
        """Initialize the room mapper."""

        self.mode = mode
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        if mode == "standalone":
            # Original standalone mode
            self.room_width_m = room_width_m
            self.room_height_m = room_height_m
            self.grid_resolution = grid_resolution

            # Grid dimensions
            self.grid_width = int(room_width_m / grid_resolution)
            self.grid_height = int(room_height_m / grid_resolution)

            # Camera at center of room
            self.camera_x_m = room_width_m / 2
            self.camera_y_m = room_height_m / 2

            # Room offset in grid (0,0 for standalone)
            self.room_offset_x = 0
            self.room_offset_y = 0

            # Map dimensions same as room
            self.map_width = self.grid_width
            self.map_height = self.grid_height

        elif mode == "existing_map":
            # Existing map mode
            if not existing_map_file or not room_bbox:
                raise ValueError("existing_map mode requires map file and room bbox")

            # Load existing map
            self.existing_grid = np.loadtxt(existing_map_file, dtype=np.int8)

            # Room location in grid
            x1, y1, x2, y2 = room_bbox
            self.room_offset_x = x1
            self.room_offset_y = y1

            # Calculate room dimensions from bbox
            room_width_cells = x2 - x1
            room_height_cells = y2 - y1

            # Infer grid resolution from JSON if available
            try:
                with open("unified_rooms.json", 'r') as f:
                    structure = json.load(f)
                    self.grid_resolution = structure["grid_resolution"]
            except:
                self.grid_resolution = 0.05  # Default

            # Calculate room dimensions in meters
            self.room_width_m = room_width_cells * self.grid_resolution
            self.room_height_m = room_height_cells * self.grid_resolution

            # Room grid dimensions
            self.grid_width = room_width_cells
            self.grid_height = room_height_cells

            # Camera at center of room
            self.camera_x_m = self.room_width_m / 2
            self.camera_y_m = self.room_height_m / 2

            # Map dimensions from existing map
            self.map_height, self.map_width = self.existing_grid.shape

        # Tile manager
        self.tiles = DynamicTileManager()

        # Storage
        self.all_objects = []
        self.rooms = {}

        # Fixed distance assumption (reduced for 2.5m room)
        self.FIXED_DISTANCE = 0.7

    def estimate_object_size_from_pixels(self,
                                         bbox: List[int],
                                         frame_width: int,
                                         frame_height: int) -> Tuple[float, float]:
        """Estimate object size with proper proportions relative to room."""
        # Calculate pixel ratios
        h_ratio = (bbox[2] - bbox[0]) / frame_width
        v_ratio = (bbox[3] - bbox[1]) / frame_height

        # Scale factor based on camera FOV and distance
        # At 0.7m distance with 100° FOV, visible width ≈ 1.5m
        visible_width = 2 * self.FIXED_DISTANCE * math.tan(self.camera_fov_h / 2)
        visible_height = 2 * self.FIXED_DISTANCE * math.tan(self.camera_fov_v / 2)

        # Calculate actual object size
        h_object_meters = h_ratio * visible_width
        v_object_meters = v_ratio * visible_height

        # Apply scaling factor to better match room proportions
        # Objects typically appear larger in pixel space than actual size
        scale_factor = 0.6  # Adjusted for 0.7m distance

        h_object_meters *= scale_factor
        v_object_meters *= scale_factor

        # Clamp to reasonable sizes (max 1/4 of room dimension for safety)
        h_object_meters = min(h_object_meters, self.room_width_m / 4)
        v_object_meters = min(v_object_meters, self.room_height_m / 4)

        return h_object_meters, v_object_meters

    def calculate_object_position(self,
                                  bbox: List[int],
                                  yaw: float,
                                  frame_width: int) -> Tuple[float, float]:
        """Calculate object position based on bbox center and yaw."""
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        angle_offset = -((bbox_center_x / frame_width) - 0.5) * self.camera_fov_h

        object_angle = yaw + angle_offset

        # Calculate position at fixed distance
        obj_x_m = self.camera_x_m - self.FIXED_DISTANCE * math.sin(object_angle)
        obj_y_m = self.camera_y_m - self.FIXED_DISTANCE * math.cos(object_angle)

        # Clamp position to room boundaries with margin
        margin = 0.2  # 20cm margin from walls
        obj_x_m = max(margin, min(self.room_width_m - margin, obj_x_m))
        obj_y_m = max(margin, min(self.room_height_m - margin, obj_y_m))

        return obj_x_m, obj_y_m

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert meters to grid coordinates."""
        # Local room coordinates
        grid_x = int(x_m / self.grid_resolution)
        grid_y = int(y_m / self.grid_resolution)

        # Clamp to room bounds
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))

        return grid_x, grid_y

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        """Add a scan to the room map."""
        if 'image' in scan_data:
            frame_width = scan_data['image'].get('width', 1280)
            frame_height = scan_data['image'].get('height', 720)
        else:
            frame_width = 1280
            frame_height = 720

        detections = scan_data.get('detections', [])

        for detection in detections:
            label = detection.get('label', '').lower()
            obj_class = label.replace('a ', '').replace('an ', '').strip()

            if not obj_class:
                continue

            bbox = detection['bbox']
            confidence = detection.get('score', 1.0)

            tile_type = self.tiles.get_tile_type(obj_class)

            # Calculate position first
            obj_x_m, obj_y_m = self.calculate_object_position(
                bbox, yaw, frame_width
            )

            # Estimate size
            width_m, height_m = self.estimate_object_size_from_pixels(
                bbox, frame_width, frame_height
            )

            # Constrain size based on position to prevent overflow
            max_width = min(
                obj_x_m - 0.1,  # Distance to left wall
                self.room_width_m - obj_x_m - 0.1  # Distance to right wall
            ) * 2
            max_height = min(
                obj_y_m - 0.1,  # Distance to top wall
                self.room_height_m - obj_y_m - 0.1  # Distance to bottom wall
            ) * 2

            width_m = min(width_m, max_width)
            height_m = min(height_m, max_height)

            obj_grid_x, obj_grid_y = self.meters_to_grid(obj_x_m, obj_y_m)

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
        """Create or update 2D grid map."""
        if self.mode == "standalone":
            # Create new grid
            grid = np.full((self.grid_height, self.grid_width),
                           self.tiles.FREE_SPACE, dtype=np.int8)

            # Draw room walls
            for x in range(self.grid_width):
                grid[0, x] = self.tiles.WALL
                grid[self.grid_height - 1, x] = self.tiles.WALL

            for y in range(self.grid_height):
                grid[y, 0] = self.tiles.WALL
                grid[y, self.grid_width - 1] = self.tiles.WALL

        else:  # existing_map mode
            # Copy existing map
            grid = self.existing_grid.copy()

            # Clear room area (except walls)
            for y in range(self.room_offset_y + 1, self.room_offset_y + self.grid_height - 1):
                for x in range(self.room_offset_x + 1, self.room_offset_x + self.grid_width - 1):
                    if y < self.map_height and x < self.map_width:
                        grid[y, x] = self.tiles.FREE_SPACE

        # Place camera
        cam_x, cam_y = self.meters_to_grid(self.camera_x_m, self.camera_y_m)
        actual_cam_x = cam_x + self.room_offset_x
        actual_cam_y = cam_y + self.room_offset_y

        if 0 <= actual_cam_x < self.map_width and 0 <= actual_cam_y < self.map_height:
            grid[actual_cam_y, actual_cam_x] = self.tiles.CAMERA

        # Place objects
        for obj in self.all_objects:
            cx, cy = obj["location"]
            tile_type = obj["tile_type"]
            width_m, height_m = obj["size_m"]

            width_cells = max(1, int(width_m / self.grid_resolution))
            height_cells = max(1, int(height_m / self.grid_resolution))

            for dy in range(height_cells):
                for dx in range(width_cells):
                    # Local coordinates
                    local_x = cx - width_cells // 2 + dx
                    local_y = cy - height_cells // 2 + dy

                    # Map coordinates
                    x = local_x + self.room_offset_x
                    y = local_y + self.room_offset_y

                    # Check bounds and don't overwrite walls/camera/doors
                    if (0 < x < self.map_width - 1 and 0 < y < self.map_height - 1 and
                            1 <= local_x < self.grid_width - 1 and 1 <= local_y < self.grid_height - 1):
                        if grid[y, x] not in [self.tiles.WALL, self.tiles.CAMERA, self.tiles.DOOR]:
                            grid[y, x] = tile_type

        return grid

    def save(self, json_file: str = "unified_rooms.json",
             map_file: str = "house_map.txt"):
        """Save the room structure and grid map."""

        # Adjust camera position for map coordinates
        cam_grid_x, cam_grid_y = self.meters_to_grid(self.camera_x_m, self.camera_y_m)
        cam_map_x = cam_grid_x + self.room_offset_x
        cam_map_y = cam_grid_y + self.room_offset_y

        self.rooms["main_room"] = {
            "name": "main_room",
            "camera_position": (cam_map_x, cam_map_y),
            "camera_position_m": (self.camera_x_m, self.camera_y_m),
            "dimensions_m": (self.room_width_m, self.room_height_m),
            "room_offset": (self.room_offset_x, self.room_offset_y),
            "objects": self.all_objects,
            "doors": []
        }

        output = {
            "house_dimensions_m": {
                "width": self.map_width * self.grid_resolution,
                "height": self.map_height * self.grid_resolution
            },
            "grid_resolution": self.grid_resolution,
            "rooms": self.rooms,
            "total_objects": len(self.all_objects),
            "tile_registry": self.tiles.get_all_tiles()
        }

        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)

        grid = self.create_grid_map()
        np.savetxt(map_file, grid, fmt='%d')

        print(f"Saved {len(self.all_objects)} objects with {len(self.tiles.get_all_tiles())} tile types")
        if self.mode == "existing_map":
            print(f"Room placed at offset ({self.room_offset_x}, {self.room_offset_y})")


def parse_yaw_from_filename(filename: str) -> float:
    """Extract yaw from filename."""
    import re
    base = os.path.basename(filename)
    yaw_match = re.search(r'yaw(\d+)', base)

    if yaw_match:
        return int(yaw_match.group(1)) / 1000000.0
    return 0.0


def process_files(mode="standalone", existing_map=None, room_bbox=None):
    """Process all detection files."""

    bbox_dir = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/ingest_out"
    json_files = glob.glob(os.path.join(bbox_dir, "*_dets.json"))

    if not json_files:
        return 0

    # Create mapper based on mode
    if mode == "standalone":
        mapper = PixelRoomMapper(
            mode="standalone",
            room_width_m=2.5,
            room_height_m=2.5,
            grid_resolution=0.05,
            camera_fov_h=100,
            camera_fov_v=50
        )
    else:
        mapper = PixelRoomMapper(
            mode="existing_map",
            existing_map_file=existing_map,
            room_bbox=room_bbox,
            camera_fov_h=100,
            camera_fov_v=50
        )

    # Process each file
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                scan_data = json.load(f)

            yaw = parse_yaw_from_filename(json_file)
            mapper.add_scan(scan_data, yaw)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    mapper.save()
    return len(json_files)


def main():
    """Monitor and process detection files."""

    import sys

    # Check for mode argument
    mode = "standalone"
    existing_map = "/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/office_map.txt"
    room_bbox = (25, 0, 42, 21)

    if room_bbox is not None and existing_map is not None:
        mode = "existing_map"

        print(f"Mode: Existing Map")
        print(f"Map file: {existing_map}")
        print(f"Room bbox: {room_bbox}")
    else:
        print("Mode: Standalone")
        print("  - Room size: 2.5m x 2.5m")

    print("\nConfiguration:")
    print("  - Camera: Center of room")
    print("  - Distance: 0.7m (fixed)")
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

                    processed = process_files(mode, existing_map, room_bbox)

                    if processed > 0:
                        print(f"Processed {processed} files")
                        print(f"Updated unified_rooms.json and house_map.txt")

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