#!/usr/bin/env python3
"""
pixel_room_mapper.py - Simplified dual-mode room mapper
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent)  # Goes up to workspace level

class DynamicTileManager:
    """Manages dynamic tile types."""

    def __init__(self, existing_registry=None):
        # Reserved tile types
        self.FREE_SPACE = 0
        self.WALL = 1
        self.CAMERA = 2
        self.DOOR = 3

        # Initialize tile registry from existing or create new
        if existing_registry:
            self.tile_registry = existing_registry.copy()
            # Find the highest tile ID to continue numbering
            self.next_tile_id = max(existing_registry.values()) + 1
        else:
            self.tile_registry = {
                'free_space': self.FREE_SPACE,
                'wall': self.WALL,
                'camera': self.CAMERA,
                'door': self.DOOR
            }
            self.next_tile_id = 4

        # Build reverse mapping
        self.id_to_name = {v: k for k, v in self.tile_registry.items()}

    def get_tile_type(self, object_class: str) -> int:
        """Get or create tile type for object class."""
        obj_key = object_class.lower().strip()

        if obj_key not in self.tile_registry:
            self.tile_registry[obj_key] = self.next_tile_id
            self.id_to_name[self.next_tile_id] = obj_key
            self.next_tile_id += 1

        return self.tile_registry[obj_key]

    def get_overlap_tile_type(self, existing_tile_id: int, new_object_class: str) -> int:
        """Create or get tile type for overlapping objects."""
        existing_name = self.id_to_name.get(existing_tile_id, "unknown")
        new_name = new_object_class.lower().strip()

        # Skip creating overlap for reserved types
        if existing_tile_id in [self.FREE_SPACE, self.WALL, self.CAMERA, self.DOOR]:
            return self.get_tile_type(new_object_class)

        # Same object type = just keep existing
        if existing_name == new_name:
            return existing_tile_id

        # Already an overlap? Just keep it
        if " and " in existing_name:
            return existing_tile_id

        # Two different objects = create simple overlap
        names = sorted([existing_name, new_name])
        overlap_name = f"{names[0]} and {names[1]}"
        return self.get_tile_type(overlap_name)

    def get_all_tiles(self) -> Dict:
        """Return all registered tiles."""
        return self.tile_registry.copy()


class PixelRoomMapper:
    """Simplified room mapper."""

    def __init__(self,
                 mode: str = "standalone",
                 room_width_m: float = 2.5,
                 room_height_m: float = 2.5,
                 grid_resolution: float = 0.1,
                 existing_map_file: Optional[str] = None,
                 existing_json_file: Optional[str] = None,
                 room_bbox: Optional[Tuple[int, int, int, int]] = None,
                 room_name: str = "main_room",
                 camera_fov_h: float = 100,
                 camera_fov_v: float = 50):
        """Initialize the room mapper."""

        self.mode = mode
        self.room_name = room_name
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        # Load existing data if JSON provided
        existing_registry = None
        self.existing_rooms = {}
        if existing_json_file and os.path.exists(existing_json_file):
            with open(existing_json_file, 'r') as f:
                existing_data = json.load(f)
                existing_registry = existing_data.get("tile_registry", None)
                # Load existing rooms
                self.existing_rooms = existing_data.get("rooms", {})

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

            # Room BBOX for standalone (full room)
            self.room_bbox = (0, 0, self.grid_width, self.grid_height)

            # Map dimensions same as room
            self.map_width = self.grid_width
            self.map_height = self.grid_height

        elif mode == "existing_map":
            # Existing map mode
            if not existing_map_file or not room_bbox:
                raise ValueError("existing_map mode requires map file and room bbox")

            # Load existing map
            self.existing_grid = np.loadtxt(existing_map_file, dtype=np.int8)

            # Store room bbox directly
            self.room_bbox = room_bbox
            x1, y1, x2, y2 = room_bbox

            # Calculate room dimensions from bbox
            room_width_cells = x2 - x1
            room_height_cells = y2 - y1

            # Calculate grid resolution based on known room size
            self.room_width_m = room_width_m
            self.room_height_m = room_height_m
            self.grid_resolution = (self.room_width_m / room_width_cells +
                                    self.room_height_m / room_height_cells) / 2

            # Room grid dimensions
            self.grid_width = room_width_cells
            self.grid_height = room_height_cells

            # Camera at center of room
            self.camera_x_m = self.room_width_m / 2
            self.camera_y_m = self.room_height_m - 0.3
            print(self.camera_x_m, self.camera_y_m)
            # Map dimensions from existing map
            self.map_height, self.map_width = self.existing_grid.shape

        # Tile manager with existing registry if available
        self.tiles = DynamicTileManager(existing_registry)

        # Storage
        self.all_objects = []

        # Fixed distance assumption
        self.FIXED_DISTANCE = 1.0

    def estimate_object_size_from_pixels(self,
                                         bbox: List[int],
                                         frame_width: int,
                                         frame_height: int,
                                         object_class: str = "") -> Tuple[float, float]:
        """Estimate object size."""
        h_ratio = (bbox[2] - bbox[0]) / frame_width
        v_ratio = (bbox[3] - bbox[1]) / frame_height

        visible_width = 2 * self.FIXED_DISTANCE * math.tan(self.camera_fov_h / 2)
        visible_height = 2 * self.FIXED_DISTANCE * math.tan(self.camera_fov_v / 2)

        h_object_meters = h_ratio * visible_width
        v_object_meters = v_ratio * visible_height

        # # Minimum and maximum sizes
        h_object_meters = max(0.1, min(h_object_meters, self.room_width_m / 3))
        v_object_meters = max(0.1, min(v_object_meters, self.room_height_m / 3))

        return h_object_meters, v_object_meters

    def calculate_object_position(self,
                                  bbox: List[int],
                                  yaw: float,
                                  frame_width: int) -> Tuple[float, float]:
        """Calculate object position."""
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        angle_offset = -((bbox_center_x / frame_width) - 0.5) * self.camera_fov_h
        object_angle = yaw + angle_offset

        # ROTATED 90 DEGREES COUNTERCLOCKWISE:
        # Original: angle 0 was up (north), now angle 0 is right (east)
        # So we use cos for x-offset and sin for y-offset (swapped from original)
        obj_x_m = self.camera_x_m + self.FIXED_DISTANCE * math.cos(object_angle)
        obj_y_m = self.camera_y_m - self.FIXED_DISTANCE * math.sin(object_angle)

        # Clamp position to room boundaries
        margin = 0.2
        obj_x_m = max(margin, min(self.room_width_m - margin, obj_x_m))
        obj_y_m = max(margin, min(self.room_height_m - margin, obj_y_m))

        return obj_x_m, obj_y_m

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert meters to grid coordinates."""
        grid_x = int(x_m / self.grid_resolution)
        grid_y = int(y_m / self.grid_resolution)
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        return grid_x, grid_y

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        """Add a scan to the room map."""
        # Try new structure first (nanoowl.result)
        if 'nanoowl' in scan_data and 'result' in scan_data['nanoowl']:
            result = scan_data['nanoowl']['result']
            if 'image' in result:
                frame_width = result['image'].get('width', 1280)
                frame_height = result['image'].get('height', 720)
            else:
                frame_width = 1280
                frame_height = 720
            detections = result.get('detections', [])
        # Fall back to old structure
        elif 'image' in scan_data:
            frame_width = scan_data['image'].get('width', 1280)
            frame_height = scan_data['image'].get('height', 720)
            detections = scan_data.get('detections', [])
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

            tile_type = self.tiles.get_tile_type(obj_class)

            # Calculate position
            obj_x_m, obj_y_m = self.calculate_object_position(
                bbox, yaw, frame_width
            )

            # Estimate size
            width_m, height_m = self.estimate_object_size_from_pixels(
                bbox, frame_width, frame_height, obj_class
            )

            # Constrain size
            max_width = min(
                obj_x_m - 0.1,
                self.room_width_m - obj_x_m - 0.1
            ) * 2
            max_height = min(
                obj_y_m - 0.1,
                self.room_height_m - obj_y_m - 0.1
            ) * 2

            width_m = min(width_m, max_width)
            height_m = min(height_m, max_height)

            # Convert to grid coordinates
            obj_grid_x, obj_grid_y = self.meters_to_grid(obj_x_m, obj_y_m)

            # Calculate object bbox in grid coords
            width_cells = max(1, int(width_m / self.grid_resolution))
            height_cells = max(1, int(height_m / self.grid_resolution))

            x1 = obj_grid_x - width_cells // 2
            y1 = obj_grid_y - height_cells // 2
            x2 = x1 + width_cells
            y2 = y1 + height_cells

            # Add offset for existing map mode
            if self.mode == "existing_map":
                x1 += self.room_bbox[0]
                y1 += self.room_bbox[1]
                x2 += self.room_bbox[0]
                y2 += self.room_bbox[1]

            # Store simplified object info
            obj_info = {
                "type": obj_class,
                "tile_type": tile_type,
                "bbox": [x1, y1, x2, y2]
            }

            self.all_objects.append(obj_info)

    def create_grid_map(self) -> np.ndarray:
        """Create or update 2D grid map."""
        if self.mode == "standalone":
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
            grid = self.existing_grid.copy()
            x1, y1, x2, y2 = self.room_bbox

            # Clear room area (except walls)
            for y in range(y1 + 1, y2 - 1):
                for x in range(x1 + 1, x2 - 1):
                    if y < self.map_height and x < self.map_width:
                        grid[y, x] = self.tiles.FREE_SPACE

        # Place camera
        cam_x, cam_y = self.meters_to_grid(self.camera_x_m, self.camera_y_m)
        if self.mode == "existing_map":
            cam_x += self.room_bbox[0]
            cam_y += self.room_bbox[1]

        if 0 <= cam_x < self.map_width and 0 <= cam_y < self.map_height:
            grid[cam_y, cam_x] = self.tiles.CAMERA

        # Place objects
        for obj in self.all_objects:
            x1, y1, x2, y2 = obj["bbox"]
            obj_class = obj["type"]

            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Check bounds
                    if 0 < x < self.map_width - 1 and 0 < y < self.map_height - 1:
                        existing_tile = grid[y, x]

                        # Skip special tiles
                        if existing_tile in [self.tiles.WALL, self.tiles.CAMERA, self.tiles.DOOR]:
                            continue

                        # Handle overlap
                        if existing_tile != self.tiles.FREE_SPACE:
                            overlap_tile = self.tiles.get_overlap_tile_type(existing_tile, obj_class)
                            grid[y, x] = overlap_tile
                        else:
                            tile_type = self.tiles.get_tile_type(obj_class)
                            grid[y, x] = tile_type

        return grid

    def save(self, json_file: str = "data/unified_rooms.json",
             map_file: str = "data/house_map.txt"):
        """Save the room structure and grid map."""

        # Create grid first
        grid = self.create_grid_map()

        # Calculate camera position in grid
        cam_grid_x, cam_grid_y = self.meters_to_grid(self.camera_x_m, self.camera_y_m)
        if self.mode == "existing_map":
            cam_grid_x += self.room_bbox[0]
            cam_grid_y += self.room_bbox[1]

        # Start with existing rooms
        rooms = self.existing_rooms.copy()

        # Add or update current room
        rooms[self.room_name] = {
            "name": self.room_name,
            "camera_position": [cam_grid_x, cam_grid_y],
            "bbox": list(self.room_bbox),  # [x1, y1, x2, y2]
            "objects": self.all_objects,
            "doors": [25, 7]
        }

        # Simplified output
        output = {
            "house_dimensions_m": {
                "width": self.map_width * self.grid_resolution,
                "height": self.map_height * self.grid_resolution
            },
            "grid_resolution": self.grid_resolution,
            "rooms": rooms,
            "tile_registry": self.tiles.get_all_tiles()
        }

        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)

        np.savetxt(map_file, grid, fmt='%d')

        # Count existing vs new tiles
        existing_count = sum(
            1 for tid in self.tiles.tile_registry.values() if tid < self.tiles.next_tile_id - len(self.all_objects))
        new_count = len(self.tiles.tile_registry) - existing_count

        print(f"\nSaved {len(self.all_objects)} objects to room '{self.room_name}'")
        print(f"Total rooms: {len(rooms)}")
        print(f"Tile types: {existing_count} existing + {new_count} new = {len(self.tiles.tile_registry)} total")


def get_yaw_from_json(scan_data: Dict) -> float:
    """Extract yaw from JSON data's pose field."""
    if 'pose' in scan_data and 'yaw' in scan_data['pose']:
        yaw = scan_data['pose']['yaw']
        print(f"  Found yaw in JSON: {yaw} radians ({math.degrees(yaw):.1f} degrees)")
        return yaw
    print("  Warning: No yaw found in JSON, defaulting to 0.0")
    return 0.0


def process_files(mode="standalone", existing_map=None, existing_json=None, room_bbox=None, room_name="main_room"):
    """Process all detection files."""
    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    # Look for ALL .json files, not just *_dets.json
    json_files = glob.glob(os.path.join(bbox_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {bbox_dir}")
        return 0

    print(f"Found {len(json_files)} JSON files to process")

    # Create mapper
    if mode == "standalone":
        mapper = PixelRoomMapper(
            mode="standalone",
            room_width_m=2.5,
            room_height_m=2.0,
            grid_resolution=0.05,
            existing_json_file=existing_json,
            room_name=room_name,
            camera_fov_h=100,
            camera_fov_v=50
        )
    else:
        mapper = PixelRoomMapper(
            mode="existing_map",
            room_width_m=2.5,
            room_height_m=2.0,
            existing_map_file=existing_map,
            existing_json_file=existing_json,
            room_bbox=room_bbox,
            room_name=room_name,
            camera_fov_h=50,
            camera_fov_v=60
        )

    # Process each file
    for json_file in sorted(json_files):
        try:
            print(f"Processing: {os.path.basename(json_file)}")
            with open(json_file, 'r') as f:
                scan_data = json.load(f)
            # Get yaw from JSON data instead of filename
            yaw = get_yaw_from_json(scan_data)
            mapper.add_scan(scan_data, yaw)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    mapper.save()
    return len(json_files)


def main():
    """Monitor and process detection files."""
    # Configuration
    mode = "standalone"
    existing_map = os.path.join(BASE_PATH, "room_mapping/office_map.txt")
    existing_json = os.path.join(BASE_PATH, "room_mapping/office.json")
    room_bbox = (23, 10, 40, 24)
    room_name = "MAMAD"  # Specify which room to update

    if room_bbox is not None and existing_map is not None:
        mode = "existing_map"
        print(f"Mode: Existing Map")
        print(f"Map file: {existing_map}")
        if existing_json and os.path.exists(existing_json):
            print(f"Existing JSON: {existing_json}")
        print(f"Room bbox: {room_bbox}")
        print(f"Room name: {room_name}")
    else:
        print("Mode: Standalone")
        if existing_json and os.path.exists(existing_json):
            print(f"Using existing JSON: {existing_json}")
        print(f"Room name: {room_name}")

    print("\nMonitoring for detection files...")
    print("Press Ctrl+C to stop\n")

    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    last_file_count = 0
    check_interval = 2

    try:
        while True:
            # Look for ALL .json files, not just *_dets.json
            current_files = glob.glob(os.path.join(bbox_dir, "*.json"))
            current_count = len(current_files)

            if current_count != last_file_count:
                if current_count > 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {current_count} files")
                    print("Processing...")

                    processed = process_files(mode, existing_map, existing_json, room_bbox, room_name)

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