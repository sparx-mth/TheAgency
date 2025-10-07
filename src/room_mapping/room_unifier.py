#!/usr/bin/env python3
"""
room_unifier.py - Block 3: Room Scanner and Unifier

Processes BBOX data from multiple camera positions, calculates real distances,
and unifies them into a single global house structure.
"""

import numpy as np
import json
import math
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
                 room_dims_m: Optional[Tuple[float, float]] = None):
        """
        Add a scan from a specific camera position.

        Args:
            scan_data: BBOX data with 'yaw' and 'bboxes'
            camera_x_m: Camera X position in meters within the house
            camera_y_m: Camera Y position in meters within the house
            camera_height_m: Camera height from ground in meters
            camera_fov_h: Horizontal field of view in degrees
            camera_fov_v: Vertical field of view in degrees
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
            room_name: Optional name for the room being scanned
            room_dims_m: Optional (width, height) of room in meters
        """
        # Convert camera position to grid
        cam_grid_x, cam_grid_y = self.meters_to_grid(camera_x_m, camera_y_m)

        # Get scan parameters
        yaw = scan_data.get('yaw', 0)  # Camera angle in radians
        bboxes = scan_data.get('bboxes', [])

        # Calculate FOV in radians for this scan
        fov_h_rad = math.radians(camera_fov_h)

        # Process detections
        detected_objects = []
        detected_doors = []

        for bbox_item in bboxes:
            obj_class = bbox_item['class'].lower()
            bbox = bbox_item['bbox']
            confidence = bbox_item.get('confidence', 1.0)

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
            object_angle = yaw + angle_offset

            # Calculate object position in meters
            obj_x_m = camera_x_m + distance_m * math.sin(object_angle)
            obj_y_m = camera_y_m - distance_m * math.cos(object_angle)

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

            print(f"Scanned '{room_name}' from position ({camera_x_m:.1f}, {camera_y_m:.1f})m")
            print(f"  Found {len(detected_objects)} objects and {len(detected_doors)} doors")
        else:
            print(f"Scanned from position ({camera_x_m:.1f}, {camera_y_m:.1f})m")
            print(f"  Found {len(detected_objects)} objects and {len(detected_doors)} doors")

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
        print(f"Saved structure to {json_file}")

        # Save grid map
        grid = self.create_grid_map()
        np.savetxt(map_file, grid, fmt='%d')
        print(f"Saved grid map to {map_file} (shape: {grid.shape})")


def main():
    """Example usage with two rooms, each scanned from 4 angles."""

    # Create unifier for a 15x15 meter house
    unifier = RoomUnifier(
        house_width_m=15.0,
        house_height_m=15.0,
        grid_resolution=0.25  # 25cm per grid cell
    )

    # =================================================================
    # LIVING ROOM - Camera at position (4.0, 4.0)
    # =================================================================
    living_cam_x = 4.0
    living_cam_y = 4.0

    # Living Room - Scan 1: Facing North (0 radians)
    living_scan_north = {
        'yaw': 0,
        'bboxes': [
            # Couch against north wall (straight ahead, close)
            {'class': 'couch', 'bbox': [300, 350, 980, 650], 'confidence': 0.92},
            # TV on wall above couch (higher in frame)
            {'class': 'tv', 'bbox': [480, 200, 800, 350], 'confidence': 0.88},
            # Plant in left corner
            {'class': 'plant', 'bbox': [50, 400, 150, 580], 'confidence': 0.75},
            # Cabinet on right side
            {'class': 'cabinet', 'bbox': [1050, 380, 1200, 620], 'confidence': 0.83}
        ]
    }
    unifier.add_scan(living_scan_north, camera_x_m=living_cam_x, camera_y_m=living_cam_y,
                     camera_height_m=1.6, camera_fov_h=70, camera_fov_v=50,
                     frame_width=1280, frame_height=720, room_name="living_room")

    # Living Room - Scan 2: Facing East (π/2 radians)
    living_scan_east = {
        'yaw': 1.57,  # 90 degrees
        'bboxes': [
            # Table against east wall (straight ahead)
            {'class': 'table', 'bbox': [400, 420, 880, 600], 'confidence': 0.89},
            # Chairs around table
            {'class': 'chair', 'bbox': [320, 460, 420, 620], 'confidence': 0.81},
            {'class': 'chair', 'bbox': [860, 460, 960, 620], 'confidence': 0.79},
            # Window on wall (high up)
            {'class': 'window', 'bbox': [500, 150, 780, 320], 'confidence': 0.91},
            # Box in corner
            {'class': 'box', 'bbox': [100, 500, 200, 600], 'confidence': 0.72}
        ]
    }
    unifier.add_scan(living_scan_east, camera_x_m=living_cam_x, camera_y_m=living_cam_y,
                     camera_height_m=1.6, camera_fov_h=70, camera_fov_v=50,
                     frame_width=1280, frame_height=720, room_name="living_room")

    # Living Room - Scan 3: Facing South (π radians)
    living_scan_south = {
        'yaw': 3.14,  # 180 degrees
        'bboxes': [
            # Door to hallway (closed)
            {'class': 'door_closed', 'bbox': [550, 300, 730, 680], 'confidence': 0.94},
            # Another chair near door
            {'class': 'plastic_chair', 'bbox': [350, 450, 480, 630], 'confidence': 0.77},
            # Suitcase in corner
            {'class': 'suitcase', 'bbox': [900, 480, 1050, 620], 'confidence': 0.68},
            # Socket on wall
            {'class': 'socket', 'bbox': [200, 520, 230, 550], 'confidence': 0.65}
        ]
    }
    unifier.add_scan(living_scan_south, camera_x_m=living_cam_x, camera_y_m=living_cam_y,
                     camera_height_m=1.6, camera_fov_h=70, camera_fov_v=50,
                     frame_width=1280, frame_height=720, room_name="living_room")

    # Living Room - Scan 4: Facing West (-π/2 or 3π/2 radians)
    living_scan_west = {
        'yaw': 4.71,  # 270 degrees
        'bboxes': [
            # Another cabinet on west wall
            {'class': 'cabinet', 'bbox': [200, 350, 500, 650], 'confidence': 0.86},
            # Computer setup in corner
            {'class': 'computer', 'bbox': [650, 400, 750, 500], 'confidence': 0.82},
            {'class': 'keyboard', 'bbox': [680, 500, 780, 530], 'confidence': 0.78},
            {'class': 'mouse', 'bbox': [800, 505, 830, 525], 'confidence': 0.71},
            # Bottle on cabinet
            {'class': 'bottle', 'bbox': [350, 330, 380, 400], 'confidence': 0.64}
        ]
    }
    unifier.add_scan(living_scan_west, camera_x_m=living_cam_x, camera_y_m=living_cam_y,
                     camera_height_m=1.6, camera_fov_h=70, camera_fov_v=50,
                     frame_width=1280, frame_height=720, room_name="living_room")

    # =================================================================
    # BEDROOM - Camera at position (11.0, 11.0) - Far from living room
    # =================================================================
    bedroom_cam_x = 11.0
    bedroom_cam_y = 11.0

    # Bedroom - Scan 1: Facing North (0 radians)
    bedroom_scan_north = {
        'yaw': 0,
        'bboxes': [
            # Bed against north wall (large, prominent)
            {'class': 'bed', 'bbox': [200, 300, 1080, 700], 'confidence': 0.95},
            # Window above bed
            {'class': 'window', 'bbox': [450, 120, 830, 280], 'confidence': 0.89},
            # Plant on nightstand (small, to the side)
            {'class': 'plant', 'bbox': [100, 440, 180, 540], 'confidence': 0.73}
        ]
    }
    unifier.add_scan(bedroom_scan_north, camera_x_m=bedroom_cam_x, camera_y_m=bedroom_cam_y,
                     camera_height_m=1.5, camera_fov_h=65, camera_fov_v=45,
                     frame_width=1280, frame_height=720, room_name="bedroom")

    # Bedroom - Scan 2: Facing East (π/2 radians)
    bedroom_scan_east = {
        'yaw': 1.57,  # 90 degrees
        'bboxes': [
            # Desk against east wall
            {'class': 'desk', 'bbox': [300, 380, 900, 620], 'confidence': 0.91},
            # Chair at desk
            {'class': 'chair', 'bbox': [520, 480, 680, 650], 'confidence': 0.84},
            # Computer on desk
            {'class': 'computer', 'bbox': [450, 340, 550, 440], 'confidence': 0.87},
            {'class': 'keyboard', 'bbox': [480, 440, 620, 470], 'confidence': 0.82},
            {'class': 'mouse', 'bbox': [640, 445, 670, 465], 'confidence': 0.76},
            # Cardboard boxes stacked in corner
            {'class': 'cardboard_box', 'bbox': [50, 500, 200, 650], 'confidence': 0.69},
            {'class': 'cardboard_box', 'bbox': [80, 400, 220, 500], 'confidence': 0.71}
        ]
    }
    unifier.add_scan(bedroom_scan_east, camera_x_m=bedroom_cam_x, camera_y_m=bedroom_cam_y,
                     camera_height_m=1.5, camera_fov_h=65, camera_fov_v=45,
                     frame_width=1280, frame_height=720, room_name="bedroom")

    # Bedroom - Scan 3: Facing South (π radians)
    bedroom_scan_south = {
        'yaw': 3.14,  # 180 degrees
        'bboxes': [
            # Door to hallway (open)
            {'class': 'door_open', 'bbox': [480, 280, 600, 700], 'confidence': 0.93},
            # Cabinet/wardrobe next to door
            {'class': 'cabinet', 'bbox': [700, 200, 1100, 680], 'confidence': 0.90},
            # Suitcase near door
            {'class': 'suitcase', 'bbox': [300, 520, 450, 650], 'confidence': 0.74},
            # Socket on wall
            {'class': 'socket', 'bbox': [150, 510, 180, 540], 'confidence': 0.67}
        ]
    }
    unifier.add_scan(bedroom_scan_south, camera_x_m=bedroom_cam_x, camera_y_m=bedroom_cam_y,
                     camera_height_m=1.5, camera_fov_h=65, camera_fov_v=45,
                     frame_width=1280, frame_height=720, room_name="bedroom")

    # Bedroom - Scan 4: Facing West (-π/2 or 3π/2 radians)
    bedroom_scan_west = {
        'yaw': 4.71,  # 270 degrees
        'bboxes': [
            # TV on west wall
            {'class': 'tv', 'bbox': [400, 250, 880, 450], 'confidence': 0.88},
            # Small table under TV
            {'class': 'table', 'bbox': [450, 450, 830, 580], 'confidence': 0.81},
            # Plastic chair in corner
            {'class': 'plastic_chair', 'bbox': [100, 480, 250, 640], 'confidence': 0.75},
            # Bottles on table
            {'class': 'bottle', 'bbox': [500, 420, 530, 480], 'confidence': 0.66},
            {'class': 'bottle', 'bbox': [750, 420, 780, 480], 'confidence': 0.68}
        ]
    }
    unifier.add_scan(bedroom_scan_west, camera_x_m=bedroom_cam_x, camera_y_m=bedroom_cam_y,
                     camera_height_m=1.5, camera_fov_h=65, camera_fov_v=45,
                     frame_width=1280, frame_height=720, room_name="bedroom")

    # Save results
    unifier.save()

    # Print summary
    print(f"\nProcessed {len(unifier.rooms)} rooms")
    print(f"Total objects detected: {len(unifier.all_objects)}")
    print(f"\nRoom details:")
    for room_name, room_info in unifier.rooms.items():
        print(f"  {room_name}:")
        print(f"    Camera position: {room_info['camera_position_m']}m")
        print(f"    Objects: {len(room_info['objects'])}")
        print(f"    Doors: {len(room_info['doors'])}")
    print(f"\nGrid info:")
    print(f"  House: {unifier.house_width_m}x{unifier.house_height_m}m")
    print(f"  Grid: {unifier.grid_width}x{unifier.grid_height} cells")
    print(f"  Resolution: {unifier.grid_resolution}m per cell")


if __name__ == "__main__":
    main()