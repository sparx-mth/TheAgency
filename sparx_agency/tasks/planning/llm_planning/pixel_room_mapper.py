#!/usr/bin/env python3
"""
pixel_room_mapper.py — Room mapper with Depth Anything support.

Grid coords:  origin = top-left (NW),  +X = east,  +Y = south.

    ┌── North (y=0) ──┐
    │                  │
  West(x=0)        East(x=W)
    │                  │
    └── South (y=H) ──┘

CHANGED: Each object occupies exactly ONE cell at its center of gravity.
         No size-based footprint — minimal overlaps on the 2D map.
"""

import numpy as np, json, math, os, glob, time, hashlib
from typing import Dict, Tuple, Optional
from pathlib import Path
from house_config import get_config

cfg = get_config()

BASE_PATH = str(Path(__file__).resolve().parent.parent)


# ── Tile Manager ─────────────────────────────────────────────────────

class DynamicTileManager:
    """Tile registry with 29 perceptually-distinct colours + overlap blending."""

    PALETTE = [
        (240,240,240), (45,45,45),    (0,210,80),    (160,100,40),   # free/wall/camera/door
        (230,25,75),   (0,130,200),   (255,190,0),   (145,30,180),   # 4-7
        (245,130,48),  (70,240,240),  (240,50,230),  (210,245,60),   # 8-11
        (0,128,128),   (34,139,34),   (128,128,0),   (0,75,145),     # 12-15
        (128,0,0),     (255,215,180), (170,255,195), (230,190,255),  # 16-19
        (255,250,200), (60,180,75),   (220,190,255), (255,127,80),   # 20-23
        (0,200,160),   (188,143,143), (75,0,130),    (255,99,71),    # 24-27
        (0,191,255),   (154,205,50),  (255,20,147),  (0,255,127),    # 28-31
        (218,112,214), (127,255,0),   (255,160,122), (72,61,139),    # 32-35
        (32,178,170),  (255,69,0),    (148,103,189), (44,160,44),    # 36-39
        (214,39,40),   (255,187,120), (152,223,138), (174,199,232),  # 40-43
        (197,176,213), (196,156,148), (247,182,210), (199,199,199),  # 44-47
    ]

    FREE_SPACE, WALL, CAMERA, DOOR = 0, 1, 2, 3

    def __init__(self, existing_registry=None):
        self.overlap_parents = {}
        if existing_registry:
            self.tile_registry = existing_registry.copy()
            self.next_tile_id = max(existing_registry.values()) + 1
        else:
            self.tile_registry = {'free_space': 0, 'wall': 1, 'camera': 2, 'door': 3}
            self.next_tile_id = 4
        self.id_to_name = {v: k for k, v in self.tile_registry.items()}

    def get_color(self, tid: int) -> Tuple[int, int, int]:
        if tid < len(self.PALETTE):
            return self.PALETTE[tid]
        h = hashlib.md5(str(tid).encode()).digest()
        return (h[0], h[1], h[2])

    def get_all_colors(self) -> Dict:
        return {tid: self.get_color(tid) for tid in self.tile_registry.values()}

    def get_color_registry_hex(self) -> Dict[str, str]:
        return {n: "#{:02x}{:02x}{:02x}".format(*self.get_color(t))
                for n, t in self.tile_registry.items()}

    def get_tile_type(self, obj_class: str) -> int:
        key = obj_class.lower().strip()
        if key not in self.tile_registry:
            self.tile_registry[key] = self.next_tile_id
            self.id_to_name[self.next_tile_id] = key
            self.next_tile_id += 1
        return self.tile_registry[key]

    def get_overlap_tile_type(self, existing_id: int, new_class: str) -> int:
        new_name = new_class.lower().strip()
        if existing_id in (0, 1, 2, 3):
            return self.get_tile_type(new_class)
        existing_name = self.id_to_name.get(existing_id, "unknown")
        existing_parts = set(existing_name.split(" and "))
        if new_name in existing_parts:
            return existing_id
        all_parts = sorted(existing_parts | {new_name})
        combo_name = " and ".join(all_parts)
        new_id = self.get_tile_type(combo_name)
        if new_id not in self.overlap_parents:
            self.overlap_parents[new_id] = tuple(
                self.tile_registry.get(p, 0) for p in all_parts)
        return new_id

    def get_all_tiles(self) -> Dict:
        return self.tile_registry.copy()


# ── Room Geometry ────────────────────────────────────────────────────

def extract_room_geometry(scan_data: Dict) -> Optional[Dict]:
    """Extract room dimensions from depth data.

    wall_depth = max(max_depth_col_m)  — distance to the far (back) wall.
    wall_width = (img_w / fx) * min(max_depth_col_m) — the min depth columns
    correspond to side walls (the closest walls the camera can see); projecting
    the full image width at that depth gives the physical room width.
    """
    cam = scan_data.get("camera")
    objects = scan_data.get("objects", [])

    # Unwrap DEPTH_ANYTHING if present (same as add_scan)
    if not cam and "DEPTH_ANYTHING" in scan_data:
        da = scan_data["DEPTH_ANYTHING"]
        cam = da.get("camera")
        objects = da.get("objects", [])

    if not cam or not objects:
        return None

    wall_depths = [obj.get("max_depth_col_m", obj.get("depth_m", 2.0))
                   for obj in objects if obj.get("max_depth_col_m", 0) > 0.1]
    if not wall_depths:
        return None

    wall_depth = max(wall_depths)           # back wall (farthest point)
    side_depth = min(wall_depths)            # side walls (closest walls)
    fx = cam.get("fx", 500)
    img_w = cam.get("width", 640)
    wall_width = (img_w / fx) * side_depth
    camera_lateral = (cam.get("cx", img_w / 2) / fx) * side_depth

    return {"wall_depth_m": round(wall_depth, 4),
            "wall_width_m": round(wall_width, 4),
            "camera_lateral_m": round(camera_lateral, 4)}


def compute_room_config(wall_geometry: Optional[Dict],
                        camera_wall: str = "north",
                        camera_position_along_wall: Optional[float] = None) -> Dict:
    """Convert wall measurements → room dimensions + camera pose.
    Yaw: north=0, south=π, east=−π/2, west=+π/2."""
    camera_wall = camera_wall.lower().strip()
    fwd = wall_geometry["wall_depth_m"] if wall_geometry else 2.0
    lat = wall_geometry["wall_width_m"] if wall_geometry else 2.5

    if camera_wall in ("north", "south"):
        room_w, room_h = lat, fwd
    elif camera_wall in ("east", "west"):
        room_w, room_h = fwd, lat
    else:
        raise ValueError(f"camera_wall must be north/south/east/west, got '{camera_wall}'")

    if camera_position_along_wall is not None:
        pos = camera_position_along_wall
    elif camera_wall in ("north", "south"):
        pos = room_w / 2
    else:
        pos = room_h / 2

    configs = {
        "north": (pos, 0.0, 0.0),           "south": (pos, room_h, math.pi),
        "east":  (room_w, pos, -math.pi/2),  "west":  (0.0, pos, math.pi/2),
    }
    cam_x, cam_y, yaw = configs[camera_wall]
    result = {"room_width_m": round(room_w, 4), "room_height_m": round(room_h, 4),
              "camera_x_m": round(cam_x, 4), "camera_y_m": round(cam_y, 4), "yaw": yaw}
    print(f"  Camera wall: {camera_wall}  |  Room: {result['room_width_m']:.2f}x"
          f"{result['room_height_m']:.2f}m  |  Camera: ({cam_x:.2f},{cam_y:.2f}) "
          f"yaw={math.degrees(yaw):.0f}deg")
    return result


# ── Pixel Room Mapper ────────────────────────────────────────────────

class PixelRoomMapper:

    def __init__(self, mode="standalone", room_width_m=2.5, room_height_m=2.5,
                 grid_resolution=0.1, res_x=None, res_y=None,
                 existing_map_file=None, existing_json_file=None,
                 room_bbox=None, room_name="main_room",
                 camera_fov_h=100, camera_fov_v=50,
                 camera_x_m=None, camera_y_m=None):
        self.mode, self.room_name = mode, room_name
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        # Load existing data
        existing_registry = None
        self.existing_rooms = {}
        if existing_json_file and os.path.exists(existing_json_file):
            with open(existing_json_file) as f:
                d = json.load(f)
            existing_registry = d.get("tile_registry")
            self.existing_rooms = d.get("rooms", {})

        if mode == "standalone":
            self.room_width_m, self.room_height_m = room_width_m, room_height_m
            self.res_x = res_x if res_x is not None else grid_resolution
            self.res_y = res_y if res_y is not None else grid_resolution
            self.grid_resolution = grid_resolution
            self.grid_width = int(room_width_m / self.res_x + 0.5)
            self.grid_height = int(room_height_m / self.res_y + 0.5)
            self.camera_x_m = camera_x_m if camera_x_m is not None else room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else room_height_m / 2
            self.room_bbox = (0, 0, self.grid_width, self.grid_height)
            self.map_width, self.map_height = self.grid_width, self.grid_height
        elif mode == "existing_map":
            if not existing_map_file or not room_bbox:
                raise ValueError("existing_map mode requires map file and room bbox")
            self.existing_grid = np.loadtxt(existing_map_file, dtype=np.int8)
            self.room_bbox = room_bbox
            x1, y1, x2, y2 = room_bbox
            self.room_width_m, self.room_height_m = room_width_m, room_height_m
            self.res_x = room_width_m / (x2 - x1)
            self.res_y = room_height_m / (y2 - y1)
            self.grid_resolution = (self.res_x + self.res_y) / 2
            self.grid_width, self.grid_height = x2 - x1, y2 - y1
            self.camera_x_m = camera_x_m if camera_x_m is not None else room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else room_height_m / 2
            self.map_height, self.map_width = self.existing_grid.shape

        print(f"Room: {self.room_width_m:.2f}x{self.room_height_m:.2f}m  "
              f"({self.grid_width}x{self.grid_height} cells)  "
              f"Camera: ({self.camera_x_m:.2f},{self.camera_y_m:.2f})m")

        self.tiles = DynamicTileManager(existing_registry)
        self.all_objects = []
        self.occupied_cells = {}  # (gx, gy) -> index in all_objects

    # ── Helpers ──

    def _clamp(self, x, y):
        return (max(0.05, min(self.room_width_m - 0.05, x)),
                max(0.05, min(self.room_height_m - 0.05, y)))

    def meters_to_grid(self, x_m, y_m):
        return (max(0, min(self.grid_width - 1, int(x_m / self.res_x))),
                max(0, min(self.grid_height - 1, int(y_m / self.res_y))))

    def camera_to_grid(self):
        def snap(val, limit, res, grid_size):
            if val <= 0: return 0
            if val >= limit: return grid_size - 1
            return max(0, min(grid_size - 1, int(val / res)))
        return snap(self.camera_x_m, self.room_width_m, self.res_x, self.grid_width), \
               snap(self.camera_y_m, self.room_height_m, self.res_y, self.grid_height)

    # ── Size & Position ──

    def estimate_object_size(self, bbox, fw, fh, depth_m, intrinsics=None):
        pw, ph = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if intrinsics:
            h_m = (pw / intrinsics["fx"]) * depth_m
            v_m = (ph / intrinsics["fy"]) * depth_m
        else:
            h_m = (pw / fw) * 2 * depth_m * math.tan(self.camera_fov_h / 2)
            v_m = (ph / fh) * 2 * depth_m * math.tan(self.camera_fov_v / 2)
        return (max(0.1, min(h_m, self.room_width_m / 3)),
                max(0.1, min(v_m, self.room_height_m / 3)))

    def calculate_position(self, bbox, yaw, fw, depth_m, intrinsics=None, max_depth_col_m=None):
        """Compute world position from depth geometry using camera intrinsics.

        Uses depth_m as the true forward Z coordinate, then derives the
        lateral X from the pinhole model:  X = (u - cx) / fx * Z
        The object ray angle is:           θ = atan2(X, Z)
        This produces a physically correct ray direction for world placement.
        """
        cx_px = (bbox[0] + bbox[2]) / 2       # bbox centre in pixels

        Z = depth_m                            # true forward depth

        if intrinsics:
            # --- Ray direction from pinhole geometry ---
            X_cam = ((cx_px - intrinsics["cx"]) / intrinsics["fx"]) * Z   # lateral in metres
            theta = math.atan2(X_cam, Z)       # signed angle from optical axis

            # World-frame displacement: rotate ray by camera yaw
            #   camera convention:  yaw=0 → looking along +Y (south in grid)
            #   sin(yaw) gives the X component, cos(yaw) gives the Y component
            #   Negate theta: in image coords +X_cam = right, but when facing
            #   south (+Y) camera-right is west (−X), so flip the sign.
            world_angle = yaw - theta
            dist = math.sqrt(X_cam * X_cam + Z * Z)   # true Euclidean distance along ray

            ox = self.camera_x_m + dist * math.sin(world_angle)
            oy = self.camera_y_m + dist * math.cos(world_angle)
        else:
            # Fallback: FOV-based angle (no intrinsics available)
            ang = yaw - ((cx_px / fw) - 0.5) * self.camera_fov_h
            ox = self.camera_x_m + depth_m * math.cos(ang)
            oy = self.camera_y_m - depth_m * math.sin(ang)

        return self._clamp(ox, oy)

    # ── Duplicate detection ──

    def _is_duplicate(self, obj_class, gx, gy):
        """Check if an object of the same class already occupies this cell."""
        for ex in self.all_objects:
            if ex["type"] == obj_class:
                ex_gx, ex_gy = ex["grid_cell"]
                if ex_gx == gx and ex_gy == gy:
                    return True
        return False

    # ── Collision resolution ──

    def _cell_is_free(self, gx, gy):
        """Check if a grid cell is available for placement."""
        if gx <= 0 or gy <= 0 or gx >= self.map_width - 1 or gy >= self.map_height - 1:
            return False  # wall / out of bounds
        return (gx, gy) not in self.occupied_cells

    def _resolve_collision(self, gx, gy, new_pos_m):
        """Nudge a new object away from the existing occupant based on their
        world-position difference.

        Returns (final_gx, final_gy) or None if every nearby cell is taken.

        Logic:
          1. Compare world positions (metres) of new vs existing object.
          2. Primary axis = the larger absolute difference (lateral vs depth).
          3. Build an ordered list of candidate offsets: preferred direction on
             primary axis first, then secondary axis, then opposites, then
             diagonals.
          4. Return the first candidate that is free.
          5. If all 12 neighbours (ring-1 + ring-2 cardinal) are taken, spiral
             outward up to radius 3.
        """
        existing_idx = self.occupied_cells[(gx, gy)]
        existing_obj = self.all_objects[existing_idx]
        ex_pos = existing_obj["position_m"]

        dx_m = new_pos_m[0] - ex_pos[0]   # positive = new is to the right
        dy_m = new_pos_m[1] - ex_pos[1]   # positive = new is further south

        # Determine preferred direction per axis
        sx = 1 if dx_m >= 0 else -1   # horizontal nudge sign
        sy = 1 if dy_m >= 0 else -1   # vertical nudge sign

        # Primary axis = larger absolute difference
        if abs(dx_m) >= abs(dy_m):
            # lateral difference dominates → try X first
            candidates = [
                (sx, 0),       # primary preferred
                (0, sy),       # secondary preferred
                (-sx, 0),      # primary opposite
                (0, -sy),      # secondary opposite
                (sx, sy),      # diagonal preferred
                (sx, -sy),
                (-sx, sy),
                (-sx, -sy),    # diagonal opposite
            ]
        else:
            # depth difference dominates → try Y first
            candidates = [
                (0, sy),       # primary preferred
                (sx, 0),       # secondary preferred
                (0, -sy),      # primary opposite
                (-sx, 0),      # secondary opposite
                (sx, sy),      # diagonal preferred
                (sx, -sy),
                (-sx, sy),
                (-sx, -sy),
            ]

        # De-duplicate candidate list (preserve order)
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        # Try ring-1
        for cdx, cdy in unique:
            nx, ny = gx + cdx, gy + cdy
            if self._cell_is_free(nx, ny):
                return (nx, ny)

        # Try ring-2 and ring-3 (spiral outward)
        for radius in range(2, 4):
            for cdx in range(-radius, radius + 1):
                for cdy in range(-radius, radius + 1):
                    if abs(cdx) < radius and abs(cdy) < radius:
                        continue  # skip inner cells already tried
                    nx, ny = gx + cdx, gy + cdy
                    if self._cell_is_free(nx, ny):
                        return (nx, ny)

        return None  # extremely congested — give up

    # ── Scan Ingestion ──

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        if "DEPTH_ANYTHING" in scan_data:
            da = scan_data["DEPTH_ANYTHING"]
            scan_data["camera"] = da.get("camera", {})
            scan_data["objects"] = da.get("objects", [])

        cam = scan_data.get("camera", {})
        intrinsics = {"fx": cam.get("fx", 500), "fy": cam.get("fy", 500),
                      "cx": cam.get("cx", cam.get("width", 640) / 2),
                      "cy": cam.get("cy", cam.get("height", 480) / 2)}
        fw = cam.get("width", 640)
        fh = cam.get("height", 480)
        detections = scan_data.get("objects", [])

        for det in detections:
            obj_class = det.get('label', '').lower().replace('a ', '').replace('an ', '').strip()
            if not obj_class: continue

            bbox = det['bbox']
            depth_m = det.get("depth_m", cfg.default_distance_m)
            max_depth = det.get("max_depth_col_m")
            tile_type = self.tiles.get_tile_type(obj_class)
            ox, oy = self.calculate_position(bbox, yaw, fw, depth_m, intrinsics, max_depth)

            # Still estimate size for metadata, but do NOT use it for grid footprint
            wm, hm = self.estimate_object_size(bbox, fw, fh, depth_m, intrinsics)

            # ── SINGLE CELL at center of gravity ──
            gx, gy = self.meters_to_grid(ox, oy)

            # Offset for existing_map mode
            if self.mode == "existing_map":
                gx += self.room_bbox[0]
                gy += self.room_bbox[1]

            # Duplicate check: same class on same cell
            if self._is_duplicate(obj_class, gx, gy):
                print(f"  Skip duplicate: {obj_class} at ({gx},{gy})")
                continue

            # ── Collision resolution: nudge if cell is occupied ──
            if (gx, gy) in self.occupied_cells:
                resolved = self._resolve_collision(gx, gy, [ox, oy])
                if resolved is None:
                    print(f"  Skip congested: {obj_class} at ({gx},{gy}) — no free neighbour")
                    continue
                old_gx, old_gy = gx, gy
                gx, gy = resolved
                print(f"  Nudged: {obj_class} ({old_gx},{old_gy})->({gx},{gy})")

            # Register the cell
            obj_idx = len(self.all_objects)
            self.occupied_cells[(gx, gy)] = obj_idx

            # Single-cell bbox: [x, y, x+1, y+1]
            self.all_objects.append({
                "type": obj_class, "tile_type": tile_type,
                "bbox": [gx, gy, gx + 1, gy + 1],
                "grid_cell": (gx, gy),
                "depth_m": depth_m,
                "position_m": [round(ox, 3), round(oy, 3)],
                "size_m": [round(wm, 3), round(hm, 3)]})
            print(f"  Added: {obj_class} ({ox:.2f},{oy:.2f})m -> cell ({gx},{gy}) "
                  f"depth={depth_m:.2f}m")

    # ── Grid Creation ──

    def create_grid_map(self) -> np.ndarray:
        T = self.tiles
        if self.mode == "standalone":
            grid = np.full((self.grid_height, self.grid_width), T.FREE_SPACE, dtype=np.int8)
            grid[0, :] = grid[-1, :] = T.WALL
            grid[:, 0] = grid[:, -1] = T.WALL
        else:
            grid = self.existing_grid.copy()
            bx1, by1, bx2, by2 = self.room_bbox
            for y in range(by1 + 1, by2 - 1):
                for x in range(bx1 + 1, bx2 - 1):
                    if y < self.map_height and x < self.map_width:
                        grid[y, x] = T.FREE_SPACE

        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]; cy += self.room_bbox[1]
        if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
            grid[cy, cx] = T.CAMERA

        # Place each object as a single cell — no overlaps
        for obj in self.all_objects:
            gx, gy = obj["grid_cell"]
            if 0 < gx < self.map_width - 1 and 0 < gy < self.map_height - 1:
                t = grid[gy, gx]
                if t in (T.WALL, T.CAMERA, T.DOOR):
                    continue
                grid[gy, gx] = T.get_tile_type(obj["type"])
        return grid

    # ── Save ──

    def save(self, json_file=None, map_file=None):
        json_file = json_file or cfg.unified_rooms_json
        map_file = map_file or cfg.house_map_txt
        grid = self.create_grid_map()
        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]; cy += self.room_bbox[1]

        rooms = self.existing_rooms.copy()
        rooms[self.room_name] = {
            "name": self.room_name,
            "camera_position": [cx, cy],
            "camera_position_m": [round(self.camera_x_m, 3), round(self.camera_y_m, 3)],
            "room_dimensions_m": [round(self.room_width_m, 3), round(self.room_height_m, 3)],
            "bbox": list(self.room_bbox), "objects": self.all_objects, "doors": [25, 7]}

        output = {
            "house_dimensions_m": {"width": self.map_width * self.grid_resolution,
                                   "height": self.map_height * self.grid_resolution},
            "grid_resolution": self.grid_resolution, "rooms": rooms,
            "tile_registry": self.tiles.get_all_tiles(),
            "tile_colors": self.tiles.get_color_registry_hex()}

        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        np.savetxt(map_file, grid, fmt='%d')
        print(f"\nSaved {len(self.all_objects)} objects -> '{self.room_name}'  "
              f"({len(rooms)} rooms, {len(self.tiles.tile_registry)} tile types)")



# ── File Processing ──────────────────────────────────────────────────

def process_files(mode="standalone", existing_map=None, existing_json=None,
                  room_bbox=None, room_name="main_room", camera_wall="north",
                  camera_position_along_wall=None, grid_resolution=None,
                  grid_cells=51,
                  room_width_m=None, room_height_m=None):
    bbox_dir = os.path.join(BASE_PATH, "room_mapping", cfg.ingest_out_dir)
    json_files = glob.glob(os.path.join(bbox_dir, "*.json"))
    if not json_files:
        print(f"No JSON files in {bbox_dir}"); return 0
    print(f"Found {len(json_files)} JSON files")

    # For existing_map mode, derive resolution from wall geometry + bbox cell count
    if mode == "existing_map" and room_bbox:
        x1, y1, x2, y2 = room_bbox
        cells_w, cells_h = x2 - x1, y2 - y1

        # Pre-scan for wall geometry
        wall_geometry = None
        for jf in sorted(json_files):
            try:
                with open(jf) as f: data = json.load(f)
                geom = extract_room_geometry(data)
                if geom:
                    wall_geometry = geom
                    print(f"  Wall data: depth={geom['wall_depth_m']:.2f}m "
                          f"width={geom['wall_width_m']:.2f}m"); break
            except Exception: continue

        # Get physical room dimensions from wall geometry
        config = compute_room_config(wall_geometry, camera_wall, camera_position_along_wall)
        if room_width_m is not None:  config["room_width_m"] = room_width_m
        if room_height_m is not None: config["room_height_m"] = room_height_m

        # Resolution = physical dimension / cell count
        res_x = config["room_width_m"] / cells_w
        res_y = config["room_height_m"] / cells_h
        grid_resolution = (res_x + res_y) / 2

        # Recompute camera for final dimensions
        config = compute_room_config(
            {"wall_depth_m": config["room_height_m"] if camera_wall in ("north","south")
                             else config["room_width_m"],
             "wall_width_m": config["room_width_m"] if camera_wall in ("north","south")
                             else config["room_height_m"],
             "camera_lateral_m": 0},
            camera_wall, camera_position_along_wall)

        print(f"  Existing map: {cells_w}x{cells_h} cells, "
              f"room={config['room_width_m']:.3f}x{config['room_height_m']:.3f}m, "
              f"res=({res_x:.4f}, {res_y:.4f})")

        # Create mapper and process
        mapper = PixelRoomMapper(
            mode=mode, room_width_m=config["room_width_m"],
            room_height_m=config["room_height_m"], grid_resolution=grid_resolution,
            res_x=res_x, res_y=res_y,
            existing_map_file=existing_map, existing_json_file=existing_json,
            room_bbox=room_bbox, room_name=room_name,
            camera_fov_h=cfg.camera_fov_h, camera_fov_v=60,
            camera_x_m=config["camera_x_m"], camera_y_m=config["camera_y_m"])

        config_yaw = config["yaw"]
        for jf in sorted(json_files):
            try:
                print(f"Processing: {os.path.basename(jf)}")
                with open(jf) as f: sd = json.load(f)
                pose = sd.get('pose', {})
                yaw = pose['yaw'] if 'yaw' in pose else config_yaw
                mapper.add_scan(sd, yaw)
            except Exception as e:
                print(f"Error: {jf}: {e}"); import traceback; traceback.print_exc()

        mapper.save()
        return len(json_files)

    # 1. Pre-scan for wall data
    wall_geometry = None
    for jf in sorted(json_files):
        try:
            with open(jf) as f: data = json.load(f)
            geom = extract_room_geometry(data)
            if geom:
                wall_geometry = geom
                print(f"  Wall data: depth={geom['wall_depth_m']:.2f}m "
                      f"width={geom['wall_width_m']:.2f}m"); break
        except Exception: continue

    # 2. Compute initial room config from wall geometry
    config = compute_room_config(wall_geometry, camera_wall, camera_position_along_wall)
    if room_width_m is not None:  config["room_width_m"] = room_width_m
    if room_height_m is not None: config["room_height_m"] = room_height_m

    # 3. Derive per-axis resolution from grid_cells
    if grid_resolution is None:
        if isinstance(grid_cells, (list, tuple)):
            lat_cells, depth_cells = grid_cells
        else:
            lat_cells = depth_cells = grid_cells

        if camera_wall in ("north", "south"):
            cw, ch = lat_cells, depth_cells
        else:
            cw, ch = depth_cells, lat_cells

        wall_w = config["room_width_m"]
        wall_h = config["room_height_m"]
        res_x = wall_w / max(1, cw - 2)
        res_y = wall_h / max(1, ch - 2)
        config["room_width_m"]  = res_x * cw
        config["room_height_m"] = res_y * ch
        grid_resolution = (res_x + res_y) / 2
        print(f"  Grid: {cw}x{ch} cells (along={lat_cells}, depth={depth_cells}), "
              f"res=({res_x:.4f}, {res_y:.4f}) m/cell, "
              f"room={config['room_width_m']:.3f}x{config['room_height_m']:.3f}m")
    else:
        res_x = res_y = grid_resolution

    # Recompute camera for final room dimensions
    config = compute_room_config(
        {"wall_depth_m": config["room_height_m"] if camera_wall in ("north","south")
                         else config["room_width_m"],
         "wall_width_m": config["room_width_m"] if camera_wall in ("north","south")
                         else config["room_height_m"],
         "camera_lateral_m": 0},
        camera_wall, camera_position_along_wall)

    # 4. Create mapper
    mapper = PixelRoomMapper(
        mode=mode, room_width_m=config["room_width_m"],
        room_height_m=config["room_height_m"], grid_resolution=grid_resolution,
        res_x=res_x, res_y=res_y,
        existing_map_file=existing_map, existing_json_file=existing_json,
        room_bbox=room_bbox, room_name=room_name,
        camera_fov_h=cfg.camera_fov_h,
        camera_fov_v=cfg.camera_fov_v if mode == "standalone" else 60,
        camera_x_m=config["camera_x_m"], camera_y_m=config["camera_y_m"])

    # 5. Process each file
    config_yaw = config["yaw"]
    for jf in sorted(json_files):
        try:
            print(f"Processing: {os.path.basename(jf)}")
            with open(jf) as f: sd = json.load(f)
            pose = sd.get('pose', {})
            yaw = pose['yaw'] if 'yaw' in pose else config_yaw
            mapper.add_scan(sd, yaw)
        except Exception as e:
            print(f"Error: {jf}: {e}"); import traceback; traceback.print_exc()

    mapper.save()
    return len(json_files)


def write_empty_room(grid_cells=51, camera_wall="west"):
    """Write empty room data files (no objects) so renderer/web show a clean slate."""
    tiles = DynamicTileManager()
    if isinstance(grid_cells, (list, tuple)):
        cw, ch = grid_cells
    else:
        cw = ch = grid_cells

    grid = np.full((ch, cw), tiles.FREE_SPACE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = tiles.WALL
    grid[:, 0] = grid[:, -1] = tiles.WALL

    res = 0.1
    output = {
        "house_dimensions_m": {"width": cw * res, "height": ch * res},
        "grid_resolution": res,
        "rooms": {"main_room": {
            "name": "main_room", "objects": [], "doors": [25, 7],
            "camera_position": [cw // 2, ch // 2],
            "camera_position_m": [cw * res / 2, ch * res / 2],
            "room_dimensions_m": [cw * res, ch * res],
            "bbox": [0, 0, cw, ch]}},
        "tile_registry": tiles.get_all_tiles(),
        "tile_colors": tiles.get_color_registry_hex()}

    os.makedirs(cfg.data_dir, exist_ok=True)
    with open(cfg.unified_rooms_json, 'w') as f:
        json.dump(output, f, indent=2)
    np.savetxt(cfg.house_map_txt, grid, fmt='%d')
    print(f"[{time.strftime('%H:%M:%S')}] Wrote empty room ({cw}x{ch})")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # All configuration now comes from config.json + CLI overrides
    mode          = cfg.mode
    existing_map  = cfg.existing_map
    existing_json = cfg.existing_json
    room_bbox     = cfg.room_bbox
    room_name     = cfg.room_name

    camera_wall   = cfg.camera_wall
    camera_position_along_wall = cfg.camera_position_along_wall

    grid_cells     = cfg.grid_cells
    grid_resolution = cfg.grid_resolution
    room_width_m   = cfg.room_width_m
    room_height_m  = cfg.room_height_m

    if room_bbox and existing_map:
        mode = "existing_map"
        print(f"Mode: Existing Map | Room: {room_name} | bbox: {room_bbox}")
    else:
        gc = f"along={grid_cells[0]} depth={grid_cells[1]}" if isinstance(grid_cells, (list, tuple)) \
             else f"{grid_cells}x{grid_cells}"
        print(f"Mode: Standalone | "
              f"Room: {room_name} | Grid: {gc}")

    print(f"Camera: {camera_wall} wall, pos={camera_position_along_wall or 'middle'}")
    print("Monitoring for detection files...  (Ctrl+C to stop)\n")

    bbox_dir = os.path.join(BASE_PATH, "room_mapping", cfg.ingest_out_dir)
    last_count = -1
    last_process_time = 0
    FORCE_REFRESH_INTERVAL = 20  # seconds
    try:
        while True:
            files = glob.glob(os.path.join(bbox_dir, "*.json"))
            force_refresh = (time.time() - last_process_time) >= FORCE_REFRESH_INTERVAL
            if len(files) != last_count or (force_refresh and files):
                if files:
                    print(f"\n[{time.strftime('%H:%M:%S')}] {len(files)} file(s)")
                    n = process_files(
                        mode, existing_map, existing_json, room_bbox, room_name,
                        camera_wall=camera_wall,
                        camera_position_along_wall=camera_position_along_wall,
                        grid_resolution=grid_resolution, grid_cells=grid_cells,
                        room_width_m=room_width_m, room_height_m=room_height_m)
                    if n: print(f"Processed {n} files -> {cfg.unified_rooms_json} + {cfg.house_map_txt}")
                    last_process_time = time.time()
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No detection files")
                    if mode != "existing_map":
                        write_empty_room(grid_cells=grid_cells, camera_wall=camera_wall)
                last_count = len(files)
            time.sleep(cfg.mapper_poll_interval)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()