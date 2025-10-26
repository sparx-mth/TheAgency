#!/usr/bin/env python3
"""
path_planner.py
---------------
A* pathfinding between rooms using bounding boxes and occupancy map.

Walkable tiles:
  0 = free space
  3 = open door
  8 = entry
Everything else is blocked.

Planner constraints:
- Traverse only through Open Space, hallway, the TARGET ROOM, and unlabeled tiles (None),
  except: door/entry tiles (3/8) are always allowed as pass-throughs.
- Target the nearest valid DOOR/ENTRY cell of the goal room; if none are listed/valid,
  fall back to any walkable cell inside the room bbox.
"""

import json, os, time, heapq, numpy as np
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

UNIFIED_ROOMS   = DATA_DIR / "unified_rooms.json"
OBJECT_LOCATION = DATA_DIR / "object_location.json"
PLANNED_PATH    = DATA_DIR / "planned_path.json"
HOUSE_MAP       = DATA_DIR / "house_map.txt"

GRID_RES = 0.15  # meters per grid cell


class PathPlanner:
    def __init__(self):
        self.rooms = {}
        self.grid = None  # list[list[int]] or np.ndarray

    # === Core data ===
    def load_rooms(self):
        try:
            self.rooms = json.load(open(UNIFIED_ROOMS)).get("rooms", {})
            # Try loading occupancy map if available
            if HOUSE_MAP.exists():
                # force integer array to avoid mixed types
                self.grid = np.loadtxt(HOUSE_MAP, dtype=int)
            else:
                # fallback empty map
                size = 40
                self.grid = np.zeros((size, size), dtype=int)
            return True
        except Exception as e:
            print(f"[ERROR] Can't load rooms: {e}")
            return False

    def _in_bounds(self, x, y):
        if self.grid is None:
            return False
        h, w = self.grid.shape
        return 0 <= x < w and 0 <= y < h

    def get_room_at(self, x, y):
        """Return room name containing (x,y) by bbox, or None."""
        for name, info in self.rooms.items():
            bbox = info.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if x1 <= x < x2 and y1 <= y < y2:
                    return name
        return None

    def get_center(self, room):
        info = self.rooms.get(room, {})
        bbox = info.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return (x1 + x2) // 2, (y1 + y2) // 2
        return None

    # === A* ===
    def a_star(self, start, goal, allowed_rooms):
        """
        Run A* using self.grid; walk only on 0/3/8 and only in allowed_rooms (except 3/8 always allowed).
        allowed_rooms: set of room names allowed to traverse (e.g., {'Open Space','hallway', target_room, None})
        8-neighborhood moves with Euclidean heuristic -> shortest valid path in continuous sense.
        """
        grid = np.array(self.grid, dtype=int)
        max_y, max_x = grid.shape

        WALKABLE = {0, 3, 8}
        h = lambda a, b: ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

        # Min-heap of (f, g, node)
        open_q = [(h(start, goal), 0.0, start)]
        came = {}
        g = {start: 0.0}
        visited = set()

        while open_q:
            _, g_cur, cur = heapq.heappop(open_q)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return list(reversed(path))

            for dx, dy, step_cost in [
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414)
            ]:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < max_x and 0 <= ny < max_y):
                    continue

                val = grid[ny, nx]
                if val not in WALKABLE:
                    continue  # walls/objects blocked immediately

                room_here = self.get_room_at(nx, ny)
                # Constrain traversal to allowed rooms. Door/Entry (3/8) always allowed as connectors.
                if (room_here not in allowed_rooms) and (val not in (3, 8)):
                    continue

                n = (nx, ny)
                g2 = g_cur + step_cost
                if n not in g or g2 < g[n]:
                    g[n] = g2
                    came[n] = cur
                    f = g2 + h(n, goal)
                    heapq.heappush(open_q, (f, g2, n))

        return []

    # === Key point logic ===
    def identify_key_points(self, path):
        if not path:
            return []
        keys = []
        last_room = self.get_room_at(*path[0])
        keys.append({"point": list(path[0]), "room": last_room, "type": "start"})
        for i in range(1, len(path)):
            room = self.get_room_at(*path[i])
            if room != last_room:
                if last_room:
                    keys.append({"point": list(path[i-1]), "room": last_room, "type": "exit_room"})
                if room:
                    t = "enter_hallway" if ("hall" in room.lower()) else "enter_room"
                    keys.append({"point": list(path[i]), "room": room, "type": t})
                last_room = room
        x, y = path[-1]
        keys.append({"point": [x, y], "room": self.get_room_at(x, y), "type": "goal"})
        return keys

    def calc_distances(self, keys):
        segs = []
        total = 0.0
        for i in range(len(keys) - 1):
            p1, p2 = keys[i]["point"], keys[i+1]["point"]
            # Manhattan grid meters between waypoints is consistent with path step costs
            d = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])) * GRID_RES
            segs.append({
                "from": keys[i]["room"],
                "to": keys[i+1]["room"],
                "distance_m": round(d, 2)
            })
            total += d
        return segs, round(total, 2)

    # === Target collection ===
    def _collect_target_cells(self, room_name):
        """
        Collect valid door/entry cells for the target room.
        Accepts both [x,y] and [[x,y], ...] formats in rooms JSON.
        Verifies that each coordinate is a 3/8 tile on the grid.
        """
        grid = np.array(self.grid, dtype=int)
        H, W = grid.shape

        def add_xy_or_list(field_val, out):
            if isinstance(field_val, list) and len(field_val) == 2 and all(isinstance(v, int) for v in field_val):
                out.append(tuple(field_val))
            elif isinstance(field_val, list) and all(isinstance(p, list) and len(p) == 2 for p in field_val):
                for p in field_val:
                    out.append(tuple(p))

        targets = []
        info = self.rooms.get(room_name, {})
        if "doors" in info:
            add_xy_or_list(info["doors"], targets)
        if "entries" in info:
            add_xy_or_list(info["entries"], targets)

        valid = []
        for (x, y) in targets:
            if 0 <= x < W and 0 <= y < H:
                if grid[y, x] in (3, 8):
                    valid.append((x, y))
        return valid

    def plan(self, start, goal_room):
        """
        Plan a shortest valid path through free/door/entry, constrained to Open Space, hallway, and target room.
        Target the nearest valid door/entry cell of the target room. Fallback: any walkable cell inside room bbox.
        """
        if self.grid is None:
            print("[WARN] No grid loaded")
            return None

        # Allowed rooms to traverse (plus unlabeled/None)
        allowed_rooms = {"Open Space", "hallway", goal_room, None}

        grid = np.array(self.grid, dtype=int)

        # Collect valid door/entry targets for the goal room
        door_targets = self._collect_target_cells(goal_room)

        # Fallback: any walkable cell inside the room bbox
        if not door_targets:
            info = self.rooms.get(goal_room, {})
            bbox = info.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cand = []
                for yy in range(y1, y2):
                    for xx in range(x1, x2):
                        if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]:
                            if grid[yy, xx] in (0, 3, 8):
                                cand.append((xx, yy))
                door_targets = cand

        if not door_targets:
            print(f"[WARN] No valid targets found for room {goal_room}")
            return None

        # Find shortest path to any target door/entry inside constraints
        best_path, best_len = None, float("inf")
        for tgt in door_targets:
            path = self.a_star(start, tgt, allowed_rooms)
            if path and len(path) < best_len:
                best_path, best_len = path, len(path)

        if not best_path:
            print("[WARN] No valid constrained path found")
            return None

        keys = self.identify_key_points(best_path)
        segs, total = self.calc_distances(keys)
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoints": keys,
            "segments": segs,
            "total_distance_m": total
        }


def main():
    planner = PathPlanner()
    if not planner.load_rooms():
        print("No room file found.")
        return

    # You can change the live start position here if needed:
    start_pos = (27, 34)

    last_mod = 0
    print("Path planner running...")
    while True:
        try:
            if os.path.exists(OBJECT_LOCATION):
                t = os.path.getmtime(OBJECT_LOCATION)
                if t > last_mod:
                    data = json.load(open(OBJECT_LOCATION))
                    # Accept 'room' or 'found_room' fields
                    target = data.get("room") or data.get("found_room")
                    if target and target != "none":
                        print(f"\nPlanning path to {target}")
                        plan = planner.plan(start_pos, target)
                        if plan:
                            json.dump(plan, open(PLANNED_PATH, "w"), indent=2)
                            print(f"Path: {plan['total_distance_m']}m, {len(plan['waypoints'])} points")
                    last_mod = t
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()
