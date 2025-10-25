#!/usr/bin/env python3
"""
path_planner.py
---------------
A* pathfinding between rooms using bounding boxes and occupancy map.
Allowed cells: 0 (free) and 3 (open door).
All other values are obstacles.
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
        self.grid = None

    # === Core data ===
    def load_rooms(self):
        try:
            self.rooms = json.load(open(UNIFIED_ROOMS)).get("rooms", {})
            # Try loading occupancy map if available
            if HOUSE_MAP.exists():
                self.grid = np.loadtxt(HOUSE_MAP, dtype=int).tolist()
            else:
                # fallback empty map
                size = 40
                self.grid = [[0]*size for _ in range(size)]
            return True
        except Exception as e:
            print(f"[ERROR] Can't load rooms: {e}")
            return False

    def get_room_at(self, x, y):
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
            return (x1 + x2)//2, (y1 + y2)//2
        return None

    # === A* ===
    def a_star(self, start, goal):
        """Run A* using self.grid; can step only on 0 or 3."""
        h = lambda a,b: abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_q=[(0,start)]; came={}; g={start:0}; f={start:h(start,goal)}
        max_y, max_x = len(self.grid), len(self.grid[0])
        while open_q:
            _,cur=heapq.heappop(open_q)
            if cur==goal:
                path=[cur]
                while cur in came:
                    cur=came[cur]; path.append(cur)
                return list(reversed(path))
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx,ny=cur[0]+dx,cur[1]+dy
                if 0<=ny<max_y and 0<=nx<max_x:
                    val = self.grid[ny][nx]
                    if val not in (0,3):  # blocked
                        continue
                    n=(nx,ny)
                    g2=g[cur]+1
                    if n not in g or g2<g[n]:
                        g[n]=g2; f[n]=g2+h(n,goal)
                        came[n]=cur
                        heapq.heappush(open_q,(f[n],n))
        return []

    # === Key point logic ===
    def identify_key_points(self, path):
        if not path: return []
        keys=[]
        last_room=self.get_room_at(*path[0])
        keys.append({"point":list(path[0]),"room":last_room,"type":"start"})
        for i in range(1,len(path)):
            room=self.get_room_at(*path[i])
            if room!=last_room:
                if last_room:
                    keys.append({"point":list(path[i-1]),"room":last_room,"type":"exit_room"})
                if room:
                    t="enter_hallway" if "hall" in room.lower() else "enter_room"
                    keys.append({"point":list(path[i]),"room":room,"type":t})
                last_room=room
        x,y=path[-1]
        keys.append({"point":[x,y],"room":self.get_room_at(x,y),"type":"goal"})
        return keys

    def calc_distances(self, keys):
        segs=[]; total=0.0
        for i in range(len(keys)-1):
            p1,p2=keys[i]["point"],keys[i+1]["point"]
            d=(abs(p2[0]-p1[0])+abs(p2[1]-p1[1]))*GRID_RES
            segs.append({"from":keys[i]["room"],"to":keys[i+1]["room"],"distance_m":round(d,2)})
            total+=d
        return segs,round(total,2)

    def plan(self, start, goal_room):
        goal = self.get_center(goal_room)
        if not goal:
            print(f"[WARN] Room {goal_room} not found")
            return None
        path = self.a_star(start, goal)
        if not path:
            print("[WARN] No path found")
            return None
        keys = self.identify_key_points(path)
        segs, total = self.calc_distances(keys)
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoints": keys,
            "segments": segs,
            "total_distance_m": total
        }


def main():
    planner=PathPlanner()
    if not planner.load_rooms():
        print("No room file found."); return
    start_pos=(27,34)
    last_mod=0
    print("Path planner running...")
    while True:
        try:
            if os.path.exists(OBJECT_LOCATION):
                t=os.path.getmtime(OBJECT_LOCATION)
                if t>last_mod:
                    data=json.load(open(OBJECT_LOCATION))
                    target=data.get("room") or data.get("found_room")
                    if target and target!="none":
                        print(f"\nPlanning path to {target}")
                        plan=planner.plan(start_pos,target)
                        if plan:
                            json.dump(plan,open(PLANNED_PATH,"w"),indent=2)
                            print(f"Path: {plan['total_distance_m']}m, {len(plan['waypoints'])} points")
                    last_mod=t
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped."); break
        except Exception as e:
            print("Error:",e); time.sleep(1)


if __name__=="__main__":
    main()
