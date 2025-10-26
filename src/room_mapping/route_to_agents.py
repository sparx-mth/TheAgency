#!/usr/bin/env python3
# route_to_agents.py — watch planned_path.json and write agent plan as JSON + TEXT (no prints)

import json, sys, time, os

PATH_FILE  = sys.argv[1] if len(sys.argv) > 1 else "data/planned_path.json"
HOUSE_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/unified_rooms.json"
TARGET     = sys.argv[3] if len(sys.argv) > 3 else None
OUT_JSON   = "data/agent_commands.json"
OUT_TXT    = "data/agent_commands.txt"              # legacy text file for existing web/HTML
INCLUDE_WALL_STEP = False

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def has_doors(rooms, room_name):
    r = rooms.get(room_name, {})
    d = r.get("doors", [])
    return isinstance(d, list) and len(d) > 0

def step_nav(n, start_room, start_point, dest_room, dest_point, entrance):
    return {
        "n": n,
        "agent": "NavigationAgent",
        "from": {"room": start_room, "point": start_point},
        "to": {"room": dest_room, "point": dest_point, "entrance": entrance}
    }

def step_door(n, room):
    return {"n": n, "agent": "DoorAgent", "action": "enter", "room": room}

def step_wall(n, room):
    return {"n": n, "agent": "WallAgent", "action": "follow_walls", "room": room}

def step_scan(n, room, target=None):
    s = {"n": n, "agent": "ScanAgent", "action": "scan_room", "room": room}
    if target: s["objective"] = target
    return s

def build_agent_plan(path_dict, house_dict, target):
    waypoints = path_dict.get("waypoints", [])
    rooms = house_dict.get("rooms", {})
    steps = []
    n = 1

    if len(waypoints) < 2:
        cur_room = waypoints[0]["room"] if waypoints else "unknown"
        if INCLUDE_WALL_STEP:
            steps.append(step_wall(n, cur_room)); n += 1
        steps.append(step_scan(n, cur_room, target)); n += 1
    else:
        cur_room = waypoints[0]["room"]
        seg_start = waypoints[0]["point"]
        for i in range(1, len(waypoints)):
            wp = waypoints[i]
            if wp["room"] != cur_room:
                t = (wp.get("type") or "").lower()
                entrance = bool(t.startswith("enter_"))
                steps.append(step_nav(n, cur_room, seg_start, wp["room"], wp["point"], entrance)); n += 1
                if has_doors(rooms, wp["room"]):
                    steps.append(step_door(n, wp["room"])); n += 1
                cur_room = wp["room"]
                seg_start = wp["point"]

        final_room = waypoints[-1]["room"]
        if INCLUDE_WALL_STEP:
            steps.append(step_wall(n, final_room)); n += 1
        steps.append(step_scan(n, final_room, target)); n += 1

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "steps": steps
    }

def plan_to_text(plan):
    """Render the JSON plan into legacy numbered text lines with a timestamp header."""
    # lines = [f"# timestamp: {plan.get('timestamp','')}"]
    lines = []
    for i, s in enumerate(plan.get("steps", []), 1):
        a = s.get("agent")
        if a == "NavigationAgent":
            fr = s.get("from", {})
            to = s.get("to", {})
            from_pt = tuple(fr.get("point", [])) if fr.get("point") else ""
            to_pt   = tuple(to.get("point", [])) if to.get("point") else ""
            dest    = f"entrance of {to.get('room')}" if to.get("entrance") else to.get("room")
            lines.append(f"{i}. Activate NavigationAgent from {from_pt} in {fr.get('room')} to {dest} at {to_pt}")
        elif a == "DoorAgent":
            lines.append(f"{i}. Activate DoorAgent to open and enter {s.get('room')}")
        elif a == "WallAgent":
            lines.append(f"{i}. Activate WallAgent to follow walls inside {s.get('room')}")
        elif a == "ScanAgent":
            obj = s.get("objective")
            tail = f" to find the {obj}" if obj else ""
            lines.append(f"{i}. Activate ScanAgent to scan {s.get('room')}{tail}")
        else:
            # Fallback
            lines.append(f"{i}. {a or 'Step'}")
    return "\n".join(lines) + "\n"

def safe_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def safe_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def main():
    last_mtime = 0.0
    house = None
    house_mtime = 0.0

    while True:
        try:
            # Reload house data if it changes
            if os.path.exists(HOUSE_FILE):
                hm = os.path.getmtime(HOUSE_FILE)
                if hm > house_mtime:
                    house = load_json(HOUSE_FILE)
                    house_mtime = hm

            # React to planned_path changes
            if os.path.exists(PATH_FILE) and house is not None:
                m = os.path.getmtime(PATH_FILE)
                if m > last_mtime:
                    time.sleep(0.1)  # let writer finish
                    path_dict = load_json(PATH_FILE)
                    plan = build_agent_plan(path_dict, house, TARGET)
                    # Write JSON (for tooling) and TEXT (for existing web/HTML)
                    safe_write_json(OUT_JSON, plan)
                    safe_write_text(OUT_TXT, plan_to_text(plan))
                    last_mtime = m

            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except:
            time.sleep(0.5)

if __name__ == "__main__":
    main()
