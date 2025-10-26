#!/usr/bin/env python3
import json
import time
import os

PLANNED_PATH_FILE = "data/planned_path.json"
ROUTE_NARRATION_FILE = "data/route_narration.json"
DEFAULT_GRID_RES_M = 0.15

# --- Helpers -----------------------------------------------------------------

def round_meters(m, step=0.5):
    return round(m / step) * step

def phrase_distance(d_m):
    d = round_meters(d_m)
    if d < 0.5:
        return "a few steps"
    if d < 1.0:
        return "about one meter"
    return f"about {d:.1f} meters"

def heading_from(a, b):
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):
        return 'E' if dx > 0 else 'W'
    else:
        return 'S' if dy > 0 else 'N'

def relative_turn(prev_h, cur_h):
    """Translate heading changes into forward/right/left/back terms."""
    order = ['N', 'E', 'S', 'W']
    i1, i2 = order.index(prev_h), order.index(cur_h)
    diff = (i2 - i1) % 4
    if diff == 0:
        return "forward"
    elif diff == 1:
        return "right"
    elif diff == 3:
        return "left"
    else:
        return "back"

def is_hall(name):
    if not name:
        return False
    name_l = name.lower()
    return ('hall' in name_l) or ('corridor' in name_l)

def waypoint_label(wp):
    room = wp.get('room')
    wtype = wp.get('type', '')
    if wtype in ('start',):
        return f"your current position in {room}" if room else "your current position"
    if wtype in ('enter_hallway',):
        return "the hallway"
    if wtype in ('enter_room',):
        return f"the {room}" if room else "the room"
    if wtype in ('exit_room',):
        return f"the exit of {room}" if room else "the room exit"
    if wtype in ('goal',):
        return f"the {room}" if room else "your goal"
    if room:
        if is_hall(room):
            return "the hallway"
        return f"the {room}"
    return "the next point"

def load_plan(path_file):
    with open(path_file) as f:
        data = json.load(f)
    grid_res = data.get('grid_res_m', DEFAULT_GRID_RES_M)
    timestamp = data.get('timestamp')
    waypoints = data['waypoints']
    return waypoints, grid_res, timestamp

# --- Narration core -----------------------------------------------------------

def narrate_waypoints(waypoints, grid_res_m):
    """Produce a human-style route using forward/left/right/back with abstract phrasing."""
    if not waypoints:
        return "No route available."

    legs = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        (x1, y1), (x2, y2) = tuple(a['point']), tuple(b['point'])
        d_cells = abs(x2 - x1) + abs(y2 - y1)
        d_m = d_cells * grid_res_m
        legs.append({
            'from': a, 'to': b,
            'distance_m': d_m,
            'heading': heading_from((x1, y1), (x2, y2))
        })

    out = []

    if not legs:
        out.append("1) You are already at your destination.")
        return "\n".join(out)

    # First leg
    first = legs[0]
    prev_h = first['heading']
    dist_phrase = phrase_distance(first['distance_m'])
    to_wp = first['to']
    to_room = to_wp.get('room', '')
    wtype = to_wp.get('type', '')

    # If first target is the goal, skip it (no post-arrival navigation)
    if wtype != 'goal':
        if wtype == 'exit_room':
            out.append(f"1) Move forward {dist_phrase} to exit {to_room}.")
        elif wtype == 'enter_hallway':
            out.append(f"1) Move forward {dist_phrase} and enter the hallway.")
        elif wtype == 'enter_room':
            out.append(f"1) Move forward {dist_phrase} and enter the {to_room}.")
        else:
            target_label = waypoint_label(to_wp)
            out.append(f"1) Move forward {dist_phrase} toward {target_label}.")

    # Subsequent legs
    step_no = 2
    for i in range(1, len(legs)):
        leg = legs[i]
        cur_h = leg['heading']
        turn = relative_turn(prev_h, cur_h)
        prev_h = cur_h

        dist_phrase = phrase_distance(leg['distance_m'])
        to_wp = leg['to']
        to_room = to_wp.get('room', '')
        wtype = to_wp.get('type', '')

        # Skip any 'goal' leg to avoid post-arrival navigation
        if wtype == 'goal':
            continue

        prev_to_wp = legs[i-1]['to']
        prev_to_room = prev_to_wp.get('room', '')
        prev_wtype = prev_to_wp.get('type', '')

        already_in_hall = is_hall(prev_to_room) or (prev_wtype == 'enter_hallway')

        if wtype == 'enter_hallway' or is_hall(to_room):
            if already_in_hall:
                out.append(f"{step_no}) Turn {turn} and go {dist_phrase} along the hallway.")
            else:
                out.append(f"{step_no}) Turn {turn} and go {dist_phrase}, then enter the hallway.")
        elif wtype == 'enter_room':
            out.append(f"{step_no}) Turn {turn} and go {dist_phrase}, then enter the {to_room}.")
        elif wtype == 'exit_room':
            out.append(f"{step_no}) Turn {turn} and go {dist_phrase} to exit {to_room}.")
        else:
            if to_room:
                out.append(f"{step_no}) Turn {turn} and proceed {dist_phrase} toward the {to_room}.")
            else:
                out.append(f"{step_no}) Turn {turn} and proceed {dist_phrase}.")

        step_no += 1

    return "\n".join(out)

# --- Loop --------------------------------------------------------------------

def main():
    print("Relative narrator (forward/right/left/back) is running...")
    last_mod = 0
    while True:
        try:
            if os.path.exists(PLANNED_PATH_FILE):
                mod = os.path.getmtime(PLANNED_PATH_FILE)
                if mod != last_mod:
                    last_mod = mod
                    waypoints, grid_res_m, ts = load_plan(PLANNED_PATH_FILE)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    narration_text = narrate_waypoints(waypoints, grid_res_m)

                    result = {
                        "timestamp": timestamp,
                        "narration": narration_text
                    }

                    os.makedirs(os.path.dirname(ROUTE_NARRATION_FILE), exist_ok=True)
                    with open(ROUTE_NARRATION_FILE, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)

                    print(f"[{timestamp}] Route narration saved to {ROUTE_NARRATION_FILE}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Waiting for valid path file... ({e})")
            time.sleep(1)

if __name__ == "__main__":
    main()
