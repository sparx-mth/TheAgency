#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
object_inroom_describer.py  (simplified)
---------------------------------------
Reads the object + room, finds up to two nearest other objects,
and writes a short "next to ..." description as JSON.
"""

import json, os, math, time

UNIFIED_ROOMS_FILE      = "data/unified_rooms.json"
OBJECT_LOCATION_FILE    = "data/object_location.json"
INROOM_DESCRIPTION_FILE = "data/inroom_description.json"
CHECK_INTERVAL_SEC      = 0.5


def load_rooms():
    if not os.path.exists(UNIFIED_ROOMS_FILE):
        return {}
    with open(UNIFIED_ROOMS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rooms", {})


def find_object(objs, obj_type):
    for o in objs:
        if o.get("type") == obj_type:
            return o
    return None


def center(bbox):
    # bbox: [x1, y1, x2, y2]
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def nearest_names(objs, target, k=2):
    """Return names of up to k nearest other objects (by center distance)."""
    tb = target.get("bbox")
    if not (isinstance(tb, list) and len(tb) == 4):
        return []
    tx, ty = center(tb)

    dists = []
    for o in objs:
        if o is target:
            continue
        b = o.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            ox, oy = center(b)
            dists.append((math.hypot(tx - ox, ty - oy), o.get("type", "object")))
    dists.sort(key=lambda x: x[0])
    return [name for _, name in dists[:k]]


def describe_next_to(room_dict, room, obj_type):
    objs = room_dict.get(room, {}).get("objects", [])
    if not objs:
        return f"I cannot find any objects in {room}."

    target = find_object(objs, obj_type)
    if not target:
        return f"I cannot find the {obj_type} in {room}."

    names = nearest_names(objs, target, k=2)
    if not names:
        return f"You can easily find the {obj_type} it is the only object in the room."
    if len(names) == 1:
        return f"It’s next to the {names[0]}."
    return f"It’s next to the {names[0]} and the {names[1]}."


def main():
    print("IN-ROOM OBJECT DESCRIBER (simple) running...")
    last_mod = 0
    rooms = {}

    while True:
        try:
            if not os.path.exists(OBJECT_LOCATION_FILE):
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            m = os.path.getmtime(OBJECT_LOCATION_FILE)
            if m == last_mod:
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            last_mod = m
            rooms = load_rooms()
            if not rooms:
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            with open(OBJECT_LOCATION_FILE, "r", encoding="utf-8") as f:
                loc = json.load(f)

            obj  = loc.get("found_object") or loc.get("object")
            room = loc.get("found_room")   or loc.get("room")

            if not obj or not room or obj == "none" or room == "none":
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            desc = describe_next_to(rooms, room, obj)
            result = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "room": room,
                "object": obj,
                "description": desc
            }

            os.makedirs(os.path.dirname(INROOM_DESCRIPTION_FILE), exist_ok=True)
            with open(INROOM_DESCRIPTION_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"[{result['timestamp']}] {room}/{obj}: {desc}")

            time.sleep(CHECK_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
