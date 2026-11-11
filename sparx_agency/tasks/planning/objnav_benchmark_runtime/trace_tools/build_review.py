"""Room review page from the landmark-logged trace: top-down map, landmark table, in-room frames, label."""
import base64
import os
import collections
import html
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SCENES = ("00853-5cdEh9F2hJL", "00890-6s7QHgap2fW")
SHOWN_CONF = 0.3
FRAMES_PER_ROOM = 4
MIN_LABELLED_SHARE = 0.6
MIN_LABELLED_FRACTION = 0.2
MIN_FLOOR_SHARE = 0.5
MIN_YAW_APART = math.radians(40)
RULE = os.environ.get("FRAME_RULE", "visible")
PALETTE = {}


def colour(cls):
    if cls not in PALETTE:
        hue = (len(PALETTE) * 37) % 180
        PALETTE[cls] = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
    return PALETTE[cls]


def jpg64(image, quality=75):
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def prompt_counts(user):
    return collections.Counter({c: int(n) for c, n in re.findall(r"- (.+?) x(\d+)", user.split("Room geometry")[0])})


def xy(record):
    pose = record.get("pose")
    return [pose["x"], pose["y"]] if pose else None


def matching_pid(snapshot, counts):
    hits = []
    for pid in snapshot["rooms"]:
        have = collections.Counter(m["cls"] for m in snapshot["landmarks"] if str(m["pid"]) == pid)
        if have == counts:
            hits.append(pid)
    return hits


def topdown(snapshot, pid, trail):
    points = [c for room in snapshot["rooms"].values() for c in room["cells"]] + [m["xy"] for m in snapshot["landmarks"]] + trail
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0, size, pad = min(xs) - 1, min(ys) - 1, 0.08, 20
    w, h = int((max(xs) + 1 - x0) / size) + 2 * pad, int((max(ys) + 1 - y0) / size) + 2 * pad
    scale = 460 / max(w, h)

    def px(p):
        return (int(((p[0] - x0) / size + pad) * scale), int((h - ((p[1] - y0) / size + pad)) * scale))

    img = np.full((int(h * scale), int(w * scale), 3), 255, np.uint8)
    for key, room in snapshot["rooms"].items():
        shade = (150, 200, 150) if key == pid else (225, 225, 225)
        for cell in room["cells"]:
            cv2.circle(img, px(cell), max(2, int(0.14 / size * scale)), shade, -1)
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, px(a), px(b), (200, 140, 60), 1)
    for m in snapshot["landmarks"]:
        inside = str(m["pid"]) == pid
        cv2.circle(img, px(m["xy"]), 6 if inside else 3, colour(m["cls"]), -1 if inside else 1)
        if inside:
            cv2.putText(img, "%s(%d)" % (m["cls"], m["n"]), (px(m["xy"])[0] + 6, px(m["xy"])[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    if trail:
        cv2.circle(img, px(trail[-1]), 5, (0, 0, 255), -1)
    return img


def inside_frames(records, call, cells):
    """The old rule: frames taken while the agent stood inside the room's outline."""
    return [r for r in records if r["kind"] == "detect" and r["call"] <= call and r.get("frame") and xy(r)
            and min(math.dist(xy(r), c) for c in cells) <= 0.45]


def room_score(record, pid):
    """How much of the frame shows this room: its share of the labelled pixels (walls fall outside every room), or of the floor pixels."""
    shares = record["frame_rooms"]
    labelled = 1 - shares["all"].get("outside", 0)
    by_label = shares["all"].get(pid, 0) / labelled if labelled >= MIN_LABELLED_FRACTION else 0
    by_floor = shares["floor"].get(pid, 0) if shares["floor"] else 0
    ok = by_label >= MIN_LABELLED_SHARE or by_floor >= MIN_FLOOR_SHARE
    return max(by_label, by_floor) if ok else 0


def visible_frames(records, call, pid):
    """Frames that mostly show this room, best first, spread over different headings."""
    ranked = sorted((r for r in records if r["kind"] == "detect" and r["call"] <= call and r.get("frame") and r.get("frame_rooms")
                     and room_score(r, pid) > 0), key=lambda r: -room_score(r, pid))
    picks = []
    for r in ranked:
        if len(picks) < FRAMES_PER_ROOM and all(abs(math.remainder(r["pose"]["yaw"] - q["pose"]["yaw"], 2 * math.pi)) >= MIN_YAW_APART for q in picks):
            picks.append(r)
    return picks


def pick_frames(records, call, pid, cells):
    if RULE == "position":
        inside = inside_frames(records, call, cells)
        return [inside[int(i * (len(inside) - 1) / max(1, FRAMES_PER_ROOM - 1))] for i in range(min(FRAMES_PER_ROOM, len(inside)))]
    return visible_frames(records, call, pid)


def draw_frames(picks, folder):
    images = []
    for r in picks:
        img = cv2.imread(str(folder / "frames" / r["frame"]))
        for b in r["boxes"]:
            if b["conf"] < SHOWN_CONF:
                continue
            x0, y0, x1, y1 = (int(v * 0.8) for v in b["xyxy"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.putText(img, "%s %.2f" % (b["cls"], b["conf"]), (x0 + 1, max(y0 + 11, 11)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        images.append(img)
    return images


cards = []
for scene in SCENES:
    folder = ROOT / scene
    by_episode = collections.defaultdict(list)
    for line in open(folder / "trace.jsonl"):
        r = json.loads(line)
        by_episode[r["episode"]].append(r)
    for episode, records in by_episode.items():
        last = {}
        for r in records:
            if r["kind"] != "llm" or not r["user"].startswith("Room observed objects") or not r.get("snapshot"):
                continue
            counts = prompt_counts(r["user"])
            for pid in matching_pid(r["snapshot"], counts):
                last[pid] = r
        for pid, r in last.items():
            snapshot = r["snapshot"]
            trail = [xy(x) for x in records if x["kind"] == "detect" and x["call"] <= r["call"] and xy(x)]
            cells = snapshot["rooms"][pid]["cells"]
            cards.append(dict(scene=scene, episode=episode, pid=pid, record=r, snapshot=snapshot, trail=trail, cells=cells,
                              picks=pick_frames(records, r["call"], pid, cells), folder=folder, target=next((x["target"] for x in records if x["kind"] == "detect"), "")))
print(len(cards), "rooms with a matched final label")

limit = int(sys.argv[3]) if len(sys.argv) > 3 else len(cards)
parts = ["<!doctype html><meta charset=utf-8><title>Room review</title><style>body{font:14px sans-serif;margin:16px;background:#fafafa}"
         ".c{background:#fff;border:1px solid #ccc;border-radius:8px;margin:14px 0;padding:10px}.l{font-size:20px;font-weight:700}"
         ".row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}.f{display:grid;grid-template-columns:repeat(2,260px);gap:4px}.f img{width:260px}"
         "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:2px 6px;font-size:12px}small{color:#666}</style>"
         "<h2>Room classification review (open YOLO), frames chosen by " + RULE + "</h2><p>Map: this room green, other rooms grey, agent path blue, red dot = agent when the label was asked. "
         "Landmarks of this room are filled and named (n = observations merged); landmarks in other rooms are hollow. Frames: see the rule in the title; "
         "boxes at conf &ge; 0.3.</p>"]
for number, c in enumerate(cards[:limit], 1):
    r, snap, pid = c["record"], c["snapshot"], c["pid"]
    mine = [m for m in snap["landmarks"] if str(m["pid"]) == pid]
    table = "".join("<tr><td>%s</td><td>%d</td><td>(%.1f, %.1f)</td></tr>" % (m["cls"], m["n"], m["xy"][0], m["xy"][1]) for m in sorted(mine, key=lambda m: m["cls"]))
    frames = "".join("<img src='data:image/jpeg;base64,%s'>" % jpg64(f) for f in draw_frames(c["picks"], c["folder"])) or "<small>no saved frame shows this room</small>"
    geometry = html.escape(r["user"].split("Room geometry:")[1].split("Classify")[0].strip().replace("\n", "; "))
    parts.append("<div class=c><div class=l>#%d  %s <small>(conf %s)</small></div><small>%s episode %s, room %s</small><br>%s<br><b>geometry sent:</b> %s"
                 "<div class=row><img src='data:image/jpeg;base64,%s'><div><table><tr><th>landmark</th><th>n</th><th>xy</th></tr>%s</table></div><div class=f>%s</div></div></div>" % (
                     number, html.escape(str(r["reply"].get("label"))), r["reply"].get("confidence"), c["scene"][:5], c["episode"].split("_")[-1], pid,
                     "<b>reply reasoning:</b> " + html.escape(str(r["reply"].get("reasoning", ""))), geometry,
                     jpg64(topdown(snap, pid, c["trail"])), table, frames))
OUT.write_text("".join(parts))
print(OUT, OUT.stat().st_size // 1024, "KB")
