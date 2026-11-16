"""One review image per room (floor map, four frames, the annotated objects, the model's guess) and a CSV for Daphna's room names."""
import argparse
import collections
import csv
import json
import string
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.room_names import majority_name

FRAME_SHOWN = (460, 345)
MAP_SIZE = 560
MAP_MARGIN_M = 2.0
FRAMES_PER_SHEET = 4
MAX_FRAMES_PER_SPOT = 2
MIN_IN_ROOM_SHARE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
ROOM_COLOUR, OTHER_COLOUR, SPOT_COLOUR = (200, 120, 30), (200, 200, 200), (30, 30, 220)


def in_room_share(view):
    """The share of the annotated pixels of a view that belong to the view's own room: low when it looks out through a door."""
    total = sum(o["pixels"] for o in view["visible"])
    return sum(o["pixels"] for o in view["visible"] if o["region"] == view["region"]) / total if total else 0.0


def pick_frames(views, count=FRAMES_PER_SHEET, per_spot=MAX_FRAMES_PER_SPOT, min_share=MIN_IN_ROOM_SHARE):
    """The views that look mostly at the room itself, best first, at most ``per_spot`` from one standing spot."""
    ranked = sorted((v for v in views if in_room_share(v) >= min_share), key=lambda v: (-in_room_share(v), v["spot"], v["heading"]))
    taken, per = [], collections.Counter()
    for view in ranked:
        if per[view["spot"]] < per_spot:
            taken.append(view)
            per[view["spot"]] += 1
        if len(taken) == count:
            break
    return taken


def floor_map(cells, regions, room, chosen):
    near_floor = np.abs(cells[:, 1] - room["floor_y"]) < 1.0
    mine = cells[(regions == room["region"]) & near_floor]
    low, high = mine[:, [0, 2]].min(axis=0) - MAP_MARGIN_M, mine[:, [0, 2]].max(axis=0) + MAP_MARGIN_M
    scale = MAP_SIZE / max(high - low)

    def pixel(x, z):
        return int((x - low[0]) * scale), int((z - low[1]) * scale)

    image = np.full((MAP_SIZE, MAP_SIZE, 3), 255, np.uint8)
    for (x, _, z), region in zip(cells[near_floor], regions[near_floor]):
        cv2.circle(image, pixel(x, z), 3, ROOM_COLOUR if region == room["region"] else OTHER_COLOUR, -1)
    for letter, view in zip(string.ascii_uppercase, chosen):
        centre = pixel(view["x"], view["z"])
        tip = pixel(view["x"] - 0.8 * np.sin(view["yaw"]), view["z"] - 0.8 * np.cos(view["yaw"]))
        cv2.arrowedLine(image, centre, tip, SPOT_COLOUR, 2, tipLength=0.35)
        cv2.putText(image, letter, (centre[0] + 4, centre[1] - 4), FONT, 0.7, SPOT_COLOUR, 2)
    return image


def object_summary(room):
    counts = collections.Counter(o["name"] for o in room["objects"])
    return ", ".join("%s x%d" % (n, c) if c > 1 else n for n, c in sorted(counts.items()))


def wrapped(text, width):
    lines, line = [], ""
    for word in text.split(", "):
        if len(line) + len(word) > width:
            lines.append(line)
            line = ""
        line += word + ", "
    return lines + [line.rstrip(", ")]


def sheet(folder, room, views, guess):
    cells = np.load(folder / "cells.npz")
    chosen = pick_frames(views)
    frames = [cv2.resize(cv2.imread(str(folder / "frames" / (v["frame"] + ".jpg"))), FRAME_SHOWN) for v in chosen]
    for letter, frame in zip("ABCD", frames):
        cv2.putText(frame, letter, (10, 34), FONT, 1.2, (30, 30, 220), 3)
    frames += [np.full_like(frames[0], 255)] * (FRAMES_PER_SHEET - len(frames))
    grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
    top = np.hstack([floor_map(cells["cells"], cells["regions"], room, chosen), grid[:MAP_SIZE]])
    text = np.full((260, top.shape[1], 3), 255, np.uint8)
    name, share = guess
    head = "%s   scene %s   %.0f m2   model guess: %s (%.0f%% of frames)" % (room["room"], room["scene"], room["area_m2"], name or "-", 100 * share)
    cv2.putText(text, head, (10, 30), FONT, 0.7, (0, 0, 0), 2)
    for i, line in enumerate(wrapped("annotated objects: " + object_summary(room), 130)[:7]):
        cv2.putText(text, line, (10, 65 + 28 * i), FONT, 0.6, (60, 60, 60), 1)
    return np.vstack([top, text])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--guesses", type=Path, required=True, help="the room_guess_*.jsonl of vlm_room_guess")
    args = p.parse_args(argv)
    answers = collections.defaultdict(list)
    for line in args.guesses.read_text().splitlines():
        row = json.loads(line)
        answers[row["room"]].append(row["answer"])
    (args.views / "review").mkdir(exist_ok=True)
    with open(args.views / "room_labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["room", "scene", "model_guess", "your_room_name", "not_a_room_or_unsure"])
        for folder in sorted(d for d in args.views.iterdir() if (d / "rooms.json").exists()):
            views = [json.loads(line) for line in (folder / "views.jsonl").read_text().splitlines()]
            for room in json.loads((folder / "rooms.json").read_text()):
                guess = majority_name(answers[room["room"]])
                cv2.imwrite(str(args.views / "review" / (room["room"] + ".jpg")),
                            sheet(folder, room, [v for v in views if v["room"] == room["room"]], guess), [cv2.IMWRITE_JPEG_QUALITY, 90])
                writer.writerow([room["room"], room["scene"], guess[0], "", ""])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
