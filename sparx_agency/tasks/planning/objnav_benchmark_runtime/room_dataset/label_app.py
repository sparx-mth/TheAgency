"""A local page to name each room from its frames out of a fixed list of room types; every choice is saved to a CSV at once."""
import argparse
import collections
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_store import ROOM_TYPES, LabelStore
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import floor_map, object_summary, pick_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.room_names import majority_name

FRAMES_SHOWN = 6
PAGE = Path(__file__).with_name("label_app.html")


class Rooms:
    def __init__(self, views, guesses):
        self.views, self.rooms, self.chosen = views, [], {}
        answers = collections.defaultdict(list)
        if guesses:
            for line in guesses.read_text().splitlines():
                row = json.loads(line)
                answers[row["room"]].append(row["answer"])
        for folder in sorted(d for d in views.iterdir() if (d / "rooms.json").exists()):
            frames = [json.loads(line) for line in (folder / "views.jsonl").read_text().splitlines()]
            for room in json.loads((folder / "rooms.json").read_text()):
                shown = pick_frames([v for v in frames if v["room"] == room["room"]], count=FRAMES_SHOWN)
                self.chosen[room["room"]] = (folder, room, shown)
                self.rooms.append({"room": room["room"], "scene": room["scene"], "area_m2": round(room["area_m2"]), "objects": object_summary(room),
                                   "guess": majority_name(answers[room["room"]])[0],
                                   "frames": ["/frame/%s/%s.jpg" % (room["scene"], v["frame"]) for v in shown]})

    def map_png(self, room):
        folder, info, shown = self.chosen[room]
        cells = np.load(folder / "cells.npz")
        return cv2.imencode(".png", floor_map(cells["cells"], cells["regions"], info, shown))[1].tobytes()


def handler(rooms, store):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, kind="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if self.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(json.dumps({"rooms": rooms.rooms, "options": ROOM_TYPES, "labels": store.all()}).encode())
            elif parts[0] == "frame" and len(parts) == 3 and ".." not in self.path:
                self.reply((rooms.views / parts[1] / "frames" / parts[2]).read_bytes(), mimetypes.guess_type(parts[2])[0])
            elif parts[0] == "map" and parts[1].removesuffix(".png") in rooms.chosen:
                self.reply(rooms.map_png(parts[1].removesuffix(".png")), "image/png")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/label":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                store.set(body["room"], body["room_type"])
            except (KeyError, ValueError):
                self.send_error(400)
                return
            self.reply(b"{}")

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--guesses", type=Path, help="the room_guess_*.jsonl shown only when the page's guess toggle is on")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(Rooms(args.views, args.guesses), LabelStore(args.labels)))
    print("labelling page on http://127.0.0.1:%d" % args.port, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
