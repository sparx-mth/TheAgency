"""Print a step-by-step timeline of one recorded ObjectNav episode from its ``steps.jsonl``.

Development tool, not a test. Compresses consecutive steps that share the same
supervisor state, room and command kind into one row, and lists the room-search
loop's events, the LLM rounds, the detections that confirmed a target and the
final verdict -- the reading a person needs before opening the video.

    venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.timeline_room_search_loop <recording dir>
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path


def _rows(path):
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def _phase(row):
    method = row.get("method") or {}
    command = row.get("command") or {}
    info = command.get("info") or {}
    kind = info.get("kind") or ("STOP" if command.get("stop") else info.get("reason", "-"))
    phase = info.get("phase") or method.get("state") or "-"
    return str(phase), method.get("room_id"), str(kind)


def main(argv):
    rec = Path(argv[1])
    rows = list(_rows(rec / "steps.jsonl"))
    episode = json.load(open(rec / "episode.json"))
    print("episode %s  target=%s  steps=%d" % (episode.get("episode_id"), episode.get("target"), len(rows)))
    first = rows[0]["pose"]
    print("start pose x=%.2f y=%.2f yaw=%.2f" % (first["x"], first["y"], first["yaw"]))

    print("\n== timeline (consecutive steps with one state/room/command kind collapsed) ==")
    print("%-9s %-10s %-5s %-22s %-5s %-30s %s" % ("steps", "phase", "room", "command", "n", "actions", "pose at start -> end"))
    start = 0
    key = _phase(rows[0])
    actions = collections.Counter()
    for i, row in enumerate(rows + [None]):
        here = _phase(row) if row is not None else None
        if row is not None:
            actions[(row.get("decision") or {}).get("action", "?")] += 1
        if here != key or row is None:
            a, b = rows[start]["pose"], rows[i - 1]["pose"]
            moved = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            summary = " ".join("%s:%d" % (k.replace("MOVE_", "").replace("TURN_", "T_"), v) for k, v in sorted(actions.items()))
            rng = "%d-%d" % (start, i - 1) if i - 1 > start else "%d" % start
            print("%-9s %-10s %-5s %-22s %-5d %-30s (%.1f,%.1f)->(%.1f,%.1f) %.1fm" % (
                rng, key[0][:10], key[1] if key[1] is not None else "-", key[2][:22], i - start, summary[:30],
                a["x"], a["y"], b["x"], b["y"], moved))
            start, key, actions = i, here, collections.Counter()
        if row is None:
            break

    print("\n== rooms and reasoning over time ==")
    last = None
    for row in rows:
        m = row.get("method") or {}
        rooms = m.get("rooms")
        n_rooms = len(rooms) if isinstance(rooms, (list, dict)) else rooms
        reasoning = m.get("reasoning") or {}
        oracle = reasoning.get("oracle") if isinstance(reasoning, dict) else None
        scores = json.dumps((oracle or {}).get("scores"), sort_keys=True) if oracle else None
        key = (n_rooms, scores)
        if key != last:
            top = ""
            if oracle and oracle.get("probs"):
                probs = sorted(oracle["probs"].items(), key=lambda kv: -kv[1])[:4]
                top = "  oracle top: " + ", ".join("R%s=%.2f" % (k, v) for k, v in probs) + "  p_present=%.2f%s" % (
                    oracle.get("p_present", 1.0), " (reused)" if oracle.get("reused") else "")
            print("  step %3d  rooms=%s%s" % (row["step"], n_rooms, top))
            last = key

    print("\n== detections that reached the target track ==")
    for row in rows:
        m = row.get("method") or {}
        for det in m.get("detections") or []:
            label = det.get("label") if isinstance(det, dict) else None
            if label and episode.get("target") and str(label).lower() in str(episode.get("target")).lower():
                print("  step %3d  %s" % (row["step"], json.dumps(det)[:160]))
                break

    last_row = rows[-1]
    print("\n== final ==")
    print("last command:", json.dumps((last_row.get("command") or {}).get("info"))[:300])
    print("last decision:", json.dumps(last_row.get("decision"))[:200])
    print("final pose: x=%.2f y=%.2f" % (last_row["pose"]["x"], last_row["pose"]["y"]))


if __name__ == "__main__":
    main(sys.argv)

