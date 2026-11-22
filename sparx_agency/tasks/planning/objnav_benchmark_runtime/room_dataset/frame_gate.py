"""Ask a vision model, once per usable sweep view, whether the view shows enough of the room to tell what kind of room it is; resumable, one jsonl row per view."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import ask_yes_no, done_rows
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import load_sweep
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import usable_frames

GATE_PROMPT = "This photo was taken inside one room of a house. Does it show enough of the room to tell what kind of room it is? Answer yes or no."
GATE_FILE = "gate_%s.jsonl"
MIN_GAIN_LOWER_BOUND = 0.0


def gate_logits(rows):
    """``{view name: logit of yes}`` for the views with a readable answer."""
    out = {}
    for row in rows:
        try:
            out[row["frame"]] = float(json.loads(row["reply"])["logit"])
        except (ValueError, KeyError, TypeError):
            continue
    return out


def gate_verdict(gain):
    """``keep`` when the lower end of the gain over random choice is above ``MIN_GAIN_LOWER_BOUND``; set before the run."""
    return "keep" if gain[1] > MIN_GAIN_LOWER_BOUND else "not shown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (GATE_FILE % args.vlm.replace(":", "_"))
    done = done_rows(path, ("frame",))
    with open(path, "a") as handle:
        for room, entry in load_sweep(args.views).items():
            for frame in sorted(usable_frames(entry["frames"])):
                if (frame,) not in done:
                    image = (args.views / entry["scene"] / "frames" / (frame + ".jpg")).read_bytes()
                    handle.write(json.dumps({"room": room, "frame": frame, "reply": ask_yes_no(args.url, args.vlm, GATE_PROMPT, image)}) + "\n")
                    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
