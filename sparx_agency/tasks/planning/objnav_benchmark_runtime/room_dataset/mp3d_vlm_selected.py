"""Ask a vision model with the room list what room each selected Matterport3D view shows; resumable, one jsonl row per distinct frame."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import list_prompt
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_rooms import ask


def selections(out):
    for folder in sorted(d for d in out.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            if "methods" in entry:
                yield folder, room, entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3-vl:4b-instruct")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    answers = args.out / ("vlm_%s_list.jsonl" % args.model.replace(":", "_"))
    done = {json.loads(line)["frame"] for line in answers.read_text().splitlines()} if answers.exists() else set()
    with open(answers, "a") as handle:
        for folder, room, entry in selections(args.out):
            for frame in entry["frames"]:
                if frame not in done:
                    done.add(frame)
                    handle.write(json.dumps({"room": room, "frame": frame, "answer": ask(args.url, args.model, list_prompt(), (folder / "frames" / (frame + ".jpg")).read_bytes())}) + "\n")
                    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
