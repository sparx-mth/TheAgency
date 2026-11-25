"""Ask a vision model with the room list what room each selected Matterport3D view shows; resumable, one jsonl row per distinct frame."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import OPEN_PROMPT, list_prompt
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_rooms import ask


def selections(out):
    for folder in sorted(d for d in out.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            if "methods" in entry:
                yield folder, room, entry


def answer_file(out, model, prompt):
    return out / ("vlm_%s_%s.jsonl" % (model.replace(":", "_"), prompt))


def frames_to_ask(entry, methods):
    """The frames of a region to ask about: all of them, or only those the named methods kept."""
    return list(entry["frames"]) if methods is None else sorted({f for m in methods for f in entry["methods"].get(m, ())})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="qwen3-vl:4b-instruct")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    p.add_argument("--prompt", choices=("open", "list"), default="list")
    p.add_argument("--methods", nargs="*", help="ask only about the frames these methods kept")
    args = p.parse_args(argv)
    answers = answer_file(args.out, args.model, args.prompt)
    prompt = OPEN_PROMPT if args.prompt == "open" else list_prompt()
    done = {json.loads(line)["frame"] for line in answers.read_text().splitlines()} if answers.exists() else set()
    with open(answers, "a") as handle:
        for folder, room, entry in selections(args.out):
            for frame in frames_to_ask(entry, args.methods):
                if frame not in done:
                    done.add(frame)
                    handle.write(json.dumps({"room": room, "frame": frame, "answer": ask(args.url, args.model, prompt, (folder / "frames" / (frame + ".jpg")).read_bytes())}) + "\n")
                    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
