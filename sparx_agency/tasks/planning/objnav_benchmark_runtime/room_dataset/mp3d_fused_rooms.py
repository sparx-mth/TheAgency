"""Ask a text model what room a Matterport3D region is from its object list and how the vision model voted over the region's photos; resumable, one jsonl row per region."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import fused_prompt, objects_text, vote_counts
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_text_rooms import object_lists
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_rooms import ask


def frame_answers(path):
    answers = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        answers.setdefault(row["room"], []).append(row["answer"])
    return answers


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--vlm-file", type=Path, required=True)
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--source", required=True)
    p.add_argument("--kind", choices=("inside", "all"), default="inside")
    p.add_argument("--prompt", choices=("open", "list"), required=True)
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    name = args.source if args.source == "annotated" else "%s_%s" % (args.source, args.kind)
    out = args.rooms / ("fused_%s_%s_%s.jsonl" % (args.model.replace(":", "_"), name, args.prompt))
    done = {json.loads(line)["room"] for line in out.read_text().splitlines()} if out.exists() else set()
    votes = frame_answers(args.vlm_file)
    with open(out, "a") as handle:
        for room, names in object_lists(args.rooms, args.source, args.kind).items():
            if room in done or room not in votes:
                continue
            prompt = fused_prompt(objects_text(names), vote_counts(votes[room]), with_list=args.prompt == "list")
            handle.write(json.dumps({"room": room, "answer": ask(args.url, args.model, prompt), "objects": objects_text(names)}) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
