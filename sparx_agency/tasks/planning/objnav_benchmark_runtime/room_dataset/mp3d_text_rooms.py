"""Ask a text model what room a Matterport3D region is from the list of objects in it (annotated, or the detector's); resumable, one jsonl row per region."""
import argparse
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import objects_prompt, objects_text
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import clean_category
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_vlm_rooms import ask


def object_lists(rooms, source, kind):
    """``{room: [object names]}``: the annotated objects of each region, or a detector's ``inside`` / ``all`` list from its objects file."""
    if source == "annotated":
        return {r["room"]: [n for n in r["objects"] if clean_category(n)] for f in sorted(rooms.iterdir()) if (f / "regions.json").exists()
                for r in json.loads((f / "regions.json").read_text())["rooms"]}
    return {room: found[kind] for room, found in json.loads((rooms / ("objects_%s.json" % source)).read_text()).items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--model", default="qwen2.5:14b-instruct")
    p.add_argument("--source", required=True, help="annotated, or a detector name whose objects_<name>.json exists")
    p.add_argument("--kind", choices=("inside", "all"), default="inside")
    p.add_argument("--prompt", choices=("open", "list"), required=True)
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    name = args.source if args.source == "annotated" else "%s_%s" % (args.source, args.kind)
    out = args.rooms / ("text_%s_%s_%s.jsonl" % (args.model.replace(":", "_"), name, args.prompt))
    done = {json.loads(line)["room"] for line in out.read_text().splitlines()} if out.exists() else set()
    with open(out, "a") as handle:
        for room, names in object_lists(args.rooms, args.source, args.kind).items():
            if room not in done:
                answer = ask(args.url, args.model, objects_prompt(objects_text(names), with_list=args.prompt == "list"))
                handle.write(json.dumps({"room": room, "answer": answer, "objects": objects_text(names)}) + "\n")
                handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
