"""Ask a vision model, per kept frame, how likely its room holds each target category, and a text model the same from the room's objects; resumable, one jsonl row per question."""
import argparse
import base64
import json
from pathlib import Path

import requests

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import objects_text
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_text_rooms import object_lists
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS, yes_no_percent

TIMEOUT_S = 90
ANSWER = 'Answer with JSON {"percent": <a number from 0 to 100>}.'
ROOM_PHOTO = "This photo was taken inside one room of a house. "
VLM_PROMPT = ROOM_PHOTO + "How likely is it that this room contains a %s? " + ANSWER
PERCENT_PROMPTS = {"percent": VLM_PROMPT,
                   "unseen": ROOM_PHOTO + "Suppose there is no %s in this photo. How likely is it that one is somewhere else in this room, or in the space just next to it? " + ANSWER,
                   "nearby": ROOM_PHOTO + "How likely is it that a %s is within a few metres of where this photo was taken, out of view? " + ANSWER}
YES_NO = {"yesno": ROOM_PHOTO + "Is there a %s in this room? Answer yes or no.",
          "couldbe": ROOM_PHOTO + "Even if you cannot see one in the photo, could this room contain a %s? Answer yes or no.",
          "keep": "A robot is searching for a %s and cannot see one in this photo. Should it keep searching in and around this room? Answer yes or no."}
STYLES = tuple(PERCENT_PROMPTS) + tuple(YES_NO)
TEXT_PROMPT = "A room of a house contains these objects: %s. How likely is it that this room also contains a %s? " + ANSWER
PHRASE = {"tv_monitor": "TV or monitor"}
TEXT_FILE = "target_text_%s_%s.jsonl"
WALK_METHOD = "walk_keep3"


def vlm_file(model, label, style="percent"):
    """The answers file of a model, a frame source (walk or door) and a prompt style."""
    return "target_vlm_%s_%s%s.jsonl" % (model.replace(":", "_"), label, "" if style == "percent" else "_" + style)


def phrase(target):
    return PHRASE.get(target, target)


def ask_percent(url, model, prompt, jpeg_bytes=None):
    message = {"role": "user", "content": prompt}
    if jpeg_bytes is not None:
        message["images"] = [base64.b64encode(jpeg_bytes).decode()]
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 20, "num_ctx": 4096}, "messages": [message]}
    if jpeg_bytes is not None:
        body["think"] = False
    try:
        return requests.post(url, json=body, timeout=TIMEOUT_S).json()["message"]["content"]
    except Exception:
        return ""


def ask_yes_no(url, model, prompt, jpeg_bytes):
    """A JSON reply with the percent and logit of a yes/no answer read from the first token's log-probabilities; empty if unreadable."""
    body = {"model": model, "stream": False, "think": False, "logprobs": True, "top_logprobs": 10, "options": {"temperature": 0, "num_predict": 1, "num_ctx": 4096},
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(jpeg_bytes).decode()]}]}
    try:
        scored = yes_no_percent(requests.post(url, json=body, timeout=TIMEOUT_S).json()["logprobs"][0]["top_logprobs"])
    except Exception:
        return ""
    return "" if scored is None else json.dumps({"percent": scored[0], "logit": scored[1]})


def kept_frames(views, every, methods=(WALK_METHOD,)):
    """``{room: (folder, [frame names])}`` for every ``every``-th room, in name order, with the frames the named methods kept (none: not listed)."""
    rooms = {}
    for folder in sorted(d for d in views.iterdir() if (d / "selection.json").exists()):
        for room, entry in json.loads((folder / "selection.json").read_text()).items():
            frames = sorted({f for m in methods for f in entry.get("methods", {}).get(m, ())})
            if frames:
                rooms[room] = (folder, frames)
    return {room: rooms[room] for room in sorted(rooms)[::every]}


def done_rows(path, keys):
    return {tuple(json.loads(line)[k] for k in keys) for line in path.read_text().splitlines()} if path.exists() else set()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--walk", type=Path, required=True, help="folder of the walk-through views (selection.json per scene)")
    p.add_argument("--door", type=Path, help="folder of the door views (selection.json per scene), asked in addition")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--walk-method", default=WALK_METHOD, help="the selection method whose frames the walk pass asks")
    p.add_argument("--vlm", default="qwen3-vl:4b-instruct")
    p.add_argument("--style", choices=STYLES, default="percent", help="the question: a percent, or yes/no read from log-probabilities")
    p.add_argument("--text", default="qwen2.5:14b-instruct")
    p.add_argument("--sources", nargs="*", default=["yolo_world", "annotated"], help="object lists for the text model")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rooms = kept_frames(args.walk, args.every)
    for label, views, methods, every in (("walk", args.walk, (args.walk_method,), args.every), ("door", args.door, ("door0", "door4"), 1)):
        if views is None:
            continue
        vlm_path = args.out / vlm_file(args.vlm, label, args.style)
        done = done_rows(vlm_path, ("frame", "target"))
        with open(vlm_path, "a") as handle:
            for room, (folder, frames) in kept_frames(views, every, methods).items():
                for frame in frames:
                    image = (folder / "frames" / (frame + ".jpg")).read_bytes()
                    for target in TARGETS:
                        if (frame, target) not in done:
                            reply = ask_percent(args.url, args.vlm, PERCENT_PROMPTS[args.style] % phrase(target), image) if args.style in PERCENT_PROMPTS else ask_yes_no(args.url, args.vlm, YES_NO[args.style] % phrase(target), image)
                            handle.write(json.dumps({"room": room, "frame": frame, "target": target, "reply": reply}) + "\n")
                            handle.flush()
    for source in args.sources:
        lists = object_lists(args.rooms, source, "inside")
        text_path = args.out / (TEXT_FILE % (args.text.replace(":", "_"), source))
        done = done_rows(text_path, ("room", "target"))
        with open(text_path, "a") as handle:
            for room in rooms:
                for target in TARGETS:
                    if (room, target) not in done:
                        reply = ask_percent(args.url, args.text, TEXT_PROMPT % (objects_text(lists.get(room, [])), phrase(target)))
                        handle.write(json.dumps({"room": room, "target": target, "reply": reply}) + "\n")
                        handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
