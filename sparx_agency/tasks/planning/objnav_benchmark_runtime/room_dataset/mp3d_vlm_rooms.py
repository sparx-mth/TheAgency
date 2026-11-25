"""Ask a vision model what room each frame of the Matterport3D regions shows (open question or a fixed list); resumable, one jsonl row per frame."""
import argparse
import base64
import json
import time
from pathlib import Path

import requests

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import OPEN_PROMPT, list_prompt, spread_frames

TIMEOUT_S = 90


def ask(url, model, prompt, jpeg_bytes=None):
    message = {"role": "user", "content": prompt}
    if jpeg_bytes is not None:
        message["images"] = [base64.b64encode(jpeg_bytes).decode()]
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 30, "num_ctx": 4096}, "messages": [message]}
    if jpeg_bytes is not None:
        body["think"] = False
    try:
        return str(json.loads(requests.post(url, json=body, timeout=TIMEOUT_S).json()["message"]["content"])["room"])
    except Exception:
        return ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", choices=("open", "list"), required=True)
    p.add_argument("--frames-per-room", type=int, default=6)
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    out = args.rooms / ("vlm_%s_%s.jsonl" % (args.model.replace(":", "_"), args.prompt))
    done = {json.loads(line)["frame"] for line in out.read_text().splitlines()} if out.exists() else set()
    prompt = OPEN_PROMPT if args.prompt == "open" else list_prompt()
    with open(out, "a") as handle:
        for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists()):
            for room in json.loads((folder / "regions.json").read_text())["rooms"]:
                for f in spread_frames(room["frames"], args.frames_per_room):
                    if f["frame"] in done:
                        continue
                    started = time.time()
                    answer = ask(args.url, args.model, prompt, (folder / "frames" / (f["frame"] + ".jpg")).read_bytes())
                    handle.write(json.dumps({"room": room["room"], "frame": f["frame"], "answer": answer, "seconds": round(time.time() - started, 2)}) + "\n")
                    handle.flush()
            print(folder.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
