"""Ask a vision model what room each room frame was taken in (open answer, no list of names) and store the answers."""
import argparse
import base64
import json
from pathlib import Path

import requests

PROMPT = 'What room in a house was this photo taken in? Answer with JSON {"room": "<name>"}.'
TIMEOUT_S = 60


def ask(url, model, jpeg_bytes):
    body = {"model": model, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 30, "num_ctx": 4096},
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(jpeg_bytes).decode()]}]}
    try:
        return str(json.loads(requests.post(url, json=body, timeout=TIMEOUT_S).json()["message"]["content"])["room"])
    except Exception:
        return ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--model", default="qwen3-vl:4b-instruct")
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    args = p.parse_args(argv)
    out = args.views / ("room_guess_%s.jsonl" % args.model.replace(":", "_"))
    with open(out, "w") as handle:
        for folder in sorted(d for d in args.views.iterdir() if (d / "views.jsonl").exists()):
            for line in (folder / "views.jsonl").read_text().splitlines():
                view = json.loads(line)
                answer = ask(args.url, args.model, (folder / "frames" / (view["frame"] + ".jpg")).read_bytes())
                handle.write(json.dumps({"scene": view["scene"], "room": view["room"], "frame": view["frame"], "answer": answer}) + "\n")
            print(folder.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
