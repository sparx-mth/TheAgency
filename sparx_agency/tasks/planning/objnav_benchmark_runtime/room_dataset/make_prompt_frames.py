"""The three frames with most room floor of each swept room, as a views folder the scorer reads, so every prompt wording sees the same pictures."""
import argparse
import json
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import frame_picks

METHOD = "pick3"
FRAMES = 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweeps", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    for sweep in args.sweeps:
        for folder in sorted(d for d in (sweep / "views").iterdir() if (d / "selection.json").exists()):
            result = {}
            for room, entry in json.loads((folder / "selection.json").read_text()).items():
                if "frames" not in entry:
                    continue
                frames = {n: dict(f, entrance_m=f["entrance_m"] if f["entrance_m"] is not None else 0.0) for n, f in entry["frames"].items()}
                picks = frame_picks(frames, "floor", FRAMES, np.random.default_rng(0))
                if picks:
                    result[room] = {"group": "", "frames": {n: entry["frames"][n] for n in picks}, "visible": {n: entry["visible"][n] for n in picks}, "methods": {METHOD: picks}, "actions": {METHOD: None}}
            target = args.out / folder.name
            (target / "frames").mkdir(parents=True, exist_ok=True)
            for entry in result.values():
                for name in entry["frames"]:
                    link = target / "frames" / (name + ".jpg")
                    if not link.exists():
                        link.symlink_to(folder / "frames" / (name + ".jpg"))
            merged = json.loads((target / "selection.json").read_text()) if (target / "selection.json").exists() else {}
            (target / "selection.json").write_text(json.dumps({**merged, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
