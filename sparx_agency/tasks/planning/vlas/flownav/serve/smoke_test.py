"""One-command smoke test for the FlowNav image-goal host server.

Sends a current frame + a goal image to a running ``flownav_trt_server`` and
prints the returned body-frame waypoints. With NO ``--obs``/``--goal`` it uses
synthetic random images, so it verifies the client -> server -> engines
round-trip without needing any files on disk (the inputs are noise, so the
waypoints are meaningless -- it only proves the plumbing).

Run (the server must be up -- see ``run_server.sh``):
    PYTHONPATH=<repo-root> python -m sparx_agency.tasks.planning.vlas.flownav.serve.smoke_test
    python -m ...smoke_test --obs /tmp/xtend_frames/frame_000123.jpg --goal target.jpg
"""
from __future__ import annotations

import argparse

import numpy as np

from sparx_agency.core.planning.vlas.flownav.client import FlowNavImageGoalClient


def _load_rgb(path):
    """Load an image file as an HxWx3 uint8 RGB array (raises on a bad path)."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8889")
    ap.add_argument("--obs", default=None, help="current-view image (default: synthetic)")
    ap.add_argument("--goal", default=None, help="target-view image (default: synthetic)")
    ap.add_argument("--steps", type=int, default=1,
                    help="repeat the step this many times (fills the context buffer)")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    obs = _load_rgb(args.obs) if args.obs else rng.randint(0, 255, (294, 504, 3), np.uint8)
    goal = _load_rgb(args.goal) if args.goal else rng.randint(0, 255, (294, 504, 3), np.uint8)
    synthetic = args.obs is None or args.goal is None

    client = FlowNavImageGoalClient(args.url, logger=lambda *a: print("[client]", *a))
    result = None
    for _ in range(max(1, args.steps)):
        result = client.step(obs, goal)
    if result is None:
        raise SystemExit("[fail] no response from %s -- is the FlowNav server running?"
                         % args.url)

    wp = client.best_trajectory(result)
    tag = "SYNTHETIC " if synthetic else ""
    print("[ok] %swaypoints (forward,left): %s" % (tag, np.round(wp, 3).tolist()))
    print("[ok] distance: %.3f" % float(result.get("distance", 0.0)))
    if synthetic:
        print("[note] inputs were random noise -> the waypoints are meaningless; "
              "this only proves the client -> server -> engines path works.")


if __name__ == "__main__":
    main()
