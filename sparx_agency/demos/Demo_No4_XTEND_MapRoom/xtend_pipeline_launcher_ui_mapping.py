#!/usr/bin/env python3
"""XTEND pipeline launcher — RGBD mapping + the FALCON object mission.

Starts and stops every process the demo needs, each with its parameters on
screen rather than buried in a command string. Picking a command shows every
parameter it accepts -- read from the node, launch file or config that declares
it, with the documentation the author wrote there -- at the default a plain run
would use, with one click to put any of them back.

What it brings up:

    XTEND WebSocket -> frames -> depth (DA3 TRT) -> point cloud
      -> AprilTag localization -> TF -> octomap
      -> FALCON object mission (A*/NavDP + YOLO-World + visual servo)

The object mission is a two-terminal workflow, and the launcher keeps it that
way: item 12 starts the detector sidecar and holds its TensorRT engines up,
item 13 restarts the mission beside it in seconds without reloading them.

SAFETY: the AUTO button arms the drone and takes off. Everything else only
starts a process.

Run:
    python3 sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_pipeline_launcher_ui_mapping.py

The pieces live in the ``launcher`` package next to this file; the reusable
parameter machinery is ``sparx_agency.tasks.common.launch_params``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run as a plain script (the usual way) as well as with `python -m`: the repo
# root must be importable for `sparx_agency.` to resolve either way.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.app import main  # noqa: E402

if __name__ == "__main__":
    main()
