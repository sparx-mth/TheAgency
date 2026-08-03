"""Duplicate a scene's own obstacles into more places, for denser VLA training data.

The stock Isaac Sim indoor scenes are sparse -- ``warehouse`` especially, see
``scene.py``'s ``INDOOR_SCENES`` comment -- which under-trains obstacle
avoidance. This copies obstacle-sized prims *already in the loaded scene* to
new positions chosen against the scene's own surveyed map, so the extra
clutter matches the existing style and never overlaps a wall, the spawn point,
or another duplicate.

Requires the scene to already have a surveyed 2D map (``survey_scene.py``) at
the given altitude -- that map is both where obstacles must avoid landing and
the source of the "landable" floor they may be placed on.

Writes a duplication recipe (default
``robots/PEGASUS/scenes/<scene>_augmented.json``) that
:func:`~sparx_agency.robots.PEGASUS.adapters.scene._register_local_scenes`
picks up automatically as an ``AUGMENTED_SCENES`` entry the next time any
script imports ``scene.py`` -- no manual registration needed. It inherits the
base scene's spawn point (the building layout did not change), so it flies
exactly like the original except for the new clutter. See
``scene.load_augmented_scene`` for why this is a *recipe* replayed against a
freshly loaded base scene rather than a second baked ``.usd`` file.

**The 3D voxel map and 2D planning maps are now stale for the augmented
scene** -- they describe the geometry before this ran. Re-survey before
collecting data against it::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py \\
        --scene office_augmented --altitude 1.5 --preview

Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/augment_scene.py \\
        --scene office --count 25 --min-spacing 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# How far a new obstacle must stay from the scene's spawn point, so a dense
# scene never blocks the one spot a campaign is guaranteed to take off from.
DEFAULT_SPAWN_KEEPOUT_M = 1.5


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="a key of scene.INDOOR_SCENES")
    ap.add_argument("--altitude", type=float, default=1.5,
                    help="altitude the scene's 2D map was surveyed at, metres")
    ap.add_argument("--count", type=int, default=20,
                    help="how many duplicated obstacles to add")
    ap.add_argument("--min-spacing", type=float, default=1.2,
                    help="minimum centre-to-centre distance between new "
                         "obstacles, metres")
    ap.add_argument("--spawn-keepout", type=float, default=DEFAULT_SPAWN_KEEPOUT_M,
                    help="radius around the scene's spawn point that stays clear, metres")
    ap.add_argument("--root", default="/World/Scene",
                    help="stage path the scene was referenced at")
    ap.add_argument("--min-height", type=float, default=0.0,
                    help="keep only obstacles whose top reaches at least this "
                         "high above the floor, metres -- 0 (default) keeps "
                         "everything obstacle-sized regardless of orientation; "
                         "set this near the flight altitude to bias toward "
                         "columns/partitions/racks a drone cruising there "
                         "would actually hit, over low desk clutter")
    ap.add_argument("--stretch-top", type=float, default=None,
                    help="stretch every duplicate that doesn't already reach "
                         "this height above the floor, metres -- a drone flying "
                         "*inside* a band it merely brushed can still slip over "
                         "or under it; combine with --min-height set to the "
                         "bottom of the flight band so the copy is guaranteed "
                         "to span the whole thing, e.g. --min-height 0.5 "
                         "--stretch-top 1.5 for a 0.5-1.5 m flight band")
    ap.add_argument("--output", type=Path, default=None,
                    help="destination .json recipe path (default: "
                         "robots/PEGASUS/scenes/<scene>_augmented.json)")
    ap.add_argument("--map-dir", type=Path, default=None,
                    help="override where the surveyed maps are read from")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    from sparx_agency.tasks.planning.sim_flight_recording import flight_session

    simulation_app = flight_session.boot_isaac(stream=False)
    try:
        import numpy as np

        from sparx_agency.robots.PEGASUS.adapters import scene_augment
        from sparx_agency.robots.PEGASUS.adapters.obstacle_placement import sample_placements
        from sparx_agency.robots.PEGASUS.adapters.scene import (
            LOCAL_SCENES_DIR, scene_spawn,
        )
        from sparx_agency.robots.PEGASUS.adapters.scene_map import LANDABLE_LAYER, load_scene_map

        world = flight_session.build_world(simulation_app, args.scene)
        world.reset()
        for _ in range(flight_session.STAGE_SETTLE_STEPS):
            world.step(render=False)

        grid, _metadata, layers = load_scene_map(args.scene, args.altitude, args.map_dir)
        spawn_x, spawn_y, _z = scene_spawn(args.scene)

        placements = sample_placements(
            grid, layers[LANDABLE_LAYER], count=args.count,
            min_spacing_m=args.min_spacing,
            keepout=[(spawn_x, spawn_y, args.spawn_keepout)],
            rng=np.random.default_rng(),
        )
        if len(placements) < args.count:
            print(f"WARNING: only {len(placements)}/{args.count} placements fit "
                  f"at {args.min_spacing:.1f} m spacing -- the free landable area "
                  f"is smaller than requested", flush=True)

        candidates = scene_augment.list_obstacle_prims(
            root=args.root, min_reach_height_m=args.min_height)
        if not candidates:
            raise RuntimeError(
                f"no duplicatable obstacle prims found under {args.root!r} in "
                f"scene {args.scene!r} reaching at least {args.min_height:.2f} m"
            )
        print(f"found {len(candidates)} candidate obstacle prims under {args.root}"
              + (f" reaching >= {args.min_height:.2f} m" if args.min_height > 0 else ""),
              flush=True)

        placed = scene_augment.augment_with_duplicates(
            placements, candidates, stretch_top_m=args.stretch_top)
        print(f"AUGMENT {args.scene}: added {len(placed)} obstacles", flush=True)
        for p in placed:
            print(f"  {p['dest_path']} <- {p['source']} at "
                  f"({p['x']:.2f}, {p['y']:.2f}), yaw {p['rotation_deg']:.0f} deg",
                  flush=True)

        root_prefix = args.root
        recipe = {
            "base_scene": args.scene,
            "stretch_top_m": args.stretch_top,
            "obstacles": [
                {
                    "source_rel": p["source"][len(root_prefix):],
                    "dx": p["dx"], "dy": p["dy"], "dz": p["dz"],
                    "rotation_deg": p["rotation_deg"],
                    "target_x": p["x"], "target_y": p["y"],
                }
                for p in placed
            ],
        }

        output_path = args.output or (LOCAL_SCENES_DIR / f"{args.scene}_augmented.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(recipe, indent=2))
        print(f"wrote augmentation recipe to {output_path}", flush=True)
        print(f"re-survey before flying against it:\n"
              f"  /isaac-sim/python.sh "
              f"sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py "
              f"--scene {output_path.stem} --altitude {args.altitude} --preview",
              flush=True)
    except Exception:
        # Kit's fast shutdown in the finally below tears the process down
        # without letting the traceback reach a terminal -- print it first.
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
