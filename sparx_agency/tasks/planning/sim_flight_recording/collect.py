"""Fly many autonomous A-to-B missions in one scene and record every one.

The entry point for a data-collection worker. One process boots Isaac Sim once,
loads a surveyed map, and then flies episode after episode without ever
returning to the simulator's start-up cost -- which is the whole point, since
booting Kit takes minutes and a flight takes seconds.

Each episode is: sample a reachable goal, plan a wall-avoiding route to it, arm,
take off, track the route, land there. The next episode starts from wherever the
last one landed, so a worker walks the building and every start point after the
first is itself a previously-drawn random point. Together with a random spawn
per worker, that is where the variety in a campaign comes from.

**Failures are expected and handled, not fatal.** An episode that hits something
or refuses to arm is written out with its outcome recorded in ``meta.json``, the
airframe is settled, and the next episode is planned from wherever it ended up.
Only a run of consecutive failures stops the worker -- at that point the aircraft
is genuinely wedged and more attempts would only produce more bad data.

Run several of these at once for throughput; ``run_collection.sh`` does that,
and :mod:`px4_launch` documents the per-instance ports and directories that make
it safe. Must run under Isaac Sim's own Python::

    /isaac-sim/python.sh sparx_agency/tasks/planning/sim_flight_recording/collect.py \\
        --scene office --episodes 5 --out-dir /tmp/dev/recordings/office
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from sparx_agency.tasks.planning.sim_flight_recording import (
    campaign_setup, episode, flight_session, px4_launch, px4_params,
)
from sparx_agency.tasks.planning.sim_flight_recording.episode_plan import (
    EpisodeSpec, sample_episode,
)

HEARTBEAT_TIMEOUT_S = 120.0
"""Simulated seconds to wait for PX4's first heartbeat. It normally takes ~1.5."""

PARAM_SETTLE_S = 3.0
"""Simulated seconds to let PX4 apply and acknowledge a parameter push."""


def _parse_args():
    from sparx_agency.robots.PEGASUS.adapters.vehicle import AIRFRAME_RADIUS_M

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="office", help="a key of scene.INDOOR_SCENES")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="directory to write recordings and the campaign manifest into")
    ap.add_argument("--episodes", type=int, default=5, help="flights to record")
    ap.add_argument("--pegasus-root", type=Path,
                    default=Path("/tmp/dev/PegasusSimulator/extensions/pegasus.simulator"))
    ap.add_argument("--px4-dir", type=Path, default=Path("/tmp/dev/PX4-Autopilot"))
    ap.add_argument("--map-dir", type=Path, default=None,
                    help="override where surveyed maps are read from")

    ap.add_argument("--worker", type=int, default=0,
                    help="worker index. Also the PX4 instance id, so every "
                         "concurrent worker MUST have a distinct one (0-9)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed. Defaults to the worker index, so concurrent "
                         "workers explore different parts of the building")

    ap.add_argument("--altitude", type=float, default=1.5,
                    help="cruise altitude, metres. Must match a surveyed map")
    ap.add_argument("--resolution", default=None,
                    help="camera resolution as WxH (e.g. 640x480). Defaults to the "
                         "platform calibration's own 504x392")
    ap.add_argument("--rate-hz", type=float, default=10.0, help="frame capture rate")
    ap.add_argument("--depth-format", choices=["png", "npy"], default="png",
                    help="png = uint16 millimetres (~4x smaller); npy = float32 metres")

    ap.add_argument("--min-distance", type=float, default=5.0,
                    help="shortest mission worth recording, metres")
    ap.add_argument("--max-distance", type=float, default=None,
                    help="longest mission, metres")
    ap.add_argument("--clearance", type=float, default=0.8,
                    help="obstacle clearance required at the start and goal, metres")
    ap.add_argument("--standoff", type=float, default=AIRFRAME_RADIUS_M + 0.25,
                    help="obstacle standoff the route is planned at, metres")
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="stop the worker after this many failed episodes in a row")
    ap.add_argument("--settle-s", type=float, default=30.0,
                    help="simulated seconds to sit still before the first flight, so "
                         "PX4's estimator converges. Cheap, and the first episode is "
                         "materially worse without it")

    ap.add_argument("--realtime", action="store_true",
                    help="throttle to wall-clock time so a live viewer can follow")
    ap.add_argument("--stream", action="store_true",
                    help="enable the WebRTC livestream on port 49100. Only one "
                         "process can bind it, so never with several workers")
    ap.add_argument("--video", action="store_true",
                    help="write a chase-camera MP4 next to each recording")
    return ap.parse_args()


def _episode_spec(args) -> EpisodeSpec:
    from sparx_agency.robots.PEGASUS.adapters.vehicle import AIRFRAME_RADIUS_M

    return EpisodeSpec(
        altitude_m=args.altitude,
        clearance_m=args.clearance,
        inflate_radius_m=args.standoff,
        inflate_floor_m=AIRFRAME_RADIUS_M,
        min_separation_m=args.min_distance,
        max_separation_m=args.max_distance,
    )


def main() -> int:
    args = _parse_args()
    from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import (
        parse_resolution,
    )

    resolution = parse_resolution(args.resolution) if args.resolution else None
    seed = args.seed if args.seed is not None else args.worker
    rng = np.random.default_rng(seed)
    spec = _episode_spec(args)

    # Load and validate the map before paying for a Kit boot: a missing or
    # unusable map is the most common way a campaign is misconfigured, and it
    # costs minutes to discover after the simulator has started.
    world_map = campaign_setup.load_map(args.scene, args.altitude, spec, args.map_dir)
    print(f"map: {world_map.summary}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    px4_launch.clear_saved_parameters(args.px4_dir, args.worker)
    px4_process = px4_launch.launch_px4(
        args.px4_dir, instance=args.worker,
        log_path=args.out_dir / f"px4_worker{args.worker}.log",
    )
    simulation_app = flight_session.boot_isaac(stream=args.stream)
    records = []
    try:
        spawn = world_map.random_pose(rng)
        print(f"spawning at ({spawn.x:.2f}, {spawn.y:.2f}) facing "
              f"{np.degrees(spawn.yaw):.0f} deg", flush=True)

        loop, adapter, px4, chase_camera = campaign_setup.bring_up(
            simulation_app, args, spawn, heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S,
        )
        campaign_setup.configure_px4(loop, px4, px4_params.all_params(), PARAM_SETTLE_S)
        campaign_setup.settle_estimator(loop, px4, args.settle_s)
        records = run_campaign(args, spec, rng, world_map, loop, adapter, px4,
                               chase_camera, resolution, seed)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        _write_manifest(args, seed, records)
        px4_launch.terminate_px4(px4_process, instance=args.worker)
        simulation_app.close()

    landed = sum(1 for r in records if r["outcome"] == episode.OUTCOME_LANDED)
    print(f"CAMPAIGN_DONE worker={args.worker} scene={args.scene} "
          f"episodes={len(records)} landed={landed}", flush=True)
    return 0 if landed else 1


def run_campaign(args, spec, rng, world_map, loop, adapter, px4, chase_camera,
                 resolution, seed) -> list:
    """Fly ``args.episodes`` missions, recording each. Returns the manifest rows."""
    records = []
    consecutive_failures = 0
    start_from = None  # the first episode is planned from wherever the aircraft spawned

    for index in range(args.episodes):
        position = adapter.vehicle.state.position
        start_hint = world_map.snap(float(position[0]), float(position[1]),
                                    yaw=adapter.yaw()) if start_from is None else start_from
        try:
            plan = sample_episode(world_map.grid, rng, spec, region=world_map.region,
                                  goal_region=world_map.landing_region,
                                  start_from=start_hint, planner=world_map.planner)
        except RuntimeError as error:
            print(f"episode {index}: could not plan -- {error}", flush=True)
            consecutive_failures += 1
            if consecutive_failures >= args.max_consecutive_failures:
                break
            continue

        name = f"{args.scene}_w{args.worker}_e{index:03d}"
        out_dir = args.out_dir / name
        print(f"episode {index}: ({plan.start.x:.1f}, {plan.start.y:.1f}) -> "
              f"({plan.goal.x:.1f}, {plan.goal.y:.1f}), "
              f"{plan.path_length_m:.1f} m over {len(plan.waypoints)} waypoints "
              f"(detour {plan.detour_ratio:.2f}x)", flush=True)

        recorder = flight_session.FlightRecorder(
            adapter, out_dir, rate_hz=args.rate_hz, camera_height_m=args.altitude,
            depth_format=args.depth_format,
            video_out=(out_dir.with_suffix(".mp4") if args.video else None),
            video_source="chase", chase_camera=chase_camera,
        )
        result = episode.fly_episode(loop, px4, adapter, plan, recorder)
        stats = recorder.finish(_episode_meta(args, seed, index, name, plan, result,
                                              resolution))
        episode.settle_after_landing(loop, px4)

        records.append(_manifest_row(name, plan, result, stats))
        print(f"episode {index}: {result.outcome} -- {result.frames} frames, "
              f"{result.duration_s:.0f} s, {result.goal_error_m:.2f} m from the goal"
              + (f" [{result.detail}]" if result.detail else ""), flush=True)

        if result.ok:
            consecutive_failures = 0
            start_from = plan.goal
        else:
            consecutive_failures += 1
            start_from = None  # re-derive from wherever it actually ended up
            for message in result.px4_messages[-4:]:
                print(f"    PX4: {message}", flush=True)
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"stopping: {consecutive_failures} failed episodes in a row -- "
                      f"the aircraft is most likely wedged", flush=True)
                break
    return records


def _episode_meta(args, seed, index, name, plan, result, resolution) -> dict:
    """Provenance written into each recording's ``meta.json``."""
    return {
        "recording": name,
        "scene": args.scene,
        "worker": args.worker,
        "seed": seed,
        "episode": index,
        "altitude_m": args.altitude,
        "camera_resolution": list(resolution) if resolution else None,
        "start_xy": [plan.start.x, plan.start.y],
        "start_yaw": plan.start.yaw,
        "goal_xy": [plan.goal.x, plan.goal.y],
        "planned_path_length_m": plan.path_length_m,
        "planned_waypoints": [list(w) for w in plan.waypoints],
        "planner_standoff_m": plan.inflate_used_m,
        "detour_ratio": plan.detour_ratio,
        "outcome": result.outcome,
        "outcome_ok": result.ok,
        "outcome_detail": result.detail,
        "goal_error_m": result.goal_error_m,
        "waypoints_reached": result.waypoints_reached,
        "waypoints_skipped": result.waypoints_skipped,
        "estimator_drift_m": result.estimator_drift_m,
    }


def _manifest_row(name, plan, result, stats) -> dict:
    """One line of the campaign manifest."""
    return {
        "recording": name,
        "outcome": result.outcome,
        "ok": result.ok,
        "frames": stats["frames"],
        "duration_s": stats["duration_s"],
        "flown_path_length_m": stats["path_length_m"],
        "planned_path_length_m": plan.path_length_m,
        "start_xy": [plan.start.x, plan.start.y],
        "goal_xy": [plan.goal.x, plan.goal.y],
        "goal_error_m": result.goal_error_m,
        "waypoints_skipped": result.waypoints_skipped,
        "estimator_drift_m": result.estimator_drift_m,
        "detail": result.detail,
    }


def _write_manifest(args, seed, records) -> None:
    """Write the campaign summary, even if the run died part way through."""
    landed = [r for r in records if r["ok"]]
    manifest = {
        "scene": args.scene,
        "worker": args.worker,
        "seed": seed,
        "altitude_m": args.altitude,
        "rate_hz": args.rate_hz,
        "requested_episodes": args.episodes,
        "recorded_episodes": len(records),
        "successful_episodes": len(landed),
        "total_frames": sum(r["frames"] for r in records),
        "total_flown_m": sum(r["flown_path_length_m"] for r in records),
        "episodes": records,
    }
    path = args.out_dir / f"campaign_w{args.worker}.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote campaign manifest to {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
