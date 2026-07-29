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

Two different runs of failures stop the worker, and the distinction is worth
throughput. A run of outcomes the aircraft *could not fly*
(:data:`UNFLYABLE_OUTCOMES`) means it is lying against something and no later
episode can recover it, so three ends the worker. A run of *poor flights* --
diverged from the route, missed the goal, ran out of budget -- happened to an
aircraft that is still airborne and still recording, so it takes eight. Treating
those alike cost a Kit boot, three and a half minutes, every time a worker had a
bad patch it would have flown out of on its own.

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
    EpisodeSpec, plan_between, sample_episode,
)
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import FollowSpec

UNFLYABLE_OUTCOMES = (episode.OUTCOME_CRASHED, episode.OUTCOME_ARM_TIMEOUT)
"""Outcomes that say the aircraft cannot take off again without help.

A crash leaves it lying against something, PX4 then refuses to arm on attitude,
and nothing here can right it -- so a run of these is the wedge that must stop a
worker. Every *other* failure happened to an aircraft that flew: a route
diverged, a goal was missed, a flight ran out of budget. Those are bad flights,
not a broken worker, and recycling on three of them costs a three-and-a-half
minute Kit boot to fix a problem the next episode would have fixed for free.
"""

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
    ap.add_argument("--cruise-speed", type=float, default=None,
                    help="speed held along the route, m/s (default 1.2). PX4's own "
                         "ceiling is 2.0, so do not exceed that")
    ap.add_argument("--max-yaw-rate", type=float, default=None,
                    help="how fast the aircraft may rotate, deg/s (default 14). This "
                         "is how fast the world spins in the recorded imagery")
    ap.add_argument("--max-bad-flights", type=int, default=8,
                    help="stop after this many poor flights in a row that the "
                         "aircraft nonetheless survived (off_route, stalled, "
                         "flight_timeout, ...). Higher than "
                         "--max-consecutive-failures because none of them means "
                         "the worker is broken")
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


def _follow_spec(args) -> FollowSpec:
    """How the aircraft flies a route, from the command line."""
    defaults = FollowSpec()
    cruise = args.cruise_speed if args.cruise_speed is not None else defaults.cruise_speed
    yaw_rate = (np.radians(args.max_yaw_rate) if args.max_yaw_rate is not None
                else defaults.max_yaw_rate)
    return FollowSpec(cruise_speed=cruise, max_speed=max(cruise * 1.25, cruise + 0.2),
                      max_yaw_rate=yaw_rate)


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
    follow_spec = _follow_spec(args)

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
        if not campaign_setup.wait_until_armable(loop, px4):
            print("ERROR: this worker never became armable; nothing to collect",
                  flush=True)
            return 1
        records = run_campaign(args, spec, follow_spec, rng, world_map, loop, adapter,
                               px4, chase_camera, resolution, seed)
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


def run_campaign(args, spec, follow_spec, rng, world_map, loop, adapter, px4,
                 chase_camera, resolution, seed) -> list:
    """Fly ``args.episodes`` missions, recording each. Returns the manifest rows."""
    records = []
    consecutive_unflyable = 0   # the aircraft is wedged and cannot take off
    consecutive_failures = 0    # it flies, but nothing is coming out well
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
            if consecutive_failures >= args.max_bad_flights:
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
        result = episode.fly_episode(loop, px4, adapter, plan, recorder, follow_spec,
                                     replan=_replanner(world_map, spec, plan.goal))
        flown = result.flown_plan or plan
        stats = recorder.finish(_episode_meta(args, seed, index, name, flown, result,
                                              resolution, follow_spec))
        episode.settle_after_landing(loop, px4)

        records.append(_manifest_row(name, flown, result, stats))
        print(f"episode {index}: {result.outcome} -- {result.frames} frames, "
              f"{result.duration_s:.0f} s, {result.goal_error_m:.2f} m from the goal"
              + (f" [{result.detail}]" if result.detail else ""), flush=True)

        if result.ok:
            consecutive_failures = 0
            consecutive_unflyable = 0
            start_from = plan.goal
        else:
            consecutive_failures += 1
            start_from = None  # re-derive from wherever it actually ended up
            if result.outcome in UNFLYABLE_OUTCOMES:
                consecutive_unflyable += 1
            else:
                # It flew, so whatever went wrong, it is not lying on its side.
                consecutive_unflyable = 0
            for message in result.px4_messages[-4:]:
                print(f"    PX4: {message}", flush=True)
            if consecutive_unflyable >= args.max_consecutive_failures:
                print(f"stopping: {consecutive_unflyable} episodes in a row the "
                      f"aircraft could not fly -- it is most likely wedged", flush=True)
                break
            if consecutive_failures >= args.max_bad_flights:
                print(f"stopping: {consecutive_failures} poor flights in a row -- "
                      f"the aircraft flies but nothing usable is coming out", flush=True)
                break
    return records


def _replanner(world_map, spec, goal):
    """A callable that plans a fresh route from a live pose to ``goal``.

    Handed to :func:`episode.fly_episode`, which uses it twice: once when the
    aircraft reaches cruise altitude -- so the route starts where the aircraft
    actually is rather than where it stood before the climb moved it -- and
    again whenever the follower reports it has diverged.

    Returns ``None`` when no route exists from that pose, which the caller reads
    as "keep flying what you have" on the first call and "give up on this
    episode" on a later one.
    """
    def replan(x: float, y: float, yaw: float):
        start = world_map.snap(float(x), float(y), yaw=float(yaw))
        return plan_between(world_map.grid, start, goal, spec,
                            planner=world_map.planner)

    return replan


def _episode_meta(args, seed, index, name, plan, result, resolution,
                  follow_spec) -> dict:
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
        # How many times the route was planned again mid-flight. The waypoints
        # above are the LAST route flown, so a non-zero count means the earlier
        # part of this recording was flown on a different one.
        "replans": result.replans,
        "outcome": result.outcome,
        "outcome_ok": result.ok,
        "outcome_detail": result.detail,
        "goal_error_m": result.goal_error_m,
        "route_remaining_m": result.route_remaining_m,
        "cross_track_error_m": result.cross_track_error_m,
        "estimator_drift_m": result.estimator_drift_m,
        "cruise_speed_mps": follow_spec.cruise_speed,
        "max_yaw_rate_dps": float(np.degrees(follow_spec.max_yaw_rate)),
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
        "route_remaining_m": result.route_remaining_m,
        "cross_track_error_m": result.cross_track_error_m,
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
