"""Fly the policy. The same missions, once untrained and once fine-tuned.

Runs **inside the isaac-sim container**::

    docker exec isaac-sim /isaac-sim/python.sh \
        /tmp/dev/repo/sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/fly_navdp.py \
        --scene office --missions 6 --seed 4242 --arm trained \
        --server http://127.0.0.1:8888 --out /tmp/dev/navdp_flights --video

Everything offline is open-loop: one prediction, from one frame, scored on its
own. That cannot see the failure that actually matters, which is a small bias
compounding over a hundred inferences until the aircraft is against a wall. This
can, because the policy's output becomes the aircraft's motion and the next
observation.

The policy runs on the **host**, not here. Isaac's Python has torch but not
``diffusers``, and NavDP's scheduler needs it; rather than mutate the container,
inference is served over the HTTP contract this repo already uses for NavDP
(``serve/navdp_trt_server.py``) and reached with the client already in
``core`` -- the same contract the FALCON nodes speak, so what flies here is
exactly what would fly on the real aircraft.

    # host, navdp env, one per arm (different ports):
    python -m ...navdp.serve.navdp_trt_server --backend torch --port 8888 \
        --ckpt ~/Downloads/navdp-cross-modal.ckpt            # baseline
    python -m ...navdp.serve.navdp_trt_server --backend torch --port 8889 \
        --ckpt ~/navdp_world_goal/navdp-world-goal.ckpt      # fine-tuned

Both arms fly the **same missions from the same seed**, and every mission is
scored against the surveyed map: how close the aircraft really came to
geometry, whether it reached the goal, how far it flew to get there. That is a
different and much harder question than "was the predicted trajectory clear",
and it is the one that decides whether the fine-tune was worth it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CRUISE_SPEED_MPS = 1.0
LOOKAHEAD_M = 1.2
INFER_HZ = 3.0
GOAL_TOLERANCE_M = 1.5
MAX_YAW_RATE_DPS = 60.0
STALL_DISTANCE_M = 0.6
STALL_WINDOW_S = 20.0
SECONDS_PER_METRE = 5.0
FIXED_OVERHEAD_S = 60.0
COLLISION_CLEARANCE_M = 0.05     # airframe centre this close to geometry = a hit


def carrot(trajectory: np.ndarray, lookahead_m: float) -> Tuple[float, float]:
    """The point ``lookahead_m`` along a body-frame trajectory.

    NavDP returns 24 waypoints covering a few metres; chasing the last one cuts
    every corner it just planned, and chasing the first one crawls. A carrot at
    a fixed arc length is the standard middle, and it is what every other
    follower in this repo does.
    """
    path = np.asarray(trajectory, dtype=np.float64)[:, :2]
    steps = np.linalg.norm(np.diff(path, axis=0, prepend=np.zeros((1, 2))), axis=1)
    along = np.cumsum(steps)
    if along[-1] <= 1e-6:
        return 0.0, 0.0
    target = min(lookahead_m, float(along[-1]))
    return (float(np.interp(target, along, path[:, 0])),
            float(np.interp(target, along, path[:, 1])))


def velocity_to(position, target_xy, current_yaw: float, altitude_m: float,
                cruise_mps: float) -> Tuple[float, float, float]:
    """World-frame velocity along the current heading, speed capped at cruise.

    Points the velocity vector at ``current_yaw``, not straight at
    ``target_xy``: a holonomic quad that flies wherever the target is
    regardless of where it's facing crabs sideways through every turn
    instead of yawing to face it first. Scaling speed by how well
    ``current_yaw`` already matches the bearing to the target (``cos`` of
    the heading error, clamped so it can't run in reverse) slows a sharp
    turn to a yaw-in-place and lets it pick speed back up smoothly once
    aligned -- see the identical fix in
    ``core/planning/trackers/pure_pursuit/algorithm.compute_velocity_3d``,
    which this mirrors for the same reason.
    """
    from sparx_agency.core.common.types.geometry import normalize_angle

    dx = float(target_xy[0]) - float(position[0])
    dy = float(target_xy[1]) - float(position[1])
    distance = math.hypot(dx, dy)
    vz = 0.8 * (altitude_m - float(position[2]))
    if distance < 1e-3:
        return 0.0, 0.0, vz
    heading_error = normalize_angle(math.atan2(dy, dx) - current_yaw)
    speed = min(cruise_mps, 1.2 * distance) * max(0.0, math.cos(heading_error))
    return speed * math.cos(current_yaw), speed * math.sin(current_yaw), vz


def sample_missions(world_map, seed: int, count: int, spec) -> List:
    """Draw ``count`` missions. Same seed, same missions, for every arm."""
    from sparx_agency.core.planning.mission import sample_start_goal

    rng = np.random.default_rng(seed)
    missions = []
    for _ in range(count):
        missions.append(sample_start_goal(
            world_map.grid, rng, clearance_m=spec.clearance_m,
            min_separation_m=spec.min_separation_m,
            max_separation_m=spec.max_separation_m,
            start_yaw_jitter_rad=spec.start_yaw_jitter_rad,
            region=world_map.landing_region))
    return missions


def _empty_result(mission, start, outcome: str, detail: str = "") -> Dict:
    """A result for a mission that never flew, with every key the caller reads.

    The summary averages ``min_clear_m``, ``path_len_m``, ``duration_s`` and
    ``goal_error_m`` across missions and counts ``collided``. A short dict here
    -- which is what an ``arm_timeout`` used to return, and a cold PX4 makes
    that the common case -- raises ``KeyError`` in the per-mission print, and
    since ``summary.json`` is only written after the loop, every mission already
    flown is lost with it. NaN propagates through ``np.mean`` visibly; a missing
    key does not.
    """
    return {
        "outcome": outcome,
        "detail": detail,
        "reached": False,
        "start_xy": [float(mission.start.x), float(mission.start.y)],
        "goal_xy": [float(mission.goal.x), float(mission.goal.y)],
        "separation_m": float(mission.separation_m),
        "goal_error_m": float("nan"),
        "min_clear_m": float("nan"),
        "p5_clear_m": float("nan"),
        "mean_clear_m": float("nan"),
        "collided": False,
        "path_len_m": 0.0,
        "duration_s": 0.0,
        "inferences": 0,
        "transport_failures": 0,
    }


def fly_mission(loop, px4, adapter, client, scene, mission, args,
                recorder=None, track_log=None) -> Dict:
    """Fly one A-to-B mission with the policy closing the loop.

    Args:
        track_log: Optional :class:`~.track_log.TrackLog`. Given one, every
            inference's proposed trajectory is recorded in world coordinates,
            which is what a map panel is drawn from afterwards. The flight is
            identical either way -- this only observes.

    Returns:
        A result dict: outcome, whether the goal was reached, the worst
        clearance the aircraft actually achieved against the surveyed map,
        collisions, path length and duration.
    """
    from sparx_agency.core.planning.vlas.navdp.geometry import (
        point_to_pointgoal, world_to_body_2d,
    )
    from sparx_agency.tasks.planning.sim_flight_recording.episode import (
        CRASH_HOLD_S, CRASH_TILT_DEG, TAKEOFF_TOLERANCE_M, arm_for_offboard,
        attitude_deg, slew_towards,
    )

    vehicle = adapter.vehicle
    goal = (mission.goal.x, mission.goal.y)
    altitude = args.altitude

    start = vehicle.state.position
    failure = arm_for_offboard(loop, px4, start, mission.start.yaw, altitude)
    if failure is not None:
        return _empty_result(mission, start, "arm_timeout", detail=failure)

    # The client builds the K matrix itself from an Intrinsics; handing it the
    # matrix instead fails inside reset with 'list' object has no attribute 'fx'.
    client.reset(adapter.intrinsics, stop_threshold=-999, batch_size=1)

    budget = FIXED_OVERHEAD_S + SECONDS_PER_METRE * mission.separation_m
    deadline = loop.sim_time + budget
    infer_period = 1.0 / max(args.infer_hz, 0.1)
    next_infer = 0.0
    target_world = (float(start[0]), float(start[1]))
    yaw_command = mission.start.yaw
    track: List[Tuple[float, float]] = []
    tilted_since: Optional[float] = None
    stall_reference = (float(start[0]), float(start[1]), loop.sim_time)
    inferences, transport_failures = 0, 0
    outcome = "flight_timeout"
    # arm_for_offboard only arms and switches modes -- it does not climb, so the
    # aircraft is still on the ground when this function's loop starts. Running
    # NavDP inference immediately would have it steering while the airframe is
    # still climbing through ground effect and out of takeoff attitude, so
    # inference is held off until the aircraft is at cruise altitude; the climb
    # itself already happens for free via velocity_to()'s vz term below.
    climbed = False

    while loop.sim_time < deadline:
        rendered = loop.step()
        position = vehicle.state.position
        pose = adapter.pose_flu()
        track.append((float(position[0]), float(position[1])))

        if not climbed and position[2] >= altitude - TAKEOFF_TOLERANCE_M:
            climbed = True
            # Don't charge the climb time against the first inference's period,
            # and don't let the stall check fire for time spent only climbing.
            next_infer = loop.sim_time
            stall_reference = (float(position[0]), float(position[1]), loop.sim_time)

        if rendered and recorder is not None:
            recorder.capture(stamp_s=loop.sim_time)

        if climbed and rendered and loop.sim_time >= next_infer:
            next_infer = loop.sim_time + infer_period
            frame = adapter.capture_frame(stamp_s=loop.sim_time)
            forward, left = world_to_body_2d(goal[0], goal[1], pose[0], pose[1], pose[2])
            gx, gy = point_to_pointgoal(forward, left)
            result = client.pointgoal_step(
                np.ascontiguousarray(frame.rgb[:, :, :3]), frame.depth, gx, gy,
                altitude=altitude)
            inferences += 1
            trajectory = None
            if result is None:
                transport_failures += 1
            else:
                trajectory = client.best_trajectory(result)
                body = carrot(trajectory, args.lookahead)
                cos, sin = math.cos(pose[2]), math.sin(pose[2])
                target_world = (pose[0] + body[0] * cos - body[1] * sin,
                                pose[1] + body[0] * sin + body[1] * cos)
                if math.hypot(body[0], body[1]) > 0.15:
                    yaw_command = math.atan2(target_world[1] - pose[1],
                                             target_world[0] - pose[0])
            if track_log is not None:
                track_log.add(loop.sim_time, pose, trajectory, target_world)

        vx, vy, vz = velocity_to(position, target_world, pose[2], altitude, args.cruise)
        yaw_command = slew_towards(pose[2], yaw_command,
                                   math.radians(MAX_YAW_RATE_DPS), loop.dt)
        px4.send_velocity_world(vx, vy, vz, yaw_command)

        if math.hypot(position[0] - goal[0], position[1] - goal[1]) < args.goal_tolerance:
            outcome = "reached"
            break

        roll, pitch, _ = attitude_deg(vehicle)
        if max(abs(roll), abs(pitch)) > CRASH_TILT_DEG:
            tilted_since = tilted_since or loop.sim_time
            if loop.sim_time - tilted_since > CRASH_HOLD_S:
                outcome = "crashed"
                break
        else:
            tilted_since = None

        moved = math.hypot(position[0] - stall_reference[0], position[1] - stall_reference[1])
        if moved > STALL_DISTANCE_M:
            stall_reference = (float(position[0]), float(position[1]), loop.sim_time)
        elif loop.sim_time - stall_reference[2] > STALL_WINDOW_S:
            outcome = "stalled"
            break

    flown = np.asarray(track, dtype=np.float64)
    clearance = scene.clearance(flown[:, 0], flown[:, 1]) if flown.shape[0] else np.array([0.0])
    steps = np.linalg.norm(np.diff(flown, axis=0), axis=1) if flown.shape[0] > 1 else np.array([0.0])
    final = flown[-1] if flown.shape[0] else np.array(start[:2])
    if track_log is not None:
        track_log.set_flown(track)
    return {
        "outcome": outcome,
        "reached": outcome == "reached",
        "start_xy": [float(mission.start.x), float(mission.start.y)],
        "goal_xy": [float(goal[0]), float(goal[1])],
        "separation_m": float(mission.separation_m),
        "goal_error_m": float(math.hypot(final[0] - goal[0], final[1] - goal[1])),
        "min_clear_m": float(clearance.min()),
        "p5_clear_m": float(np.percentile(clearance, 5)),
        "mean_clear_m": float(clearance.mean()),
        "collided": bool(clearance.min() < COLLISION_CLEARANCE_M),
        "path_len_m": float(steps.sum()),
        "duration_s": float(len(track) * loop.dt),
        "inferences": inferences,
        "transport_failures": transport_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="office")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--missions", type=int, default=6,
                        help="size of the mission set the seed draws; with "
                             "--mission-index only one of them is flown")
    parser.add_argument("--mission-index", type=int, default=-1,
                        help="fly only this mission of the set, in a session of "
                             "its own. This is the correct way to fly more than "
                             "one: the aircraft cannot be repositioned between "
                             "missions, so a single session can only fly the "
                             "first honestly. -1 flies them all in one session.")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--arm", required=True, help="label for this set of weights")
    parser.add_argument("--server", default="http://127.0.0.1:8888")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev-root", default="/tmp/dev")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--cruise", type=float, default=CRUISE_SPEED_MPS)
    parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_M)
    parser.add_argument("--infer-hz", type=float, default=INFER_HZ)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_TOLERANCE_M)
    parser.add_argument("--video", action="store_true")
    # campaign_setup.bring_up reads all of these off the namespace, so they have
    # to exist here even where this script has no reason to vary them. Defaults
    # mirror collect.py's.
    parser.add_argument("--resolution", default=None,
                        help="camera WxH; defaults to the platform's own 504x392")
    # type=Path, as collect.py has it: spawn_vehicle does path arithmetic on
    # this straight away, so a bare string dies with "unsupported operand
    # type(s) for /: 'str' and 'str'" only once Kit has finished booting.
    parser.add_argument("--pegasus-root", type=Path,
                        default=Path("/tmp/dev/PegasusSimulator/extensions/pegasus.simulator"))
    parser.add_argument("--px4-dir", type=Path, default=Path("/tmp/dev/PX4-Autopilot"))
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true",
                        help="throttle the simulation to wall-clock time")
    parser.add_argument("--stream", action="store_true",
                        help="WebRTC livestream on :49100")
    parser.add_argument("--settle-s", type=float, default=30.0,
                        help="simulated seconds to let PX4's estimator converge "
                             "before the first mission. Without it the first "
                             "mission of each arm flies on a half-converged EKF2 "
                             "and the two arms are not comparable")
    args = parser.parse_args()

    repo = Path(args.dev_root) / "repo"
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from sparx_agency.core.planning.vlas.navdp.client import NavDPPointgoalClient
    from sparx_agency.tasks.planning.sim_flight_recording import (
        campaign_setup, flight_session, px4_launch, px4_params,
    )
    from sparx_agency.tasks.planning.sim_flight_recording.episode_plan import EpisodeSpec
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
        Scene, SceneConfig,
    )
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.track_log import (
        TrackLog,
    )

    spec = EpisodeSpec(altitude_m=args.altitude)
    world_map = campaign_setup.load_map(args.scene, args.altitude, spec)
    scene = Scene.load(SceneConfig(scene=args.scene, altitude_m=args.altitude))
    missions = sample_missions(world_map, args.seed, args.missions, spec)
    print(f"[fly] arm={args.arm}  {len(missions)} missions from seed {args.seed}",
          flush=True)
    for index, mission in enumerate(missions):
        print(f"[fly]   {index}: ({mission.start.x:.1f}, {mission.start.y:.1f}) -> "
              f"({mission.goal.x:.1f}, {mission.goal.y:.1f})  "
              f"{mission.separation_m:.1f} m", flush=True)

    # One mission per process is the only sound shape here. A mission begins
    # wherever the aircraft already is -- there is no teleport in the vehicle
    # adapter and none in flight_session, because collect.py spawns a fresh
    # Isaac and PX4 for every episode and never needed one. Flying several in
    # one session therefore starts mission N+1 from wherever mission N ended,
    # and once one crashes the aircraft is on the ground for good: every later
    # mission returns in three seconds having flown 0.1 m. Sampling still uses
    # the full list so the seed picks the same missions for every arm.
    selected = args.mission_index
    if selected >= 0:
        if selected >= len(missions):
            raise ValueError(
                f"--mission-index {selected} is past the {len(missions)} "
                f"missions seed {args.seed} produces")
        missions = [missions[selected]]
        print(f"[fly] flying mission {selected} only", flush=True)

    out = Path(args.out).expanduser() / args.arm
    out.mkdir(parents=True, exist_ok=True)
    client = NavDPPointgoalClient(args.server)

    # Order matters and is collect.py's: clear the persisted parameter store
    # first (PX4 keeps every parameter it is ever sent, so one arm's settings
    # would otherwise leak into the next), start PX4, then boot Kit.
    px4_launch.clear_saved_parameters(args.px4_dir, args.worker)
    px4_process = px4_launch.launch_px4(
        args.px4_dir, instance=args.worker,
        log_path=out / f"px4_worker{args.worker}.log")
    simulation_app = flight_session.boot_isaac(stream=args.stream)

    results: List[Dict] = []
    try:
        loop, adapter, px4_link, chase = campaign_setup.bring_up(
            simulation_app, args, missions[0].start, heartbeat_timeout_s=300.0)
        campaign_setup.configure_px4(loop, px4_link, px4_params.all_params(), 3.0)
        campaign_setup.settle_estimator(loop, px4_link, args.settle_s)
        if not campaign_setup.wait_until_armable(loop, px4_link):
            print("[fly] PX4 never became armable; nothing to fly", flush=True)
            return

        for index, mission in enumerate(missions):
            # The true index in the seed's mission set, not the position in this
            # session's list. Everything named after a mission uses this -- the
            # recording, the MP4 and the stored result -- so that ten one-mission
            # sessions in the same directory do not all call themselves "00".
            mission_id = selected if selected >= 0 else index
            recorder = None
            if args.video:
                recorder = flight_session.FlightRecorder(
                    adapter, out / f"mission_{mission_id:02d}", rate_hz=args.rate_hz,
                    camera_height_m=args.altitude,
                    video_out=out / f"mission_{mission_id:02d}.mp4",
                    video_source="chase", chase_camera=chase)
            # Logged whenever video is, because the map panel is drawn from it
            # and a video without the panel is half the comparison.
            log = TrackLog((mission.goal.x, mission.goal.y),
                           (mission.start.x, mission.start.y)) if args.video else None
            started = time.time()
            result = fly_mission(loop, px4_link, adapter, client, scene, mission,
                                 args, recorder, track_log=log)
            result.update({"mission": mission_id, "arm": args.arm,
                           "wall_s": round(time.time() - started, 1)})
            if log is not None:
                log.write(out / f"mission_{mission_id:02d}_track.json",
                          extra={"scene": args.scene, "arm": args.arm,
                                 "infer_hz": args.infer_hz, **result})
            if recorder is not None:
                recorder.finish({"scene": args.scene, "arm": args.arm, **result})
            results.append(result)
            print(f"[fly] mission {result['mission']}: {result['outcome']}  "
                  f"min_clear {result['min_clear_m']:.2f} m  "
                  f"goal_error {result['goal_error_m']:.2f} m", flush=True)
            # Written every mission, not at the end: a run that dies on mission
            # five should not also lose the four that flew. One-mission runs get
            # their own file so that twenty sessions in the same directory
            # accumulate instead of overwriting each other.
            name = "results.json" if selected < 0 else f"results_{selected:02d}.json"
            (out / name).write_text(json.dumps(results, indent=2))
    except Exception:
        # Kit runs with --/app/fastShutdown=True, so ``simulation_app.close()``
        # in the finally below tears the process down before the interpreter
        # gets to print the traceback. Without this the run looks like a clean
        # "no missions completed" and the real error is never seen.
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        _write_summary(out, args.arm, results)
        # Leaving either alive costs the *other* arm its ports and its GPU
        # memory, and this comparison is two sequential runs.
        simulation_app.close()
        px4_launch.terminate_px4(px4_process, instance=args.worker)


def _write_summary(out: Path, arm: str, results: List[Dict]) -> None:
    """Aggregate whatever flew, even when the run ended early."""
    if not results:
        print(f"[fly] {arm}: no missions completed", flush=True)
        return
    reached = sum(1 for r in results if r["reached"])
    collided = sum(1 for r in results if r.get("collided"))
    mean = lambda key: float(np.nanmean([r.get(key, float("nan")) for r in results]))
    summary = {
        "arm": arm, "missions": len(results), "reached": reached,
        "collisions": collided,
        "min_clear_m": mean("min_clear_m"),
        "path_len_m": mean("path_len_m"),
        "duration_s": mean("duration_s"),
        "goal_error_m": mean("goal_error_m"),
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fly] {arm}: reached {reached}/{len(results)}, "
          f"{collided} collisions, mean min clearance "
          f"{summary['min_clear_m']:.2f} m", flush=True)


if __name__ == "__main__":
    main()
