#!/usr/bin/env python3
"""Fly one FALCON exploration run on Isaac Sim. Runs inside the isaac-sim container.

This is the aircraft half of a run. The FALCON half
(``run_falcon_pegasus.sh``) must already be up: it binds the two localhost
sockets this connects to, and its planner needs the depth stream this sends
before it will plan anything.

    docker exec isaac-sim bash -c "cd /tmp/dev/repo && /isaac-sim/python.sh \\
      sparx_agency/tasks/planning/falcon_pegasus/isaac/run_exploration.py \\
      --run 3_open_plan --video"

What it does, in order: launch PX4 SITL, boot Isaac Sim, load the scene, spawn
the Iris with FALCON's reference camera, wait out PX4's warm-up, check the
camera extrinsics against Isaac's own, connect to FALCON, and fly
(:mod:`~.mission`).

Must run under Isaac Sim's own Python (``/isaac-sim/python.sh``) -- it needs
``omni``/``carb``/``pegasus.*``, which only exist inside a running Kit app.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparx_agency.core.common.types import KinematicLimits
from sparx_agency.core.control.trajectory_tracking import TrajectoryTrackerParams
from sparx_agency.core.planning.trackers.reference_tracker_3d import ReferenceTrackerParams
from sparx_agency.tasks.planning.falcon_pegasus.isaac import setup
from sparx_agency.tasks.planning.falcon_pegasus.isaac.falcon_client import FalconLink
from sparx_agency.tasks.planning.falcon_pegasus.isaac.mission import (
    CONTROL_ATTITUDE, CONTROL_MODES, ExplorationMission, MissionSpec,
)
from sparx_agency.tasks.planning.falcon_pegasus.link.socket_link import (
    DOWNLINK_PORT, UPLINK_PORT,
)
from sparx_agency.tasks.planning.sim_flight_recording import flight_session, px4_launch

DEFAULT_DEV_ROOT = Path("/tmp/dev")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="3_open_plan",
                        help="a runs/*.yaml, by name or path (e.g. 3_open_plan)")
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT,
                        help="where install.sh put PegasusSimulator and PX4-Autopilot")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the flight recording, video and result "
                             "(default: <dev-root>/falcon_pegasus/<run>)")
    parser.add_argument("--worker", type=int, default=0,
                        help="PX4 instance id; selects every port and lock file")
    parser.add_argument("--video", action="store_true",
                        help="also write a chase-camera MP4 of the aircraft")
    parser.add_argument("--onboard-video", action="store_true",
                        help="record the drone's own camera instead of the chase view")
    parser.add_argument("--stream", action="store_true",
                        help="WebRTC livestream on :49100 (one process only)")
    parser.add_argument("--settle-s", type=float, default=30.0,
                        help="simulated seconds to let PX4's estimator converge")
    parser.add_argument("--camera", default=None,
                        help="override the run config's camera calibration "
                             "(a file in robots/PEGASUS/config/). Changing it also "
                             "changes what FALCON must be told to unproject with, "
                             "and the bridge will refuse a mismatch")
    parser.add_argument("--rate-hz", type=float, default=None,
                        help="override the run config's depth/render rate")
    parser.add_argument("--spawn", default=None, metavar="X,Y",
                        help="override the run config's spawn point. FALCON's own "
                             "init pose comes from the run YAML and will NOT follow, "
                             "so this is for shaking the simulator out, not for flying")
    parser.add_argument("--spawn-yaw-deg", type=float, default=None,
                        help="override the run config's spawn heading, same caveat")
    parser.add_argument("--max-flight-s", type=float, default=None,
                        help="override the run config's flight budget")
    parser.add_argument("--max-speed", type=float, default=1.6,
                        help="tracker horizontal speed ceiling, m/s. Must exceed "
                             "FALCON's own max_linear_velocity and stay under PX4's "
                             "MPC_XY_VEL_MAX")
    parser.add_argument("--control", choices=CONTROL_MODES, default=CONTROL_ATTITUDE,
                        help="where to cut into PX4. 'attitude' rebuilds FALCON's "
                             "B-spline here and commands attitude + throttle, leaving "
                             "PX4 only its attitude and rate loops. 'velocity' is the "
                             "older path -- follow the 100 Hz sampled command and send "
                             "world velocities, keeping PX4's velocity controller in "
                             "the chain. Use it to reproduce the baseline numbers")
    parser.add_argument("--uplink-port", type=int, default=UPLINK_PORT)
    parser.add_argument("--downlink-port", type=int, default=DOWNLINK_PORT)
    parser.add_argument("--connect-timeout-s", type=float, default=300.0,
                        help="how long to wait for the FALCON bridge to appear")
    parser.add_argument("--arm-only", action="store_true",
                        help="stop once PX4 has armed, without connecting to FALCON. "
                             "A spawn point that will not arm costs a whole flight to "
                             "discover otherwise; this checks one in a few minutes")
    return parser.parse_args()


def _spawn_xy(run: dict, args) -> tuple:
    """Where to put the aircraft, honouring the command-line override."""
    if args.spawn is None:
        return float(run["spawn_x"]), float(run["spawn_y"])
    x, y = args.spawn.split(",")
    return float(x), float(y)


def _mission_spec(config: dict, args) -> MissionSpec:
    """Turn a run config into the mission's parameters."""
    run = config["run"]
    limits = KinematicLimits(
        max_speed_xy=args.max_speed,
        max_speed_z=0.8,
        max_yaw_rate=math.radians(60.0),
        max_accel_xy=2.0, max_accel_z=1.5,
    )
    # A ceiling of 3 m, not the tracker's 2 m default. FALCON plans through
    # corners at its full speed limit and the airframe rounds them; two metres of
    # transient lag is normal and is corrected by the next replan, so aborting on
    # it throws away good flights.
    return MissionSpec(
        name=str(run["name"]),
        scene=str(run["scene"]),
        spawn_xy=_spawn_xy(run, args),
        spawn_yaw=math.radians(float(args.spawn_yaw_deg if args.spawn_yaw_deg is not None
                                     else run["spawn_yaw_deg"])),
        cruise_altitude_m=float(run["cruise_altitude_m"]),
        frame_rate_hz=float(args.rate_hz if args.rate_hz is not None
                           else run["frame_rate_hz"]),
        max_flight_s=float(args.max_flight_s if args.max_flight_s is not None
                           else run["max_flight_s"]),
        control_mode=args.control,
        tracker=ReferenceTrackerParams(limits=limits, max_position_error_m=3.0),
        tracking=TrajectoryTrackerParams(max_position_error_m=3.0),
    )


def main() -> int:
    args = _parse_args()
    run_path = setup.find_run(args.run)
    config = setup.load_run(run_path)
    spec = _mission_spec(config, args)
    pegasus_root, px4_dir = setup.resolve_paths(args.dev_root)

    out_dir = args.out_dir or (args.dev_root / "falcon_pegasus" / run_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("run %s from %s" % (spec.name, run_path), flush=True)
    print("spawn (%.2f, %.2f) facing %.0f deg, cruise %.2f m, budget %.0f s"
          % (spec.spawn_xy[0], spec.spawn_xy[1], math.degrees(spec.spawn_yaw),
             spec.cruise_altitude_m, spec.max_flight_s), flush=True)

    # PX4 first: it dials into the simulator's HIL port as a client and retries
    # until the vehicle is listening, so starting it early costs a few reconnects.
    # A vehicle waiting for a PX4 that was never started has no such recovery.
    px4_launch.clear_saved_parameters(px4_dir, args.worker)
    px4_process = px4_launch.launch_px4(
        px4_dir, instance=args.worker, log_path=out_dir / "px4.log")

    simulation_app = flight_session.boot_isaac(stream=args.stream)
    link = FalconLink(args.uplink_port, args.downlink_port, args.connect_timeout_s)
    result = None
    recorder = None
    try:
        from sparx_agency.robots.PEGASUS.adapters.scene import SPAWN_HEIGHT_M
        from sparx_agency.tasks.planning.falcon_pegasus.isaac import sensing

        loop, adapter, px4, chase_camera = setup.bring_up(
            simulation_app, spec.scene, pegasus_root, px4_dir,
            (spec.spawn_xy[0], spec.spawn_xy[1], SPAWN_HEIGHT_M), spec.spawn_yaw,
            camera_config=str(args.camera or config["run"]["camera"]),
            rate_hz=spec.frame_rate_hz,
            worker=args.worker, want_chase_camera=args.video and not args.onboard_video,
            settle_s=args.settle_s, control_mode=spec.control_mode)

        # Before a single frame is sent: does the pose we will label the depth
        # with actually describe the camera that took it? Nothing downstream can
        # tell, and a wrong answer produces a confident, wrong map.
        print(sensing.verify_camera_pose(adapter), flush=True)

        if args.video or args.onboard_video:
            recorder = flight_session.FlightRecorder(
                adapter, out_dir / "recording", rate_hz=spec.frame_rate_hz,
                camera_height_m=spec.cruise_altitude_m, depth_format="png",
                video_out=out_dir / ("%s_flight.mp4" % spec.name),
                video_source="onboard" if args.onboard_video else "chase",
                chase_camera=chase_camera)

        if args.arm_only:
            print("ARM_ONLY: this spawn arms. Stopping before the flight.", flush=True)
            return 0

        print("connecting to the FALCON bridge on 127.0.0.1:%d/%d ..."
              % (args.uplink_port, args.downlink_port), flush=True)
        link.connect(adapter.intrinsics, spec.scene, spec.name)
        print("connected to FALCON", flush=True)

        mission = ExplorationMission(loop, px4, adapter, link, spec, recorder=recorder)
        result = mission.fly()
        print("MISSION %s: %s %s" % (spec.name, result.outcome, result.detail), flush=True)

        from sparx_agency.tasks.planning.sim_flight_recording.episode import (
            settle_after_landing,
        )
        settle_after_landing(loop, px4)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if recorder is not None:
            recorder.finish({"run": spec.name, "scene": spec.scene,
                             "outcome": None if result is None else result.outcome})
        link.close()
        px4_launch.terminate_px4(px4_process, instance=args.worker)
        _write_result(out_dir, run_path, spec, result)
        simulation_app.close()

    return 0 if result is not None and result.ok else 1


def _write_result(out_dir: Path, run_path: Path, spec: MissionSpec, result) -> None:
    """Record what the run did, next to the video it produced."""
    payload = {
        "run": spec.name,
        "config": str(run_path),
        "scene": spec.scene,
        "spawn": {"x": spec.spawn_xy[0], "y": spec.spawn_xy[1],
                  "yaw_deg": math.degrees(spec.spawn_yaw)},
        "cruise_altitude_m": spec.cruise_altitude_m,
        "result": None if result is None else asdict(result),
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    print("wrote %s" % (out_dir / "result.json"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
