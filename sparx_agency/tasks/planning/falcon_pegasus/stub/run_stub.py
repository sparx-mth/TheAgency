#!/usr/bin/env python3
"""A stand-in aircraft: everything the Isaac side does, minus Isaac Sim.

An Isaac Sim run costs a Kit boot, a PX4 warm-up and a GPU, which is the wrong
loop to iterate a FALCON *configuration* in. This flies the same mission against
the same bridge with the same wire protocol, on a laptop, in seconds:

* depth comes from raycasting the surveyed ground-truth voxel map instead of
  rendering (:mod:`~.voxel_camera`) -- the same building, measured rather than
  drawn;
* the airframe is a first-order velocity lag instead of PhysX and PX4. Not a toy
  choice: a lag is what an inner-loop velocity controller looks like from
  outside, so the outer-loop tracker has something real to close;
* everything else -- the wire protocol, the timestamps, the handover order, the
  tracker, the exit conditions -- is the code that flies the real aircraft.

So a green stub run means FALCON's configuration, the exploration box, the
camera contract, the bridge and the controller are all right, and the only thing
left to prove on Isaac Sim is the simulator itself. A red one localises the fault
in a minute rather than an hour.

    .venv/bin/python sparx_agency/tasks/planning/falcon_pegasus/stub/run_stub.py \\
        --run 3_open_plan

Run the FALCON side first, exactly as for a real flight.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from sparx_agency.core.common.spatial_math import quat_to_rot
from sparx_agency.core.common.types import KinematicLimits, normalize_angle
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)
from sparx_agency.robots.PEGASUS.adapters.camera_pose import BODY_TO_OPTICAL, camera_pose_world
from sparx_agency.robots.PEGASUS.adapters.vehicle import CAMERA_OFFSET_FLU, camera_intrinsics
from sparx_agency.tasks.planning.falcon_pegasus.isaac import setup
from sparx_agency.tasks.planning.falcon_pegasus.isaac.falcon_client import FalconLink
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.depth_codec import encode_depth
from sparx_agency.tasks.planning.falcon_pegasus.stub.voxel_camera import VoxelDepthCamera

DT = 0.02                    # 50 Hz, the stub's control and physics rate
ODOMETRY_EVERY_N_STEPS = 1   # 50 Hz, matching what the real aircraft sends
CLIMB_RATE_MPS = 0.6
YAW_RATE = math.radians(60.0)
SURVEY_TURN_RATE = math.radians(35.0)
SETTLE_S = 2.0
FIRST_COMMAND_TIMEOUT_S = 40.0
PLANNER_GONE_GRACE_S = 5.0
# See isaac/mission.py: traj_server outlives a dead exploration node and keeps
# publishing the last trajectory's endpoint, so a frozen trajectory id is the
# only sign that FALCON has stopped.
PLANNER_STALL_S = 30.0


class LaggingAircraft:
    """A velocity-commanded rigid body with first-order lag and no attitude.

    ``tau`` is how long the inner loop takes to reach a commanded velocity --
    the one property of a real airframe the outer loop actually has to fight.
    Everything else a multirotor does (tilt, rotor dynamics, ground effect) is
    left to the simulator that has physics.
    """

    def __init__(self, position, yaw: float, tau: float = 0.35):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.zeros(3)
        self.yaw = float(yaw)
        self.tau = float(tau)

    def step(self, velocity_command, yaw_command: float, dt: float) -> None:
        alpha = dt / (self.tau + dt)
        self.velocity += alpha * (np.asarray(velocity_command, dtype=float) - self.velocity)
        self.position += self.velocity * dt
        error = normalize_angle(yaw_command - self.yaw)
        step = YAW_RATE * dt
        self.yaw = normalize_angle(self.yaw + max(-step, min(step, error)))

    @property
    def quaternion_xyzw(self):
        """Attitude as a yaw-only quaternion, scalar last."""
        return (0.0, 0.0, math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0))

    @property
    def nav_position(self):
        """Where FALCON is told the aircraft is: at the camera, not the body.

        The same rule the real aircraft follows, and for the same reason -- see
        ``isaac/sensing.py``'s ``nav_position``. The stub reproduces the mount
        offset exactly so that a planning failure caused by it shows up here,
        where a run costs a minute, rather than on Isaac Sim.
        """
        translation, _quaternion = camera_pose_world(
            self.position, self.quaternion_xyzw, CAMERA_OFFSET_FLU)
        return translation


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="3_open_plan", help="a runs/*.yaml, by name")
    parser.add_argument("--map", type=Path, default=None,
                        help="voxel map npz (default: the run's scene map)")
    parser.add_argument("--max-flight-s", type=float, default=None,
                        help="override the run config's flight budget")
    parser.add_argument("--realtime", action="store_true",
                        help="pace to wall clock. On by default and almost always "
                             "what you want: FALCON walks its trajectory on the wall "
                             "clock and cannot be told otherwise")
    parser.add_argument("--free-run", action="store_true",
                        help="do NOT pace to wall clock. Only useful for profiling; "
                             "the aircraft will outrun or lag its own commands")
    parser.add_argument("--rays", default=None, metavar="WxH",
                        help="ray grid for the depth render (default: a quarter of "
                             "the image, matching FALCON's skip_pixel)")
    parser.add_argument("--out", type=Path, default=None, help="write a result JSON here")
    return parser.parse_args()


def _load_voxels(scene: str, override):
    """The surveyed ground truth for a scene."""
    from sparx_agency.robots.PEGASUS.adapters import scene_map

    path = Path(override) if override else Path(scene_map.MAP_DIR) / ("%s_voxels.npz" % scene)
    if not path.exists():
        raise FileNotFoundError(
            "no surveyed voxel map at %s. The stub renders depth from the survey, "
            "so a scene it has never surveyed cannot be stubbed -- run "
            "tasks/planning/sim_flight_recording/survey_scene.py --scene %s first."
            % (path, scene))
    data = np.load(path)
    return data["voxels"], data["origin"], float(data["resolution"])


def main() -> int:
    args = _parse_args()
    config = setup.load_run(setup.find_run(args.run))
    run = config["run"]
    name = str(run["name"])
    cruise = float(run["cruise_altitude_m"])
    spawn = np.array([float(run["spawn_x"]), float(run["spawn_y"]), 0.15])
    spawn_yaw = math.radians(float(run["spawn_yaw_deg"]))
    frame_period = 1.0 / float(run["frame_rate_hz"])
    budget = float(args.max_flight_s if args.max_flight_s is not None
                   else run["max_flight_s"])
    realtime = not args.free_run

    intrinsics = camera_intrinsics(name=str(run["camera"]))
    voxels, origin, resolution = _load_voxels(str(run["scene"]), args.map)
    ray_shape = None
    if args.rays:
        width, height = args.rays.lower().split("x")
        ray_shape = (int(width), int(height))
    camera = VoxelDepthCamera(voxels, origin, resolution, intrinsics, ray_shape=ray_shape)
    print("stub: %s, %s at %.2f m, camera %dx%d, rays %dx%d"
          % (name, run["scene"], cruise, intrinsics.width, intrinsics.height,
             camera._ray_shape[0], camera._ray_shape[1]), flush=True)

    aircraft = LaggingAircraft(spawn, spawn_yaw)
    tracker = ReferenceTracker3D(ReferenceTrackerParams(limits=KinematicLimits(
        max_speed_xy=1.6, max_speed_z=0.8, max_yaw_rate=math.radians(60.0),
        max_accel_xy=2.0, max_accel_z=1.5)))

    link = FalconLink()
    link.connect(intrinsics, str(run["scene"]), name)
    print("connected to the FALCON bridge", flush=True)

    state = _Flight(link, aircraft, tracker, camera, intrinsics, cruise, spawn_yaw,
                    frame_period, budget, realtime)
    try:
        outcome, detail = state.run()
    finally:
        link.close()

    print("STUB %s: %s %s" % (name, outcome, detail), flush=True)
    summary = {
        "run": name, "outcome": outcome, "detail": detail,
        "flight_s": state.sim_time, "commands": link.commands_received,
        "frames": link.frames_sent, "dropped": link.frames_dropped,
        "trajectories": link.trajectory_id,
        "mean_tracking_error_m": state.mean_error,
        "max_tracking_error_m": state.max_error,
        "distance_m": state.distance,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    return 0 if outcome in ("explored", "flight_timeout", "planner_stopped") else 1


class _Flight:
    """The stub's own version of the mission: climb, hand over, track, stop."""

    def __init__(self, link, aircraft, tracker, camera, intrinsics, cruise, spawn_yaw,
                 frame_period, budget, realtime):
        self.link = link
        self.aircraft = aircraft
        self.tracker = tracker
        self.camera = camera
        self.intrinsics = intrinsics
        self.cruise = cruise
        self.spawn_yaw = spawn_yaw
        self.frame_period = frame_period
        self.budget = budget
        self.realtime = realtime
        self.sim_time = 0.0
        self.distance = 0.0
        self._previous_position = None
        self.errors = []
        self._next_frame_at = 0.0
        self._steps = 0
        self._streaming_odometry = False
        self._finished = False
        self._planner_gone_at = None
        self._wall_start = time.monotonic()

    @property
    def mean_error(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    @property
    def max_error(self) -> float:
        return max(self.errors) if self.errors else 0.0

    def run(self):
        """Fly the whole thing. Returns ``(outcome, detail)``."""
        outcome, detail = self._climb()
        if outcome is not None:
            return outcome, detail
        self._survey_turn()
        outcome, detail = self._handover()
        if outcome is not None:
            return outcome, detail
        return self._explore()

    def _survey_turn(self) -> None:
        """One slow turn on the spot, for the same reason the real mission does it.

        A 90-degree camera that has only pointed one way leaves FALCON planning
        out of a wedge; its coverage tour then targets a cell it cannot route to
        and the FSM never leaves PLAN_TRAJ. See ``isaac/mission.py``.
        """
        print("    surveying: one turn on the spot", flush=True)
        turned = 0.0
        yaw = self.aircraft.yaw
        while turned < 2.0 * math.pi:
            step = SURVEY_TURN_RATE * DT
            yaw = normalize_angle(yaw + step)
            turned += step
            self.aircraft.step((0.0, 0.0, 0.0), yaw, DT)
            self._tick()
        while abs(normalize_angle(self.spawn_yaw - self.aircraft.yaw)) > 0.1:
            self.aircraft.step((0.0, 0.0, 0.0), self.spawn_yaw, DT)
            self._tick()

    def _climb(self):
        target = np.array([self.aircraft.position[0], self.aircraft.position[1], self.cruise])
        started = self.sim_time
        while True:
            error = target - self.aircraft.position
            command = np.clip(error * 1.0, -CLIMB_RATE_MPS, CLIMB_RATE_MPS)
            self.aircraft.step(command, self.spawn_yaw, DT)
            self._tick()
            if (abs(self.aircraft.position[2] - self.cruise) < 0.1
                    and abs(normalize_angle(self.aircraft.yaw - self.spawn_yaw)) < 0.1):
                for _ in range(int(SETTLE_S / DT)):
                    self.aircraft.step((0.0, 0.0, 0.0), self.spawn_yaw, DT)
                    self._tick()
                print("    at %.2f m -- handing over to FALCON" % self.aircraft.position[2],
                      flush=True)
                return None, ""
            if self.sim_time - started > 60.0:
                return "climb_failed", "never reached cruise altitude"
            if not self.link.alive:
                return "link_lost", "the bridge went away during the climb"

    def _handover(self):
        self._streaming_odometry = True
        self.tracker.reset(yaw=self.aircraft.yaw,
                           hold_position=self.aircraft.nav_position)
        started = self.sim_time
        # has_trajectory, not `reference is not None`: traj_server publishes a
        # parked command with trajectory_id 0 before it has planned anything.
        while not self.link.has_trajectory:
            self.aircraft.step((0.0, 0.0, 0.0), self.aircraft.yaw, DT)
            self._tick()
            if not self.link.alive:
                return "link_lost", "the bridge went away before FALCON planned"
            if self.sim_time - started > FIRST_COMMAND_TIMEOUT_S:
                return "no_commands", (
                    "FALCON planned nothing within %.0f s of odometry. Check "
                    "`rostopic echo /uav_simulator/depth_image --noarr` is flowing and "
                    "that the exploration box contains unknown space."
                    % FIRST_COMMAND_TIMEOUT_S)
        print("    first trajectory received -- exploring", flush=True)
        return None, ""

    def _explore(self):
        started = self.sim_time
        last_report = self.sim_time
        last_trajectory = self.link.trajectory_id
        trajectory_at = self.sim_time
        while True:
            # Closed on the SENSOR's position, matching the frame FALCON's
            # reference is expressed in. See LaggingAircraft.nav_position.
            command = self.tracker.update(
                self.link.reference, self.aircraft.nav_position, self.aircraft.yaw,
                DT, velocity=tuple(self.aircraft.velocity),
                reference_age=self.link.reference_age_s(time.time()))
            self.aircraft.step(command.velocity(), command.yaw, DT)
            if not command.holding:
                self.errors.append(command.position_error_m)
            self._tick()

            if self.link.trajectory_id != last_trajectory:
                last_trajectory = self.link.trajectory_id
                trajectory_at = self.sim_time
            elif self.sim_time - trajectory_at >= PLANNER_STALL_S:
                return "planner_stopped", (
                    "FALCON published no new trajectory for %.0f s (still on #%d)"
                    % (PLANNER_STALL_S, last_trajectory))

            if self._finished:
                return "explored", ""
            if not self.link.alive:
                return "link_lost", "the bridge closed the link"
            if self._planner_gone_at is not None and (
                    self.sim_time - self._planner_gone_at >= PLANNER_GONE_GRACE_S):
                return "no_commands", "the trajectory server stopped and did not return"
            if self.sim_time - started > self.budget:
                return "flight_timeout", ("reached the %.0f s budget with exploration "
                                          "still running" % self.budget)
            if self.sim_time - last_report >= 10.0:
                last_report = self.sim_time
                print("    t=%5.1fs pos=(%6.2f,%6.2f,%5.2f) err=%4.2fm traj#%d frames=%d"
                      % (self.sim_time - started, self.aircraft.position[0],
                         self.aircraft.position[1], self.aircraft.position[2],
                         command.position_error_m, self.link.trajectory_id,
                         self.link.frames_sent), flush=True)

    def _tick(self) -> None:
        """One step: service the link, advance the clock, send what is due."""
        for name, detail in self.link.poll():
            if name == protocol.EVENT_EXPLORATION_FINISHED:
                self._finished = True
                print("    FALCON says the box is explored (%s)" % detail, flush=True)
            elif name == protocol.EVENT_PLANNER_GONE:
                self._planner_gone_at = self._planner_gone_at or self.sim_time

        self.sim_time += DT
        self._steps += 1
        if self._previous_position is not None:
            self.distance += float(
                np.linalg.norm(self.aircraft.position - self._previous_position))
        self._previous_position = self.aircraft.position.copy()

        if self._streaming_odometry and self._steps % ODOMETRY_EVERY_N_STEPS == 0:
            self.link.send_odometry(
                time.time(), self.aircraft.nav_position,
                self.aircraft.quaternion_xyzw, tuple(self.aircraft.velocity),
                (0.0, 0.0, 0.0))

        if self.sim_time >= self._next_frame_at:
            self._next_frame_at = self.sim_time + self.frame_period
            self._send_frame()

        if self.realtime:
            behind = self._wall_start + self.sim_time - time.monotonic()
            if behind > 0:
                time.sleep(behind)

    def _send_frame(self) -> None:
        """Render and send one depth frame with the pose that took it."""
        translation, quaternion = camera_pose_world(
            self.aircraft.position, self.aircraft.quaternion_xyzw, CAMERA_OFFSET_FLU)
        rotation_world_body = quat_to_rot(*self.aircraft.quaternion_xyzw)
        depth = self.camera.render(translation,
                                   np.asarray(rotation_world_body).dot(BODY_TO_OPTICAL))
        encoded = encode_depth(depth)
        self.link.send_frame(time.time(), self.intrinsics.width, self.intrinsics.height,
                             translation, quaternion, encoded.tobytes())


if __name__ == "__main__":
    sys.exit(main())
