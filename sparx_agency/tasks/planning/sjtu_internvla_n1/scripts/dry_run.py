#!/usr/bin/env python3
"""Fly the whole N1 stack with no Gazebo, no GPU and no model, in seconds.

The two ROS2 nodes under test are the **real** ones. What is faked is everything
expensive: a kinematic drone that integrates ``/cmd_vel`` and publishes odometry
and camera frames, and an HTTP server that answers ``/agent/.../step`` from a
script instead of a 7B VLM. Both live here rather than in ``core`` because they
are a test rig, and a test rig belongs beside what it tests.

It exists for the same reason ``falcon_pegasus/stub/check.sh`` does: a hospital
recording costs a world bring-up, a model load and a couple of minutes of
flight, and until now the only way to find out that a decision never reached the
aircraft was to spend one. Every scenario below is a behaviour that has actually
been got wrong on this stack, phrased as something a fake drone can be watched
doing:

* **holds still while the model thinks.** System 2 takes seconds. If the
  aircraft moves through them, the frame the model answered about and the pose
  its route is anchored at are both stale, and the pursuit's first move is
  backwards along its own route.
* **turns when told to turn.** A discrete TURN action is a rotation. Flown as
  the short bent waypoint it is rendered as, a holonomic tracker satisfies it by
  crabbing sideways -- the heading barely moves, the view does not change, and
  the model asks again.
* **flies the whole curve.** A System-1 prediction is 1-2.5 m of route; the
  aircraft has to still be following it a second later, not re-planning.
* **escapes a corridor it cannot fly down.** The depth reflex can pin the
  aircraft at zero forward speed for ever, and a policy asking from a
  *stationary* frame gets the same answer every time -- a closed loop that ate
  seventy seconds of the last real flight.
* **flies through a doorway its route goes through.** The depth corridor used
  to follow the NOSE, so a holonomic tracker crossing an opening with its nose
  off-axis swung the corridor onto the jamb and refused a path drawn straight
  through the middle of the gap.
* **puts the System-2 goal in the world.** A pixel goal redrawn at its original
  coordinate is a sticker: the aircraft turns, the scene slides past, and on
  screen the goal never moves however often the model changes it.

Usage::

    .venv/bin/python -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.dry_run
    ... --scenario turn --seconds 25 --verbose

Run it from the repo root with ROS 2 sourced. It needs no display, no docker and
no card; it deliberately pins ``CUDA_VISIBLE_DEVICES=""`` like every node here.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int8, String

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))
_CONFIG = os.path.join(_REPO_ROOT, "sparx_agency", "robots", "SJTU", "config",
                       "vla", "internvla_n1.yaml")


def _dip_target():
    """The altitude a look-down is configured to reach, and the slack allowed.

    Read from the binding YAML rather than written here, because the dip is
    deliberately floored (`look_down_min_altitude_m`) to keep the aircraft above
    the furniture -- and a test that hard-codes the old 0.70 m fails the day
    that floor does its job, which is exactly the wrong signal.
    """
    with open(_CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    pp, foll = cfg.get("policy_params", {}), cfg.get("follower", {})
    cruise = float(foll.get("target_altitude_m", 1.2))
    dip = abs(float(pp.get("look_down_dip_m", 0.5)))
    floor = float(pp.get("look_down_min_altitude_m", 0.0))
    if floor > 0.0:
        dip = min(dip, max(0.0, cruise - floor))
    return cruise - dip, float(pp.get("look_down_tolerance_m", 0.15))


# ── the scripted model server ────────────────────────────────────────────
def curve(reach_m=2.0, bend_deg=0.0, points=33):
    """A System-1 style body-frame curve: ``reach_m`` of arc bending ``bend_deg``."""
    t = np.linspace(0.0, 1.0, points)
    heading = np.deg2rad(bend_deg) * t
    step = reach_m / max(1, points - 1)
    xy = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], axis=1) * step, axis=0)
    xy -= xy[0]
    return [[float(a), float(b)] for a, b in xy]


def reply(action, trajectory=None, look_down=False, pixel_goal=(300, 300),
          step=1, s1_ms=45.0, s2_ms=2600.0):
    """One ``/agent/<name>/step`` body, in the patched server's exact shape."""
    return {
        "action": [{"action": [action], "ideal_flag": True,
                    "pixel_goal": list(pixel_goal),
                    "trajectory": trajectory,
                    "look_down": bool(look_down),
                    "s1_ms": s1_ms, "s2_ms": s2_ms}],
        "pixel_goal": list(pixel_goal),
        "pixel_goal_step": int(step),
    }


class ScriptedServer(object):
    """A stand-in for the InternVLA-N1 agent server, with a deliberate delay.

    The delay is the point of the fake as much as the answers are: nothing in
    this stack goes wrong at 100 ms per decision, and everything does at three
    seconds.
    """

    def __init__(self, script, think_s=2.5):
        self.script = list(script)
        self.think_s = float(think_s)
        self.calls = 0
        self.port = _free_port()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, {"openapi": "3.0.0"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                if self.path.endswith("/reset"):
                    self._send(200, {"ok": True})
                    return
                if self.path == "/agent/init":
                    self._send(201, {"agent_name": "internvla_n1"})
                    return
                time.sleep(outer.think_s)
                index = min(outer.calls, len(outer.script) - 1)
                outer.calls += 1
                self._send(200, outer.script[index])

        self._server = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── the fake aircraft ────────────────────────────────────────────────────
class FakeDrone(Node):
    """Integrate ``/cmd_vel`` into odometry, and publish camera frames.

    First-order lag on every axis, because the real airframe translates by
    tilting and **coasts** for the better part of a second after the command
    goes to zero -- which is exactly what the settle gate in the policy node
    exists to wait out, so a fake that stops instantly would prove nothing.
    """

    def __init__(self, x=0.0, y=0.0, z=1.2, yaw=0.0, wall_x=None, tau=0.35,
                 door=None):
        super().__init__("fake_drone")
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)
        self.vx = self.vy = self.vz = self.wz = 0.0
        self.cmd = (0.0, 0.0, 0.0, 0.0)
        self.tau = float(tau)
        # A wall at a fixed WORLD x, not a fixed distance ahead: a wall that
        # follows the aircraft cannot be escaped from, so a scenario built on
        # one proves nothing about an escape. This one recedes when the aircraft
        # backs off and leaves the frame when it turns away, which is what a
        # real wall does and the only way to tell a working reflex from a
        # rotating one.
        self.wall_x = wall_x
        # A wall at world x with a gap: ``(x, half_width)``. The hospital's
        # doorways are 0.93 m clear against a 0.70 m corridor, so the opening is
        # passable with 0.115 m to spare -- if the corridor is pointed at it.
        self.door = door
        self.travelled = 0.0
        self._last = time.monotonic()
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Twist, "/simple_drone/cmd_vel", self._on_cmd, 1)
        self._odom = self.create_publisher(Odometry, "/simple_drone/odom", sensor)
        self._rgb = self.create_publisher(Image, "/simple_drone/front/image_raw", sensor)
        self._depth = self.create_publisher(
            Image, "/simple_drone/front_depth/depth/image_raw", sensor)
        self._state = self.create_publisher(Int8, "/simple_drone/state", 1)
        self.create_timer(0.02, self._physics)
        self.create_timer(0.1, self._camera)
        self.create_timer(0.5, self._flying)

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)

    def _physics(self):
        now = time.monotonic()
        dt = min(0.1, max(1e-3, now - self._last))
        self._last = now
        a = dt / (self.tau + dt)
        self.vx += a * (self.cmd[0] - self.vx)
        self.vy += a * (self.cmd[1] - self.vy)
        self.vz += a * (self.cmd[2] - self.vz)
        self.wz += a * (self.cmd[3] - self.wz)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = (c * self.vx - s * self.vy) * dt
        dy = (s * self.vx + c * self.vy) * dt
        self.x += dx
        self.y += dy
        self.z += self.vz * dt
        self.yaw += self.wz * dt
        self.travelled += math.hypot(dx, dy)
        msg = Odometry()
        msg.header.frame_id = "world"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = self.z
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        msg.twist.twist.linear.x = self.vx
        msg.twist.twist.linear.y = self.vy
        msg.twist.twist.linear.z = self.vz
        msg.twist.twist.angular.z = self.wz
        self._odom.publish(msg)

    def _flying(self):
        msg = Int8()
        msg.data = 1
        self._state.publish(msg)

    def _door_depth(self, size=600, fx=390.642735, cx=300.0, room=6.0):
        """Depth of a wall with a gap in it, as this pose sees it."""
        x0, half = self.door
        u = np.arange(size, dtype=np.float32)[None, :].repeat(size, 0)
        right = np.arctan((u - cx) / fx)
        world_right = right - self.yaw          # relative to the wall normal
        gap = x0 - self.x
        if gap <= 0.05:
            return np.full((size, size), room, np.float32)
        hit = self.y + gap * np.tan(-world_right)
        d = np.where(np.abs(hit) < half, room,
                     gap / np.maximum(0.05, np.cos(world_right)))
        return d.astype(np.float32)

    def _camera(self):
        rgb = Image()
        rgb.height = rgb.width = 600
        rgb.encoding = "rgb8"
        rgb.step = 1800
        rgb.data = bytes(600 * 1800)
        self._rgb.publish(rgb)
        depth = np.full((600, 600), 10.0, dtype=np.float32)
        if self.door is not None:
            depth = self._door_depth()
        elif self.wall_x is not None:
            ahead = math.cos(self.yaw)
            gap = self.wall_x - self.x
            if ahead > 0.05 and gap > 0.0:
                depth[:, :] = float(min(10.0, gap / ahead))
        msg = Image()
        msg.height = msg.width = 600
        msg.encoding = "32FC1"
        msg.step = 600 * 4
        msg.data = depth.tobytes()
        self._depth.publish(msg)


# ── the observer ─────────────────────────────────────────────────────────
class Watcher(Node):
    """Record what the stack actually did, for the verdict at the end."""

    def __init__(self, drone):
        super().__init__("dry_run_watcher")
        self.drone = drone
        self.samples = []       # (t, phase, x, y, yaw, held, speed, z, blocked)
        self.routes = []        # (t, points, metres)
        self.holds = []         # (t, bool)
        self.yaw_goals = []     # (t, heading)
        self.info = {}
        self.goal_worlds = []
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Path, "/simple_drone/n1/trajectory", self._on_path, latched)
        self.create_subscription(String, "/simple_drone/n1/info", self._on_info, latched)
        self.create_subscription(Bool, "/simple_drone/n1/hold", self._on_hold, 1)
        self.create_subscription(Float32, "/simple_drone/n1/yaw_goal",
                                 self._on_yaw_goal, 1)
        self._t0 = time.monotonic()
        self.create_timer(0.05, self._sample)

    def _t(self):
        return time.monotonic() - self._t0

    def _sample(self):
        d = self.drone
        self.samples.append((self._t(), self.info.get("phase", ""), d.x, d.y, d.yaw,
                             bool(self.info.get("held")), math.hypot(d.vx, d.vy), d.z,
                             bool(self.info.get("blocked"))))

    def _on_path(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        length = (float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                  if len(pts) >= 2 else 0.0)
        self.routes.append((self._t(), len(pts), length))

    def _on_info(self, msg):
        try:
            self.info = json.loads(msg.data)
        except ValueError:
            return
        gw = self.info.get("goal_world")
        if gw and (not self.goal_worlds or self.goal_worlds[-1] != gw):
            self.goal_worlds.append(gw)

    def _on_hold(self, msg):
        self.holds.append((self._t(), bool(msg.data)))

    def _on_yaw_goal(self, msg):
        self.yaw_goals.append((self._t(), float(msg.data)))


# ── scenarios ────────────────────────────────────────────────────────────
def _config(server_port, overrides=None):
    """A temp copy of the real binding YAML pointed at the fake server."""
    with open(_CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    cfg["server"]["port"] = int(server_port)
    cfg["server"]["host"] = "127.0.0.1"
    cfg["recorder"] = dict(cfg.get("recorder", {}), map="")
    for path, value in (overrides or {}).items():
        section, _, key = path.partition(".")
        cfg.setdefault(section, {})[key] = value
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, handle)
    handle.close()
    return handle.name


SCENARIOS = {
    "curve": dict(
        script=[reply(1, trajectory=curve(2.0, 20.0))],
        think_s=7.0, seconds=60.0, wall_x=None, door=None,
        blurb="a 2 m curve every decision, 7 s of thinking. It must fly them."),
    "turn": dict(
        script=[reply(2), reply(1, trajectory=curve(1.5, 0.0)),
                reply(3), reply(1, trajectory=curve(1.5, 0.0))],
        think_s=2.0, seconds=34.0, wall_x=None, door=None,
        blurb="TURN_LEFT, curve, TURN_RIGHT, curve. The turns must be rotations."),
    "blocked": dict(
        script=[reply(1, trajectory=curve(2.0, 0.0))],
        think_s=1.5, seconds=50.0, wall_x=0.5, door=None,
        # Genuinely blocked immediately before most thinks, so the expiry window
        # overlaps them. What is being checked here is that it expires at all.
        stale_blocked_max=0.60,
        blurb="a wall 0.5 m ahead and a policy that will only say forward."),
    "doorway": dict(
        # OFF-AXIS ON PURPOSE. Started 0.7 m to the right of the opening with a
        # curve that bends back toward it, so the aircraft crosses the gap while
        # its nose is still catching up -- which is precisely the geometry a
        # nose-aligned corridor refused.
        script=[reply(1, trajectory=curve(2.5, 25.0), pixel_goal=(300, 300))],
        think_s=1.5, seconds=45.0, wall_x=None, door=(2.0, 0.465),
        # Started so the bent curve crosses the wall plane at the MIDDLE of the
        # gap while the nose is still catching up. Off-centre by even 0.12 m and
        # the jamb enters the corridor legitimately -- the airframe is 0.63 m
        # wide in a 0.93 m door, so the route's own margin is +-0.115 m and no
        # reflex can give it back.
        start=(0.0, -0.36),
        blurb="a 0.93 m doorway 2 m ahead, crossed off-axis through the middle."),
    "lookdown": dict(
        script=[reply(-1, look_down=True), reply(1, trajectory=curve(2.0, 10.0))],
        think_s=1.5, seconds=30.0, wall_x=None, door=None,
        blurb="a look-down, then a curve. The dip must happen from a standstill."),
}


def run(name, args):
    spec = SCENARIOS[name]
    seconds = args.seconds or spec["seconds"]
    server = ScriptedServer(spec["script"], think_s=spec["think_s"]).start()
    config = _config(server.port)
    print("\n=== %s: %s" % (name, spec["blurb"]))
    print("    fake server on %d, %.1f s per decision, %.0f s of flight"
          % (server.port, spec["think_s"], seconds))

    from sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.n1_policy_node import (
        N1PolicyNode,
    )
    from sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.trajectory_follower_node import (
        TrajectoryFollowerNode,
    )

    # Both nodes read `config_file` in their constructors, so the override has
    # to exist before either is built -- which means it goes on the context, not
    # on the node afterwards. The fake drone and the watcher simply never
    # declare it and are unaffected.
    rclpy.init(args=["--ros-args", "-p", "config_file:=%s" % config])
    start = spec.get("start", (0.0, 0.0))
    drone = FakeDrone(x=start[0], y=start[1],
                      wall_x=spec["wall_x"], door=spec.get("door"))
    watcher = Watcher(drone)
    policy = N1PolicyNode()
    follower = TrajectoryFollowerNode()

    executor = MultiThreadedExecutor(num_threads=8)
    for node in (drone, watcher, policy, follower):
        executor.add_node(node)
    stop = time.monotonic() + seconds
    try:
        while time.monotonic() < stop and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    finally:
        for node in (drone, watcher, policy, follower):
            node.destroy_node()
        rclpy.shutdown()
        server.stop()
        os.unlink(config)
    return verdict(name, drone, watcher, server, args)


def _distance_during(samples, phase):
    """Ground covered while the stack reported ``phase``, metres."""
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        if a[1] == phase:
            total += math.hypot(b[2] - a[2], b[3] - a[3])
    return total


def verdict(name, drone, watcher, server, args):
    """Print what happened and return True when it is what should have happened."""
    samples = watcher.samples
    moving_while_held = [s for s in samples if s[5] and s[6] > 0.08]
    phases = {}
    for s in samples:
        phases[s[1]] = phases.get(s[1], 0) + 1
    routes = [r for r in watcher.routes if r[1] >= 2]
    yaw0 = samples[0][4] if samples else 0.0
    yaw1 = samples[-1][4] if samples else 0.0
    yaws = [s[4] for s in samples] or [0.0]
    swing = math.degrees(max(yaws) - min(yaws))

    print("    server calls      %d" % server.calls)
    print("    routes published  %d  (lengths %s)"
          % (len(routes), ", ".join("%.2f" % r[2] for r in routes[:8])))
    print("    yaw goals         %d" % len(watcher.yaw_goals))
    print("    distance flown    %.2f m" % drone.travelled)
    print("    heading change    %+.1f deg  (swing %.1f deg)"
          % (math.degrees(yaw1 - yaw0), swing))
    print("    phases seen       %s" % ", ".join(
        "%s=%.0f%%" % (k or "-", 100.0 * v / max(1, len(samples)))
        for k, v in sorted(phases.items(), key=lambda kv: -kv[1])))
    print("    moving while held %d of %d samples" % (len(moving_while_held), len(samples)))
    print("    blocked            %.0f%% of samples, %.0f%% of stationary ones"
          % (100.0 * sum(s[8] for s in samples) / max(1, len(samples)),
             100.0 * sum(s[8] for s in samples if s[1] in ("thinking", "settling", "dipping"))
             / max(1, sum(1 for s in samples if s[1] in ("thinking", "settling", "dipping")))))
    if args.verbose:
        for t, held in watcher.holds:
            print("      %6.2fs hold=%s" % (t, held))
        for t, heading in watcher.yaw_goals:
            print("      %6.2fs yaw_goal=%.1f deg" % (t, math.degrees(heading)))

    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print("    [%s] %s%s" % ("PASS" if passed else "FAIL", label,
                                 ("  -- " + detail) if detail else ""))

    check("the model was asked at all", server.calls >= 2)
    # BLOCKED MUST NOT OUTLIVE THE ATTEMPT. The depth reflex has an opinion only
    # while the aircraft is translating; published as a state it used to survive
    # every hold, rotation and dip that followed, so a recording said "BLOCKED
    # settling" about a drone that had not tried to move for half a minute.
    # Measured across five hospital runs before the fix: true for 4-18% of
    # settling samples and up to 62% of turning samples.
    stationary = [s for s in samples if s[1] in ("thinking", "settling", "dipping")]
    stale = [s for s in stationary if s[8]]
    # The bound is per scenario, because "blocked while stationary" is not
    # always wrong: in the `blocked` scenario the aircraft is genuinely against
    # a wall right up to the moment it stops to think, and the flag's expiry
    # window then legitimately overlaps the first part of the think. What must
    # never happen is the flag surviving the whole of it, which is what it used
    # to do -- across the five hospital runs it reported blocked through 62% of
    # *turning* samples, minutes after the last attempt.
    limit = SCENARIOS[name].get("stale_blocked_max", 0.10)
    check("blocked does not outlive the attempt",
          len(stale) <= limit * max(1, len(stationary)),
          "%d of %d stationary samples still reported blocked (limit %.0f%%)"
          % (len(stale), len(stationary), 100 * limit))
    # The aircraft must be still while it thinks. A handful of samples above the
    # threshold is the coast at the start of a hold; a sustained count is the
    # bug this whole change exists to fix.
    check("stands still while thinking",
          len(moving_while_held) <= 0.06 * max(1, len(samples)),
          "%d samples moving while held" % len(moving_while_held))

    if name == "curve":
        check("routes are curves, not stubs",
              bool(routes) and min(r[2] for r in routes) > 1.0,
              "shortest %.2f m" % (min(r[2] for r in routes) if routes else 0.0))
        # FLOWN, not merely committed. A 7 s think is realistic for System 2,
        # and a deadline stamped before the inference is already part-spent when
        # the aircraft is handed the route: measured in the hospital, 19 m of
        # route committed and 2.3 m of ground covered. The test is the ratio.
        committed_m = sum(r[2] for r in routes)
        check("and flew most of what it committed",
              drone.travelled > 0.6 * committed_m,
              "flew %.1f m of %.1f m committed" % (drone.travelled, committed_m))
    if name == "turn":
        check("turns were requested as rotations", len(watcher.yaw_goals) >= 2)
        # The EXCURSION, not the net change: this script turns left and then
        # right, so a net-zero heading is the correct outcome and testing the
        # net would pass an aircraft that never turned at all.
        check("the heading actually swung", swing > 10.0, "%.1f deg" % swing)
        # And the turns must be rotations, not crabs: a bent 0.25 m waypoint
        # moves the aircraft while barely moving the nose, which is the whole
        # failure this mode replaces.
        turn_move = _distance_during(samples, "turning")
        check("it rotated rather than crabbed", turn_move < 0.15,
              "%.2f m travelled while turning" % turn_move)
    if name == "blocked":
        # The MAXIMUM standoff reached, not the final one: this fake policy only
        # ever says "fly forward", so it walks straight back into the wall after
        # every escape and the last sample says nothing about whether the escape
        # worked. What is being tested is the reflex, not the fake.
        max_gap = max(drone.wall_x - s[2] for s in samples)
        check("the escape fired", watcher.info.get("escapes", 0) >= 1,
              "escapes=%s" % watcher.info.get("escapes"))
        # BOTH halves, because either alone is a reflex that does not work.
        # Rotating on the spot next to a wall leaves it inside the depth
        # corridor across most of the arc -- thirteen rotations and zero metres,
        # measured. Backing off without looking anywhere else re-approaches the
        # same wall.
        check("it broke contact", max_gap > 0.8,
              "reached %.2f m from the wall (started at 0.50)" % max_gap)
        check("and looked somewhere else", swing > 20.0, "%.1f deg swing" % swing)
        check("and got moving again", drone.travelled > 0.8,
              "%.2f m flown" % drone.travelled)
    if name == "doorway":
        x0, _ = drone.door
        check("it went through the opening", drone.x > x0 + 0.3,
              "reached x=%.2f, door at %.2f" % (drone.x, x0))
        check("without a hard block", watcher.info.get("escapes", 0) == 0,
              "escapes=%s" % watcher.info.get("escapes"))
        check("and the goal was placed in the world", bool(watcher.goal_worlds),
              "goal_world=%s" % (watcher.goal_worlds[:1] or "never published"))
    if name == "lookdown":
        low = min(s[7] for s in samples)
        target, tol = _dip_target()
        # ARRIVAL is what is being tested, not depth. The dip used to stall
        # around 0.82 m because the altitude ramp outran what a proportional
        # hold could track (rate / kp = 0.29 m against a 0.35 m release band),
        # and the run reported "never reached" three times in some flights.
        check("the aircraft reached the dip altitude",
              low <= target + tol,
              "lowest %.2f m, asked for %.2f m (+%.2f tolerance)" % (low, target, tol))
        check("and then flew a curve", bool(routes))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="all",
                        choices=sorted(SCENARIOS) + ["all"])
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="override the scenario's flight time")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = {}
    for name in names:
        results[name] = run(name, args)
    print("\n=== dry run summary ===")
    for name, passed in results.items():
        print("  %-10s %s" % (name, "PASS" if passed else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
