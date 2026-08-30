#!/usr/bin/env python3
"""Record a run: drone camera + N1's top-down route + how much has been seen.

Subscribes only to topics -- the camera, the odometry, the committed and full N1
routes, and the `/simple_drone/n1/info` status the policy node publishes (action,
System-1/System-2 FPS, the S2 pixel goal). Every pixel is drawn by the ROS-free
:mod:`~sparx_agency.tasks.planning.sjtu_internvla_n1.recording` helpers, so this
node is thin: pull the latest of each, compose, write a frame at a fixed rate.

It also **measures the flight**, because an exploration order has no state at
which it is satisfied and a recording of one is only evidence if it carries a
number. :class:`~sparx_agency.core.planning.exploration.visibility_coverage.
VisibilityCoverage` sweeps the camera's field of view across the same
ground-truth map the route is drawn on and reports the share of the building's
floor that has been looked at. It lives here rather than in the policy node on
purpose: the flight loop is synchronous and timing-critical, and nothing that
merely watches a flight belongs inside it.

Writes with OpenCV's own `mp4v` encoder (no system ffmpeg needed). Pair it with
`ros2 bag record` in `scripts/record_run.sh` for a lossless copy of every topic.

CPU-only, like every node in this stack.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

# cv_bridge is convenient but absent from the SJTU sim's Humble image, where this
# recorder runs so the camera is native (host<->container DDS drops large
# messages). Decode manually when it is missing; the front camera is rgb8.
try:
    from cv_bridge import CvBridge
    _CV_BRIDGE = CvBridge()
except Exception:  # noqa: BLE001
    _CV_BRIDGE = None


def _imgmsg_to_bgr(msg):
    """sensor_msgs/Image -> HxWx3 BGR, without requiring cv_bridge."""
    if _CV_BRIDGE is not None:
        try:
            return _CV_BRIDGE.imgmsg_to_cv2(msg, "bgr8")
        except Exception:  # noqa: BLE001
            pass
    enc = (msg.encoding or "rgb8").lower()
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "rgb8" else img)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(buf.reshape(msg.height, msg.width, -1)[:, :, :3])

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration.survey_state import load_survey
from sparx_agency.core.planning.exploration.visibility_coverage import (
    VisibilityCoverage,
    cone_from_intrinsics,
)
from sparx_agency.core.planning.vlas.common.pixel_geometry import (
    body_to_pixel,
    world_to_body,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    load_map_backdrop,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import (
    CoverageOverlay,
    OverlayInfo,
    TopDownRenderer,
    compose,
    draw_camera_panel,
)

# <repo>/sparx_agency/tasks/planning/sjtu_internvla_n1/ros2/<this file>
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))


def _yaw_from_quat(q):
    """Yaw (radians, CCW from +x) from a geometry_msgs quaternion."""
    return yaw_from_quaternion((q.x, q.y, q.z, q.w))


def _load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _path_xy(msg):
    return np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses],
                    dtype=float) if msg.poses else None


class _NoBookkeeping(object):
    """Somewhere for ``load_survey`` to put the half of a survey this node has
    no use for. The recorder draws what has been seen; which rooms are finished
    and which orders are retired are the supervisor's business."""

    def __init__(self):
        self._accepted = set()
        self._exhausted = set()
        self._attempts = {}
        self._issues = {}


class N1RunRecorderNode(Node):
    """Compose the drone camera and N1's route into a recorded MP4."""

    def __init__(self):
        super().__init__("n1_run_recorder_node")
        self.declare_parameter("config_file", "")
        # Seconds to record before closing the file and exiting; 0 means "until
        # stopped". This exists because the video must NOT depend on a signal
        # arriving: `ros2 launch` shutting down in a hurry, a sibling node's
        # on_exit racing it, or a parent that dies first all leave an mp4 with
        # every frame in it and no moov atom -- a plausible 15 MB file that
        # nothing can play. A recorder that knows when it is finished closes its
        # own file.
        self.declare_parameter("record_seconds", 0.0)
        self.declare_parameter("output", "")
        cfg = _load_config(self.get_parameter("config_file").value)
        self._cfg = cfg
        topics = cfg.get("topics", {})
        rec = cfg.get("recorder", {})

        rgb_topic = topics.get("rgb", "/simple_drone/front/image_raw")
        self._rgb_compressed = topics.get("rgb_type", "raw") == "compressed"
        odom_topic = topics.get("odom", "/simple_drone/odom")
        traj_topic = topics.get("trajectory", "/simple_drone/n1/trajectory")
        full_topic = topics.get("trajectory_full", "/simple_drone/n1/trajectory_full")
        info_topic = topics.get("info", "/simple_drone/n1/info")

        camera = cfg.get("camera", {})
        # The camera model the pixel goal was computed with. The recorder needs
        # it to put that goal back on a LATER frame: the goal is a place, and a
        # place has to be re-projected from where the aircraft is now.
        self._intrinsics = Intrinsics(
            fx=float(camera.get("fx", 390.642735)), fy=float(camera.get("fy", 390.642735)),
            cx=float(camera.get("cx", 300.0)), cy=float(camera.get("cy", 300.0)),
            width=int(camera.get("width", 600)), height=int(camera.get("height", 600)))
        self._goal_world = None

        self._panel_w = int(rec.get("panel_width", 640))
        self._panel_h = int(rec.get("panel_height", 480))
        self._record_fps = float(rec.get("fps", 10.0))
        output = self.get_parameter("output").value or rec.get(
            "output", "/tmp/sjtu_n1/run.mp4")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        self._output = output

        self._bridge = _CV_BRIDGE
        self._lock = threading.Lock()
        self._frame = None
        self._pose = None
        self._committed = None
        self._full = None
        self._info = OverlayInfo()
        # The route is drawn ON the building when a map is configured. The path
        # is resolved relative to the repo root so the config can name it the
        # way every other path in this tree is named.
        map_path = rec.get("map", "")
        if map_path and not os.path.isabs(map_path):
            map_path = os.path.join(_REPO_ROOT, map_path)
        backdrop = load_map_backdrop(map_path, logger=self.get_logger())
        self._topdown = TopDownRenderer(
            size=(self._panel_w, self._panel_h),
            backdrop=backdrop,
            local_span_m=float(rec.get("local_span_m", 14.0)),
            overview_fraction=float(rec.get("overview_fraction", 0.42)))
        self._coverage = self._build_coverage(backdrop, rec.get("coverage", {}))
        self._coverage_period_s = 1.0 / max(1e-3, float(
            rec.get("coverage", {}).get("rate_hz", 5.0)))
        self._coverage_log_s = float(rec.get("coverage", {}).get("log_every_s", 10.0))
        self._last_observe_s = 0.0
        self._last_coverage_log_s = 0.0
        self._coverage_checked = False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output, fourcc, self._record_fps,
                                       (self._panel_w * 2, self._panel_h), True)
        if not self._writer.isOpened():
            raise RuntimeError("could not open VideoWriter at %s" % (output,))
        self._frames_written = 0
        self._goal_fresh_s = float(rec.get("goal_fresh_s", 1.0))
        self._record_seconds = float(self.get_parameter("record_seconds").value or 0.0)
        self._started_s = time.monotonic()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        rgb_type = CompressedImage if self._rgb_compressed else Image
        self.create_subscription(rgb_type, rgb_topic, self._on_rgb, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(Path, traj_topic, self._on_committed, latched)
        self.create_subscription(Path, full_topic, self._on_full, latched)
        self.create_subscription(String, info_topic, self._on_info, latched)

        self.create_timer(1.0 / max(1e-3, self._record_fps), self._record)
        self.get_logger().info(
            "recording drone camera + N1 route to %s (%dx%d @ %.0f fps)"
            % (output, self._panel_w * 2, self._panel_h, self._record_fps))

    # ── coverage ─────────────────────────────────────────────────────
    def _build_coverage(self, backdrop, cfg):
        """The seen-floor tracker, or None when it cannot be built.

        Never fatal. This node's contract is a playable MP4; a measurement that
        cannot be set up is a line in the log, not a lost recording.
        """
        if backdrop is None:
            self.get_logger().warn(
                "no map backdrop, so no coverage measurement -- there is no "
                "building to divide by")
            return None
        if not bool(cfg.get("enabled", True)):
            return None
        try:
            grid = occupancy_from_mask(
                backdrop.occupied_mask, backdrop.resolution,
                backdrop.origin_x, backdrop.origin_y,
                known=backdrop.known_mask)
            cone = cone_from_intrinsics(
                width=self._intrinsics.width, fx=self._intrinsics.fx,
                # The depth sensor's far clip, not the camera's: past it the
                # pixels carry no measurement, and a ray that keeps going marks
                # free space straight through a wall.
                max_range_m=float(cfg.get("max_range_m", 10.0)),
                # The SJTU front camera sits 0.2 m ahead of the body origin
                # (robots/SJTU/adapters/topics.py, FRONT_CAMERA_OFFSET_FLU).
                forward_offset_m=float(cfg.get("camera_forward_m", 0.2)))
            coverage = VisibilityCoverage(grid, cone)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn("coverage disabled: %s" % (exc,))
            return None
        # RESUME WHAT THE SURVEY ALREADY KNOWS. When a supervisor is carrying a
        # survey across segments, a recorder that started from zero every run
        # would draw 5% on the video while the building was 26% surveyed -- two
        # numbers for one question, and the one on screen the wrong one. Read
        # only: the supervisor owns the file, this just starts where it is.
        state_file = self._cfg.get("supervisor", {}).get("state_file", "")
        if state_file:
            if not os.path.isabs(state_file):
                state_file = os.path.join(_REPO_ROOT, state_file)
            try:
                load_survey(state_file, coverage, _NoBookkeeping(),
                            logger=self.get_logger())
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    "not resuming coverage from %s: %s" % (state_file, exc))
        self.get_logger().info(
            "coverage: %.0f m2 of building floor to see, %.0f deg cone to %.1f m"
            % (coverage.area_total_m2,
               np.degrees(2.0 * cone.half_fov_rad), cone.max_range_m))
        return coverage

    def _observe(self, pose, now_s):
        """Fold one pose into the seen mask, and say so periodically."""
        if self._coverage is None or pose is None:
            return
        if (now_s - self._last_observe_s) < self._coverage_period_s:
            return
        self._last_observe_s = now_s
        if not self._coverage_checked:
            self._coverage_checked = True
            if not self._coverage.contains(pose[0], pose[1]):
                # Loudly, because the alternative is a whole recording that
                # reports 0% and looks exactly like a broken tracker.
                self.get_logger().warn(
                    "the aircraft is at (%.2f, %.2f), which is NOT inside the "
                    "building the coverage is measured against -- the number "
                    "will not move" % (pose[0], pose[1]))
        self._coverage.observe(pose[0], pose[1], pose[3])
        if (now_s - self._last_coverage_log_s) >= self._coverage_log_s:
            self._last_coverage_log_s = now_s
            self.get_logger().info("N1 COVERAGE  %s" % (self._coverage.summary(),))

    def _coverage_overlay(self):
        """What the renderer needs to draw the wash and the banner."""
        if self._coverage is None:
            return None
        return CoverageOverlay(
            seen=self._coverage.seen_mask,
            fraction=self._coverage.fraction_seen,
            area_seen_m2=self._coverage.area_seen_m2,
            area_total_m2=self._coverage.area_total_m2)

    # ── subscriptions ────────────────────────────────────────────────
    def _on_rgb(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = _imgmsg_to_bgr(msg)
            with self._lock:
                self._frame = frame
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error("rgb decode failed: %s" % (exc,))

    def _on_odom(self, msg):
        p = msg.pose.pose
        # z as well as x, y, yaw: re-projecting the goal needs the height the
        # aircraft is at now, and a goal on a table is not a goal on the floor.
        pose = (p.position.x, p.position.y, p.position.z, _yaw_from_quat(p.orientation))
        with self._lock:
            self._pose = pose
        self._topdown.add_pose(pose[0], pose[1])

    def _on_committed(self, msg):
        xy = _path_xy(msg)
        # Once per commitment, which is exactly what this callback is: the
        # renderer keeps every route so a viewer can see what the policy has
        # been producing over the flight rather than only what it is flying
        # this instant.
        self._topdown.note_route(xy)
        with self._lock:
            self._committed = xy

    def _on_full(self, msg):
        with self._lock:
            self._full = _path_xy(msg)

    def _on_info(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        pg = d.get("pixel_goal")
        pgf = d.get("pixel_goal_frame")
        gw = d.get("goal_world")
        with self._lock:
            self._goal_world = tuple(float(c) for c in gw) if gw else None
            self._info = OverlayInfo(
                instruction=d.get("instruction", ""),
                action=d.get("action") or "",
                status="STOP" if d.get("stop") else "navigating",
                s1_fps=d.get("s1_fps"), s2_fps=d.get("s2_fps"),
                s1_ms=d.get("s1_ms"), s2_ms=d.get("s2_ms"),
                pixel_goal=(int(pg[0]), int(pg[1])) if pg else None,
                pixel_goal_frame=(int(pgf[0]), int(pgf[1])) if pgf else None,
                from_curve=bool(d.get("from_curve")),
                curve_share_pct=d.get("curve_share_pct"),
                phase=d.get("phase", ""),
                think_s=d.get("think_s"),
                blocked=bool(d.get("blocked")),
                traj_m=d.get("traj_m"),
                traj_pts=d.get("traj_pts"),
                turn_deg=d.get("turn_deg"),
                commits=d.get("commits"),
                turns=d.get("turns"),
                escapes=d.get("escapes"),
                pixel_goal_fresh=bool(d.get("pixel_goal_fresh")),
                pixel_goal_age=d.get("pixel_goal_age"),
                decision_time=d.get("decision_time"))

    # ── the record loop ──────────────────────────────────────────────
    def _record(self):
        if self._record_seconds > 0.0 and \
                (time.monotonic() - self._started_s) >= self._record_seconds:
            self.get_logger().info("recorded %.0f s; closing" % self._record_seconds)
            self.close_video()
            os._exit(0)
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            pose = self._pose
            committed = None if self._committed is None else self._committed.copy()
            full = None if self._full is None else self._full.copy()
            info = self._info
            goal_world = self._goal_world
        if frame is None:
            return
        # AFTER the frame test, deliberately, though be clear about what that
        # buys: it catches the failure this world actually has -- Gazebo
        # Classic disables its camera sensors outright when it cannot open a
        # display, so not one frame is ever published while odometry runs at
        # 30 Hz, and a run like that reports `seen=n/a` instead of crediting
        # itself with a whole building it never looked at. It does NOT detect a
        # camera that stops mid-flight: `_frame` holds the last one, and the
        # video would be frozen too.
        #
        # Wrapped, because coverage is a WATCHER. This node's contract is a
        # playable MP4, and no measurement bug -- this one or a later one -- may
        # be able to raise out of the frame timer, kill the node, and take the
        # flight down with it through the launch file's `on_exit=Shutdown()`.
        try:
            self._observe(pose, time.monotonic())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn("coverage stopped after an error: %s" % (exc,))
            self._coverage = None
        info = self._reproject_goal(info, goal_world, pose)
        # A goal is only "fresh" for about as long as the aircraft has not yet
        # moved away from the pose it was computed at. Beyond that it is drawn
        # stale even though the decision that produced it is still the current
        # one -- the alternative is a solid target ring held for a hundred
        # frames, which is the lie this whole field exists to stop telling.
        if info.pixel_goal_fresh and info.decision_time is not None:
            if (time.time() - float(info.decision_time)) > self._goal_fresh_s:
                info = replace(info, pixel_goal_fresh=False)
        left = draw_camera_panel(frame, info, (self._panel_w, self._panel_h))
        # The renderer's pose is (x, y, yaw) and always has been; z is this
        # node's own business, for re-projecting the goal.
        flat = None if pose is None else (pose[0], pose[1], pose[3])
        right = self._topdown.render(flat, committed, full, goal_world,
                                     self._coverage_overlay())
        self._writer.write(compose(left, right))
        self._frames_written += 1

    def _reproject_goal(self, info, goal_world, pose):
        """Put the System-2 goal where it is NOW, in the frame about to be drawn.

        This is the whole difference between a marker that tracks the scene and
        one pinned to a screen coordinate. The goal was a pixel in a frame the
        aircraft has since flown away from; re-projected from the live pose it
        stays on the place the model chose, moves across the image as the
        aircraft turns, and leaves the frame when the aircraft looks elsewhere.

        Falls back to the reported pixel when there is no world point -- the
        goal had no usable depth -- so the marker degrades to the old
        fixed-coordinate behaviour rather than disappearing.
        """
        if goal_world is None or pose is None:
            return info
        forward, left, up = world_to_body(goal_world, pose)
        pixel = body_to_pixel(forward, left, up, self._intrinsics)
        if pixel is None:
            # Behind the aircraft. Say so rather than drawing it on the far side
            # of the image, which is where an unguarded projection puts it.
            return replace(info, pixel_goal=None, goal_behind=True,
                           goal_range_m=float(np.hypot(forward, left)))
        w, h = self._intrinsics.width, self._intrinsics.height
        offscreen = not (0 <= pixel[0] < w and 0 <= pixel[1] < h)
        return replace(info,
                       pixel_goal=(int(round(pixel[0])), int(round(pixel[1]))),
                       pixel_goal_frame=(w, h),
                       goal_projected=True, goal_offscreen=offscreen,
                       goal_behind=False,
                       goal_range_m=float(np.hypot(forward, left)))

    def close_video(self):
        """Release the writer, once, and say what was written.

        THIS IS THE WHOLE RECORDING. An mp4 whose writer is never released has
        every frame in it and no ``moov`` atom, so nothing will play it -- the
        file looks like a successful 5 MB result and is worthless. It must
        therefore happen on every exit path, not only the tidy one.
        """
        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        self._writer = None
        writer.release()
        self.get_logger().info(
            "wrote %d frames to %s" % (self._frames_written, self._output))
        # The run's headline, on every exit path the video takes -- the shell's
        # summary and campaign_report both read it back out of nodes.log, and a
        # run that is torn down by a signal has to carry it too.
        coverage = getattr(self, "_coverage", None)
        if coverage is not None:
            self.get_logger().info("N1 COVERAGE FINAL  %s" % (coverage.summary(),))

    def destroy_node(self):
        try:
            self.close_video()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = N1RunRecorderNode()

    # `ros2 launch` stops its children with SIGINT and then, if they linger,
    # SIGTERM. Python turns SIGINT into KeyboardInterrupt but SIGTERM kills the
    # process outright with no unwinding at all -- so without this handler a
    # slightly slow shutdown loses the video even though the code below looks
    # like it covers everything.
    def _finish(_signum, _frame):
        # Close the file, then leave immediately. Raising KeyboardInterrupt out
        # of the handler is not enough: rclpy.spin blocks in C, so the exception
        # is only delivered when a callback next returns to Python, and
        # `ros2 launch`'s shutdown does not always wait that long -- measured
        # here, SIGINT alone left the process running past four seconds. The
        # video is fully flushed by close_video, so there is nothing left to
        # unwind that is worth risking the recording for.
        try:
            node.close_video()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _finish)
    signal.signal(signal.SIGINT, _finish)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()

