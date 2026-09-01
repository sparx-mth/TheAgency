"""Live scene-graph visualization node — the mission dashboard.

Collects every scene-graph topic into a plain-data snapshot and hands it to
:mod:`sparx_agency.tasks.mapping.scene_graph.viz_render`, which does ALL the
drawing (and is unit-tested without ROS). This node owns only ROS plumbing,
timers and file/video output.

Run::

    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.scene_graph_viz_node \
        --ros-args -p use_sim_time:=true

Outputs into ``out_dir`` (default ``/tmp/scene_graph_viz``): ``latest.png``
every render tick (atomic replace), numbered ``frame_NNNNNN.png`` every
``save_period_s``, and ``scene_graph.mp4`` when ``record_mp4`` is true. A cv2
window is shown only when ``show_window`` is true AND ``$DISPLAY`` is set —
never crashes headless.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import (latched_qos,
                                                            sensor_qos)
from sparx_agency.tasks.mapping.scene_graph.viz_render import render_scene
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    load_map_backdrop,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MAP_YAML = _REPO_ROOT / "robots" / "SJTU" / "maps" / "hospital.yaml"

LATCHED = latched_qos()
SENSOR = sensor_qos()

CANVAS_SIZE = (1600, 900)
HEARTBEAT_PERIOD_S = 10.0


def _grid_to_dict(msg):
    """nav_msgs/OccupancyGrid -> the plain dict viz_render consumes."""
    grid = np.asarray(msg.data, dtype=np.int8).reshape(
        msg.info.height, msg.info.width)
    return {"grid": grid,
            "resolution": float(msg.info.resolution),
            "origin": (float(msg.info.origin.position.x),
                       float(msg.info.origin.position.y))}


class _LoggerShim:
    """Adapts the rclpy logger to map_backdrop's ``.info``/``.warn`` duck type."""

    def __init__(self, logger):
        self._logger = logger

    def info(self, text):
        self._logger.info(text)

    def warn(self, text):
        self._logger.warning(text)


class SceneGraphVizNode(Node):
    """Render the live scene graph to PNG/MP4 (and a window, when displayed)."""

    def __init__(self):
        super().__init__("scene_graph_viz_node")
        self.declare_parameter("render_period_s", 0.5)
        self.declare_parameter("save_period_s", 5.0)
        self.declare_parameter("out_dir", "/tmp/scene_graph_viz")
        self.declare_parameter("record_mp4", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("map_yaml", str(_DEFAULT_MAP_YAML))
        self.declare_parameter("use_backdrop", False)
        self.declare_parameter("trail_max", 3000)

        self._render_period = float(self.get_parameter("render_period_s").value)
        self._save_period = float(self.get_parameter("save_period_s").value)
        self._out_dir = Path(str(self.get_parameter("out_dir").value))
        self._record_mp4 = bool(self.get_parameter("record_mp4").value)
        self._show_window = bool(self.get_parameter("show_window").value)
        self._map_yaml = str(self.get_parameter("map_yaml").value)
        self._trail_max = int(self.get_parameter("trail_max").value)
        self.get_logger().info(
            "params: render_period_s=%.2f save_period_s=%.2f out_dir=%s "
            "record_mp4=%s show_window=%s map_yaml=%s trail_max=%d"
            % (self._render_period, self._save_period, self._out_dir,
               self._record_mp4, self._show_window, self._map_yaml,
               self._trail_max))

        self._out_dir.mkdir(parents=True, exist_ok=True)
        # OFF by default, and the default matters more than it looks.
        # map_yaml is the SURVEYED hospital floor plan -- the whole building,
        # including every room the drone has never seen. Drawing it underneath
        # made undiscovered space look explored, which is the opposite of what
        # this view is for: the map must start UNKNOWN and only become free or
        # occupied where the drone has actually looked. With no backdrop the
        # panel is built from the live BEV alone (unknown flat, free washed,
        # occupied bright) exactly as the RViz view is. Turn it on only to
        # check the live map against ground truth, never to present a run.
        self._use_backdrop = bool(self.get_parameter("use_backdrop").value)
        self._backdrop = None
        if self._use_backdrop:
            self._backdrop = load_map_backdrop(
                self._map_yaml, _LoggerShim(self.get_logger()))

        # -- live state (all plain data; consumed by viz_render) ----------
        self._bev = None
        self._room_grid = None
        self._scene_graph = None
        self._room_labels = None
        self._oracle = None
        self._objects = None
        self._target_seen = False
        self._target_info = None
        self._pose = None
        self._trail = []
        self._sim_time = 0.0
        self._bev_wall_t = None
        self._oracle_wall_t = None
        self._room_grid_seen = False
        self._frames = 0
        self._saved = 0
        self._writer = None
        self._window_ok = self._show_window and bool(os.environ.get("DISPLAY"))
        if self._show_window and not self._window_ok:
            self.get_logger().warning(
                "show_window=true but $DISPLAY is unset; running headless")
        self._last_save_wall = 0.0
        self._last_heartbeat_wall = time.monotonic()

        # -- subscriptions (topic contract; do not rename) ----------------
        self.create_subscription(OccupancyGrid, "/falcon/bev_2d",
                                 self._on_bev, LATCHED)
        self.create_subscription(OccupancyGrid, "/scene_graph/room_labels_grid",
                                 self._on_room_grid, LATCHED)
        self.create_subscription(String, "/scene_graph",
                                 self._json_cb("_scene_graph"), LATCHED)
        self.create_subscription(String, "/semantic_mapper/room_labels",
                                 self._json_cb("_room_labels",
                                               unwrap="labels"), LATCHED)
        self.create_subscription(String, "/llm_oracle/probabilities",
                                 self._json_cb("_oracle"), LATCHED)
        self.create_subscription(String, "/perception/objects",
                                 self._json_cb("_objects"), LATCHED)
        self.create_subscription(Bool, "/target_seen",
                                 self._on_target_seen, LATCHED)
        self.create_subscription(String, "/target_seen/info",
                                 self._json_cb("_target_info"), LATCHED)
        self.create_subscription(Odometry, "/simple_drone/odom",
                                 self._on_odom, SENSOR)
        self.create_timer(self._render_period, self._on_render)

    # -- callbacks --------------------------------------------------------

    def _on_bev(self, msg):
        self._bev = _grid_to_dict(msg)
        self._bev_wall_t = time.monotonic()

    def _on_room_grid(self, msg):
        self._room_grid = _grid_to_dict(msg)
        if not self._room_grid_seen:
            self._room_grid_seen = True
            self.get_logger().info("room label grid is up: %dx%d @ %.2f m"
                                   % (msg.info.width, msg.info.height,
                                      msg.info.resolution))

    def _json_cb(self, attr, unwrap=None):
        """A String-JSON callback that parses into ``self.<attr>``.

        A malformed payload is logged loudly and the previous value kept — a
        stale panel labelled by the log beats a crashed dashboard mid-flight.
        """
        def _cb(msg):
            try:
                parsed = json.loads(msg.data)
            except (ValueError, TypeError) as exc:
                self.get_logger().error("bad JSON on %s: %s" % (attr, exc))
                return
            if unwrap is not None:
                parsed = parsed.get(unwrap, {})
            setattr(self, attr, parsed)
            if attr == "_oracle":
                self._oracle_wall_t = time.monotonic()
        return _cb

    def _on_target_seen(self, msg):
        if msg.data and not self._target_seen:
            self.get_logger().info("target seen — banner up")
        self._target_seen = bool(msg.data)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose = (float(p.x), float(p.y),
                      yaw_from_quaternion((q.x, q.y, q.z, q.w)))
        self._sim_time = (msg.header.stamp.sec
                          + msg.header.stamp.nanosec * 1e-9)
        if (not self._trail
                or abs(p.x - self._trail[-1][0]) >= 0.05
                or abs(p.y - self._trail[-1][1]) >= 0.05):
            self._trail.append((float(p.x), float(p.y)))
            if len(self._trail) > self._trail_max:
                self._trail = self._trail[-self._trail_max:]

    # -- rendering / output -----------------------------------------------

    def _snapshot(self):
        now = time.monotonic()
        footer = [
            "bev age %s" % ("%.1fs" % (now - self._bev_wall_t)
                            if self._bev_wall_t else "never"),
            "oracle tick %s" % ("%.1fs ago" % (now - self._oracle_wall_t)
                                if self._oracle_wall_t else "never"),
            "room grid %s   frames %d" % (
                "up" if self._room_grid is not None else "not yet published",
                self._frames),
        ]
        return {"bev": self._bev, "room_grid": self._room_grid,
                "scene_graph": self._scene_graph,
                "room_labels": self._room_labels, "oracle": self._oracle,
                "objects": self._objects, "target_seen": self._target_seen,
                "target_info": self._target_info, "pose": self._pose,
                "trail": self._trail, "sim_time": self._sim_time,
                "footer": footer}

    def _on_render(self):
        canvas = render_scene(self._snapshot(), backdrop=self._backdrop,
                              size=CANVAS_SIZE)
        self._frames += 1
        # The temp name still has to END in .png: imwrite picks its codec from
        # the extension, so ".latest.png.tmp" raises "could not find a writer
        # for the specified extension" every tick and kills the node. The point
        # of the temp file is that a reader tailing latest.png never catches a
        # half-written frame, and os.replace is atomic within a directory.
        tmp = self._out_dir / ".latest.tmp.png"
        if not cv2.imwrite(str(tmp), canvas):
            raise RuntimeError("cv2.imwrite failed for %s" % (tmp,))
        os.replace(str(tmp), str(self._out_dir / "latest.png"))
        wall = time.monotonic()
        if wall - self._last_save_wall >= self._save_period:
            self._last_save_wall = wall
            self._saved += 1
            cv2.imwrite(str(self._out_dir / ("frame_%06d.png" % self._saved)),
                        canvas)
        if self._record_mp4:
            self._write_video(canvas)
        if self._window_ok:
            try:
                cv2.imshow("scene_graph", canvas)
                cv2.waitKey(1)
            except cv2.error as exc:
                self._window_ok = False
                self.get_logger().warning(
                    "cv2 window failed (%s); continuing headless" % exc)
        if wall - self._last_heartbeat_wall >= HEARTBEAT_PERIOD_S:
            self._last_heartbeat_wall = wall
            self._heartbeat()

    def _write_video(self, canvas):
        if self._writer is None:
            fps = max(1.0, 1.0 / max(self._render_period, 1e-3))
            path = self._out_dir / "scene_graph.mp4"
            self._writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (canvas.shape[1], canvas.shape[0]))
            if not self._writer.isOpened():
                self.get_logger().error(
                    "cv2.VideoWriter could not open %s; disabling record_mp4"
                    % path)
                self._writer = None
                self._record_mp4 = False
                return
            self.get_logger().info("recording %s at %.1f fps" % (path, fps))
        self._writer.write(canvas)

    def _heartbeat(self):
        graph = self._scene_graph or {}
        self.get_logger().info(
            "viz: frames=%d saved=%d rooms=%d doors=%d objects=%d "
            "bev=%s room_grid=%s oracle=%s target_seen=%s trail=%d"
            % (self._frames, self._saved, len(graph.get("rooms", [])),
               len(graph.get("doors", [])),
               len((self._objects or {}).get("objects", [])),
               "yes" if self._bev is not None else "no",
               "yes" if self._room_grid is not None else "no",
               (self._oracle or {}).get("source", "no"),
               self._target_seen, len(self._trail)))

    def close(self):
        """Flush the MP4 (a moov-less file is unplayable) and the window."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self.get_logger().info("mp4 finalized")
        if self._window_ok:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphVizNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT and SIGTERM. stop_scene_graph.sh sends SIGTERM,
        # which rclpy turns into ExternalShutdownException out of
        # spin() -- uncaught it printed a traceback on every clean
        # teardown and exited non-zero, so a normal stop read as a
        # crash in the node log.
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
