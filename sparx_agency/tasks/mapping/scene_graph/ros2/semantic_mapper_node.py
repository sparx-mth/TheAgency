"""semantic_mapper_node — FALCON BEV -> rooms -> latched /scene_graph JSON.

The ROS glue half of the sjtu_project ``semantic_mapper_node.py``, ported
onto the new core (:mod:`sparx_agency.core.mapping.topology`). Every tick it
segments the latest ``/falcon/bev_2d`` occupancy grid into rooms by cutting
the free-space skeleton at the discovered doors, keeps room identities
stable across ticks (IoU registry), accumulates the drone's dwell time per
room, counts frontier clusters, links doors to the rooms they open into,
assigns confirmed object landmarks to rooms, and publishes the whole scene
graph as latched JSON on ``/scene_graph``.

Alongside it, and in the same tick, it publishes the per-cell room picture as
a latched ``nav_msgs/OccupancyGrid`` on ``/scene_graph/room_labels_grid`` —
info (and frame_id) copied from the incoming ``/falcon/bev_2d`` so the two
grids overlay cell for cell but stamped with the tick that segmented it rather
than with the BEV's own build time, data ``0`` where there is no room and
otherwise the room's small ``1..100`` grid value (an int8 field cannot hold a
pid). The
``grid_pid_map`` key of the ``/scene_graph`` payload resolves those values
back to pids; the viz needs both to tint the rooms.

Third, behind ``publish_markers`` (default true), it draws the whole scene
as a latched ``visualization_msgs/MarkerArray`` on ``/scene_graph/markers``
every ``marker_period_s`` — coloured room fills and their outlines, the
Voronoi "open space" spine, door pillars, the gold room -> door -> room
edges of the topology and the green object -> parent-room edges that hang
each landmark off the room containing it — plus the LLM room type and
target probability read off
``/semantic_mapper/room_labels`` and ``/llm_oracle/probabilities``. The
geometry lives in ``scene_graph.ros2.scene_markers``; this node only
gathers the state. ``config/scene_graph.rviz`` opens the matching view.

Doors come from a YAML file of world-XY pairs
(``robots/SJTU/maps/hospital_doors.yaml`` by default) instead of the old
hardcoded parameter list. A grid geometry change (origin/shape/resolution)
resets the registry, dwell times and door discovery — pids are meaningless
across a reshape.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.semantic_mapper_node \
        --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from sparx_agency.core.mapping.topology import (
    FREE_MAX, RoomRegistry, RoomSegmentationParams, WatershedRoomParams,
    compute_rooms, count_frontier_clusters, discover_doors, door_room_pairs,
    link_doors, room_adjacency, room_at_cell, room_color,
    segment_rooms_watershed)
from sparx_agency.tasks.mapping.scene_graph.ros2.payloads import (
    MAX_ROOM_VALUE, assign_room_grid_values, door_entry, room_entry,
    room_value_grid, scene_graph_payload)
from sparx_agency.tasks.mapping.scene_graph.ros2.scene_markers import (
    SceneMarkerState, scene_marker_array)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import (latched_qos,
                                                            sensor_qos)

_PACKAGE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DOORS_YAML = _PACKAGE_ROOT / "robots" / "SJTU" / "maps" / \
    "hospital_doors.yaml"
ROOM_SNAP_CELLS = 3  # flown majority-vote window for cell -> room lookups
# Values of the ``segmentation`` parameter -> the core segmenter each picks.
SEGMENTATION_MODES = ("watershed", "doors")


def load_door_xy(path: Path):
    """The ``doors:`` list of a doors YAML as ``[(x, y), ...]`` floats.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML has no well-formed ``doors`` list.
    """
    if not path.is_file():
        raise FileNotFoundError("doors yaml not found: %s" % (path,))
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    doors = data.get("doors")
    if not isinstance(doors, list) or not doors:
        raise ValueError("doors yaml %s has no 'doors' list" % (path,))
    out = []
    for i, pair in enumerate(doors):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("door %d in %s is not an [x, y] pair: %r"
                             % (i, path, pair))
        out.append((float(pair[0]), float(pair[1])))
    return out


class SemanticMapperNode(Node):
    """BEV occupancy grid -> room segmentation -> /scene_graph."""

    def __init__(self):
        super().__init__("semantic_mapper")

        p = self.declare_parameter
        p("bev_topic", "/falcon/bev_2d")
        p("odom_topic", "/simple_drone/odom")
        p("objects_topic", "/perception/objects")
        p("doors_yaml", str(DEFAULT_DOORS_YAML))
        # The flown node declared this as a RATE (``tick_rate`` = 2.0 Hz) and
        # divided; this one declares a PERIOD, so the flown value is 0.5 s, not
        # 2.0. The dwell-time accounting below credits a whole tick's dt to the
        # room the drone was in at the previous tick, so the period IS the
        # quantisation error on every tau_r the oracle ranks on — at 2.0 s (and
        # a sim-time real-time factor well under 1 on this CPU-only box) the
        # drone crosses whole rooms between ticks. MEASURED on a hospital-sized
        # grid (413x200 @ 0.15 m): the segmentation alone is 46 ms in watershed
        # mode (1.2 ms of it the merge stage) and 32 ms in doors mode, so a tick
        # is ~50 ms and 2 Hz costs ~10% of a core.
        p("tick_period_s", 0.5)
        # Which core segmenter turns the free mask into rooms.
        #   "watershed" (default) -- room_watershed.segment_rooms_watershed:
        #       rooms are basins of the clearance field, doors only forced on
        #       top. Stable as coverage grows.
        #   "doors"               -- room_segmentation.compute_rooms: the
        #       flown skeleton-cut pipeline, unchanged and still reachable.
        # MEASURED on a captured 413x200 @ 0.15 m BEV while growing coverage,
        # largest room as a share of the segmented area: door-cut runs
        # 29 -> 65 -> 79 -> 76%, i.e. it COLLAPSES into one dominant room,
        # while watershed runs 35 -> 28 -> 26 -> 23%. The cause is not tuning:
        # the live BEV only marks walls the drone actually saw, so free space
        # leaks between rooms at openings absent from the 35-entry door list,
        # and the explored medial axis is one connected component that 35 cuts
        # cannot sever. Full table in room_watershed's module docstring.
        p("segmentation", SEGMENTATION_MODES[0])
        # MEASURED, not inherited. The flown stack used 0.60 m, but it cut a
        # 0.05 m ground-truth grid; our BEV is 0.15 m, where 0.60 m is only 4
        # cells and the disk fails to sever the coarser skeleton. Segmenting
        # the hospital interior resampled to 0.15 m, against a ground truth of
        # 20 rooms + 7 corridors:
        #     0.60 m ( 4 cells) ->  5 rooms, largest 96% of the floor
        #     1.20 m ( 8 cells) -> 25 rooms, largest 62%  (one room still eats it)
        #     1.60 m (11 cells) -> 36 rooms, largest 27%  <- chosen
        #     2.40 m (16 cells) -> 48 rooms, largest 19%  (corridors fragmenting)
        # 0.60 m is why a fully explored hospital read as ONE room, and 1.20 m
        # still left one room covering three fifths of it. Raise this only with
        # a measurement -- too large and open-plan space fractures into rooms
        # that are not there.
        p("door_cut_m", 1.60)
        # Watershed only. Repairs the one defect the clearance watershed has:
        # it never under-segments, but a room with two wide spots either side
        # of some furniture (or bent into an L) grows two clearance peaks and
        # is reported as two rooms. Adjacent basins are merged when their
        # DYNAMICS -- the clearance lost from the shallower peak down to the
        # saddle between them -- fall below this. MEASURED on the captured
        # 413x200 BEV with all 35 doors carved, against a ground truth of 27
        # regions: 0.0 (off) 43 rooms / largest 11%, 0.30 -> 36, 0.50 -> 29,
        # 1.00 -> 27, 2.00 -> 26, with the largest room pinned near 12%
        # throughout. 0.50 is the low end of that plateau. A listed door is
        # never merged across whatever this says. Set 0.0 to disable.
        p("merge_dynamics_m", 0.50)
        p("door_match_radius_m", 0.90)
        p("door_discover_m", 0.30)
        # 40 cells is 0.9 m2 at the BEV's 0.15 m -- small enough that wall
        # slivers and doorway stubs survive as "rooms" with no frontiers and no
        # objects, cluttering the ranking panel with entries nothing can be
        # found in. 150 cells is 3.4 m2: smaller than any real hospital room,
        # larger than the fragments. Measured on the same sweep as door_cut_m
        # (1.60 m): 41 rooms at 40 cells, 36 at 150, with the same largest room.
        p("min_room_cells", 150)
        p("room_iou_threshold", 0.15)
        # Flown default from the source node: 4 cells (~0.09 m^2 at 0.15 m
        # resolution) filters single-cell flicker but keeps small openings.
        p("frontier_min_cluster_cells", 4)
        # RViz view. Markers are a strictly larger picture than the JSON --
        # every room mask cell is a point -- so they get their own, slower
        # period rather than riding the segmentation tick.
        p("publish_markers", True)
        p("marker_period_s", 1.0)
        p("marker_cut_disks", True)
        p("room_labels_topic", "/semantic_mapper/room_labels")
        p("probabilities_topic", "/llm_oracle/probabilities")

        g = lambda n: self.get_parameter(n).value
        self._segmentation = str(g("segmentation"))
        if self._segmentation not in SEGMENTATION_MODES:
            raise ValueError(
                "segmentation=%r is not a segmenter; valid values are %s"
                % (self._segmentation, " ".join(SEGMENTATION_MODES)))
        self._door_cut_m = float(g("door_cut_m"))
        self._merge_dynamics_m = float(g("merge_dynamics_m"))
        self._door_match_m = float(g("door_match_radius_m"))
        self._door_discover_m = float(g("door_discover_m"))
        self._min_room_cells = int(g("min_room_cells"))
        self._iou_threshold = float(g("room_iou_threshold"))
        self._frontier_min_cells = int(g("frontier_min_cluster_cells"))
        self._publish_markers_enabled = bool(g("publish_markers"))
        self._marker_cut_disks = bool(g("marker_cut_disks"))

        doors_path = Path(str(g("doors_yaml")))
        self._door_xy = load_door_xy(doors_path)

        # ── grid state (set by the first /falcon/bev_2d) ──
        self._grid = None          # (H, W) int8
        self._res = None
        self._origin = None        # (xmin, ymin)
        self._bev_header = None    # frame_id source for the room-label grid
        self._bev_info = None      # verbatim, for the room-label grid
        self._door_cells = []      # [(cx, cy)] in the current grid
        self._discovered = np.zeros(len(self._door_xy), dtype=bool)

        # ── per-room state across ticks ──
        self._registry = RoomRegistry(self._iou_threshold)
        self._time_in_room = {}    # pid -> float seconds
        self._grid_values = {}     # pid -> 1..100 room-label grid value
        self._last_room_pid = None
        self._last_tick_s = None

        self._drone_xy = None
        self._objects = []         # latest confirmed landmarks (wire dicts)
        # {pid: [object, ...]} as published in /scene_graph rooms[].objects.
        # Refreshed by _tick and read by the (independently timed) marker
        # build, so an object detected between two ticks draws its cube but
        # no parent-room edge until the next tick places it -- the marker
        # and the JSON stay in agreement, which matters more than the lag.
        self._objects_by_room = {}
        self._counts = dict(grid=0, tick=0)
        self._last_frontier_total = 0
        self._last_objects_assigned = 0

        # Marker-view state, all of it filled by _tick. The skeleton is the
        # one piece the JSON payload has no room for and the picture cannot
        # do without -- it IS the "open space" spine the RViz view shows.
        self._skeleton = None      # (H, W) bool, cut at discovered doors
        self._frontier_by_pid = {}
        self._door_rooms = {}      # door index -> [pid, ...]
        self._door_pairs = {}      # door index -> [(pid, pid), ...] edges
        self._room_types = {}      # pid -> {"label", "confidence"}
        self._room_probs = {}      # pid -> {"prob"}

        latched = latched_qos()
        sensor = sensor_qos()

        # The BEV is a bridged ROS1 latch: reliable + transient_local, or a
        # late-joining subscriber never sees the map.
        self.create_subscription(OccupancyGrid, str(g("bev_topic")),
                                 self._grid_cb, latched)
        self.create_subscription(Odometry, str(g("odom_topic")),
                                 self._odom_cb, sensor)
        self.create_subscription(String, str(g("objects_topic")),
                                 self._objects_cb, latched)
        self._pub_sg = self.create_publisher(String, "/scene_graph", latched)
        self._pub_room_grid = self.create_publisher(
            OccupancyGrid, "/scene_graph/room_labels_grid", latched)

        # The RViz view. The two LLM topics are drawn here rather than by
        # their own nodes so the label/probability text shares this node's
        # room centroids and DELETEALL -- three separate MarkerArrays could
        # (and did) disagree about which rooms exist after a pid restart.
        self._pub_markers = None
        self._room_labels_topic = str(g("room_labels_topic"))
        self._probabilities_topic = str(g("probabilities_topic"))
        if self._publish_markers_enabled:
            self.create_subscription(String, self._room_labels_topic,
                                     self._room_labels_cb, latched)
            self.create_subscription(String, self._probabilities_topic,
                                     self._probabilities_cb, latched)
            self._pub_markers = self.create_publisher(
                MarkerArray, "/scene_graph/markers", latched)
            self.create_timer(float(g("marker_period_s")),
                              self._publish_markers)

        self.create_timer(float(g("tick_period_s")), self._tick)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "semantic_mapper up: bev=%s odom=%s objects=%s | %d doors from "
            "%s | segmentation=%s tick=%.1fs cut=%.2fm merge=%.2fm "
            "match=%.2fm discover=%.2fm min_room=%dc iou=%.2f "
            "frontier_min=%dc" % (
                g("bev_topic"), g("odom_topic"), g("objects_topic"),
                len(self._door_xy), doors_path, self._segmentation,
                float(g("tick_period_s")), self._door_cut_m,
                self._merge_dynamics_m,
                self._door_match_m, self._door_discover_m,
                self._min_room_cells, self._iou_threshold,
                self._frontier_min_cells))
        self.get_logger().info(
            "markers: %s on /scene_graph/markers every %.1fs (labels=%s "
            "probs=%s cut_disks=%s)" % (
                "on" if self._publish_markers_enabled else "OFF",
                float(g("marker_period_s")), g("room_labels_topic"),
                g("probabilities_topic"), self._marker_cut_disks))

    # ── grid geometry helpers ────────────────────────────────────────
    def _world_to_cell(self, wx, wy):
        return (int((wx - self._origin[0]) / self._res),
                int((wy - self._origin[1]) / self._res))

    def _cell_to_world(self, cx, cy):
        return (self._origin[0] + (cx + 0.5) * self._res,
                self._origin[1] + (cy + 0.5) * self._res)

    @property
    def _door_cut_cells(self):
        return max(1, int(round(self._door_cut_m / self._res)))

    @property
    def _door_discover_cells(self):
        return max(1, int(round(self._door_discover_m / self._res)))

    @property
    def _door_match_cells(self):
        return max(self._door_cut_cells + 2,
                   int(round(self._door_match_m / self._res)))

    # ── input callbacks ──────────────────────────────────────────────
    def _grid_cb(self, msg: OccupancyGrid):
        self._counts["grid"] += 1
        width, height = msg.info.width, msg.info.height
        res = float(msg.info.resolution)
        origin = (float(msg.info.origin.position.x),
                  float(msg.info.origin.position.y))
        geometry_changed = (
            self._grid is None
            or self._grid.shape != (height, width)
            or abs(res - self._res) > 1e-6
            or abs(origin[0] - self._origin[0]) > 1e-6
            or abs(origin[1] - self._origin[1]) > 1e-6)

        self._grid = np.asarray(msg.data, dtype=np.int8).reshape(
            height, width)
        # The info is kept verbatim — the room-label grid must overlay the BEV
        # cell for cell, so it republishes it unchanged. The header is kept
        # only for its frame_id; see _publish_room_grid for why not its stamp.
        self._bev_header = msg.header
        self._bev_info = msg.info

        if geometry_changed:
            self._res, self._origin = res, origin
            self._door_cells = [self._world_to_cell(*xy)
                                for xy in self._door_xy]
            # Pids, dwell times and discovery are meaningless across a
            # reshape — restart them all cleanly. Grid values are keyed by
            # pid, so they restart with the pids.
            self._registry = RoomRegistry(self._iou_threshold)
            self._time_in_room = {}
            self._grid_values = {}
            self._last_room_pid = None
            self._last_tick_s = None
            self._discovered = np.zeros(len(self._door_xy), dtype=bool)
            # The marker view is pid-keyed too, and its two LLM inputs are
            # LATCHED and republished only on change: a room type or a target
            # probability left standing across the renumbering would be
            # painted onto whichever room inherits that pid, and would stay
            # there for as long as the classifier and the oracle stay quiet.
            # The skeleton and the per-pid frontier/door links belong to the
            # old shape as well; every one of them is rebuilt by the next
            # _tick, so dropping them costs at most one marker period.
            self._room_types = {}
            self._room_probs = {}
            self._frontier_by_pid = {}
            self._door_rooms = {}
            self._door_pairs = {}
            self._skeleton = None
            self.get_logger().info(
                "grid geometry: %dx%d @ %.3f m, origin (%.2f, %.2f); "
                "door cells cut=%d discover=%d match=%d — state reset" % (
                    width, height, res, origin[0], origin[1],
                    self._door_cut_cells, self._door_discover_cells,
                    self._door_match_cells))

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._drone_xy = (float(p.x), float(p.y))

    def _objects_cb(self, msg: String):
        """Latest confirmed landmarks, filtered ENTRY BY ENTRY.

        One malformed object must not cost the whole snapshot, which is what
        building the replacement list in a single ``try`` did: the
        comprehension aborted, ``self._objects`` was never reassigned and the
        node silently kept its previous set. ``/perception/objects`` is
        latched depth 1 and republished only on change, so for a restarted
        mapper that one bad payload can be the only payload for minutes. The
        well-formed entries are kept, the rest counted and logged.
        """
        try:
            objs = json.loads(msg.data).get("objects", [])
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            self.get_logger().error(
                "bad /perception/objects payload: %s" % (exc,),
                throttle_duration_sec=5.0)
            return
        if not isinstance(objs, list):
            self.get_logger().error(
                "bad /perception/objects payload: 'objects' is %s, not a list"
                % (type(objs).__name__,), throttle_duration_sec=5.0)
            return

        kept, rejected = [], 0
        for o in objs:
            try:
                xy = o["xy"]
                if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                    raise ValueError("xy is not an [x, y] pair: %r" % (xy,))
                kept.append({"id": int(o["id"]), "class": str(o["class"]),
                             "xy": [float(xy[0]), float(xy[1])],
                             "count": int(o["count"])})
            except (KeyError, IndexError, AttributeError, TypeError,
                    ValueError):
                rejected += 1
        if rejected:
            self.get_logger().warning(
                "/perception/objects: dropped %d malformed of %d entries, "
                "kept %d" % (rejected, len(objs), len(kept)),
                throttle_duration_sec=5.0)
        self._objects = kept

    def _marker_json(self, msg: String, topic: str, key: str, kind):
        """The ``key`` field of a latched JSON payload, or None if malformed.

        Both marker inputs are latched depth 1 and republished only on
        change, so one bad payload can be the only payload for minutes —
        it is logged (throttled) rather than passed on as an empty view.

        Args:
            msg: The received ``std_msgs/String``.
            topic: Topic name, for the log line only.
            key: Top-level field to return.
            kind: Type the field must have (``dict`` or ``list``).

        Returns:
            The field, or None.
        """
        try:
            field = json.loads(msg.data)[key]
        except (json.JSONDecodeError, AttributeError, TypeError,
                KeyError) as exc:
            self.get_logger().error("bad %s payload: %s" % (topic, exc),
                                    throttle_duration_sec=5.0)
            return None
        if not isinstance(field, kind):
            self.get_logger().error(
                "bad %s payload: '%s' is %s, not %s"
                % (topic, key, type(field).__name__, kind.__name__),
                throttle_duration_sec=5.0)
            return None
        return field

    def _room_labels_cb(self, msg: String) -> None:
        """LLM room type + confidence per pid, for the z=3.7 marker text."""
        labels = self._marker_json(msg, self._room_labels_topic,
                                   "labels", dict)
        if labels is None:
            return
        kept, rejected = {}, 0
        for pid, entry in labels.items():
            try:
                kept[int(pid)] = {
                    "label": str(entry["label"]),
                    "confidence": float(entry.get("confidence", 0.0))}
            except (AttributeError, KeyError, TypeError, ValueError):
                rejected += 1
        if rejected:
            self.get_logger().warning(
                "%s: dropped %d malformed label(s) of %d"
                % (self._room_labels_topic, rejected, len(labels)),
                throttle_duration_sec=5.0)
        self._room_types = kept

    def _probabilities_cb(self, msg: String) -> None:
        """LLM per-room target probability, for the z=4.5 marker text."""
        rooms = self._marker_json(msg, self._probabilities_topic,
                                  "rooms", list)
        if rooms is None:
            return
        kept, rejected = {}, 0
        for entry in rooms:
            try:
                kept[int(entry["id"])] = {"prob": float(entry["prob"])}
            except (KeyError, TypeError, ValueError):
                rejected += 1
        if rejected:
            self.get_logger().warning(
                "%s: dropped %d malformed room(s) of %d"
                % (self._probabilities_topic, rejected, len(rooms)),
                throttle_duration_sec=5.0)
        self._room_probs = kept

    # ── room lookup ──────────────────────────────────────────────────
    def _pid_at_xy(self, pid_lbl, wx, wy):
        """Pid of the room containing world (wx, wy), or None."""
        cx, cy = self._world_to_cell(wx, wy)
        label = room_at_cell(pid_lbl, cx, cy, snap_cells=ROOM_SNAP_CELLS)
        return None if label is None else label - 1

    def _segment(self, free_mask, cut_cells):
        """Run the configured core segmenter over one free mask.

        Args:
            free_mask: (H, W) bool free-space mask of the current BEV.
            cut_cells: Discovered door cells as ``(cx, cy)`` pairs.

        Returns:
            The ``(room_lbl, skeleton, stats)`` triple both segmenters
            share, so the caller need not know which one ran.
        """
        if self._segmentation == "watershed":
            return segment_rooms_watershed(
                free_mask, self._res,
                WatershedRoomParams(
                    min_room_cells=self._min_room_cells,
                    door_cut_m=self._door_cut_m,
                    merge_dynamics_m=self._merge_dynamics_m),
                cut_cells)
        return compute_rooms(
            free_mask, cut_cells,
            RoomSegmentationParams(door_cut_cells=self._door_cut_cells,
                                   min_room_cells=self._min_room_cells))

    # ── the tick ─────────────────────────────────────────────────────
    def _tick(self):
        self._counts["tick"] += 1
        if self._grid is None:
            return

        grid = self._grid
        free_mask = (grid >= 0) & (grid <= FREE_MAX)

        # Door discovery is sticky: OR the stateless per-tick answer in.
        newly = discover_doors(grid, self._door_cells,
                               self._door_discover_cells)
        for i in np.nonzero(newly & ~self._discovered)[0]:
            self.get_logger().info(
                "door %d discovered at (%.2f, %.2f)"
                % (i, self._door_xy[i][0], self._door_xy[i][1]))
        self._discovered |= newly

        cut_cells = [c for c, d in zip(self._door_cells, self._discovered)
                     if d]
        # The skeleton is kept (the flown node dropped it): it is the
        # "open space" spine the RViz view draws, and it is only valid
        # against the masks of this same tick. Both segmenters return the
        # same triple, so everything below this line is mode-agnostic.
        _, self._skeleton, stats = self._segment(free_mask, cut_cells)
        rooms = self._registry.update(stats, self._cell_to_world)

        # Room-label grid values: stable per pid, recycled only once a room
        # is gone, so the viz never re-tints a room that is still there.
        self._grid_values = assign_room_grid_values(rooms.keys(),
                                                    self._grid_values)
        if len(self._grid_values) < len(rooms):
            self.get_logger().warning(
                "%d rooms but only %d room-label grid values (1..%d); the "
                "surplus rooms are absent from /scene_graph/room_labels_grid"
                % (len(rooms), len(self._grid_values), MAX_ROOM_VALUE),
                throttle_duration_sec=10.0)

        # Pid label image (pid+1, 0 = no room) for every cell -> room lookup.
        pid_lbl = np.zeros(grid.shape, dtype=np.int32)
        for pid, room in rooms.items():
            pid_lbl[room.mask] = pid + 1

        # Dwell time: credit the elapsed sim-time to the room the drone was
        # in at the PREVIOUS tick, then re-locate the drone.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._last_tick_s is not None and self._last_room_pid is not None:
            dt = max(0.0, now_s - self._last_tick_s)
            self._time_in_room[self._last_room_pid] = (
                self._time_in_room.get(self._last_room_pid, 0.0) + dt)
        self._last_tick_s = now_s
        self._last_room_pid = (self._pid_at_xy(pid_lbl, *self._drone_xy)
                               if self._drone_xy is not None else None)

        # Frontier clusters (keys are pid+1 labels -> pids).
        frontier_by_label = count_frontier_clusters(
            grid, pid_lbl, self._frontier_min_cells)
        frontier_by_pid = {label - 1: n
                           for label, n in frontier_by_label.items()}
        self._frontier_by_pid = frontier_by_pid
        self._last_frontier_total = sum(frontier_by_pid.values())

        # Doors <-> rooms. link_doors answers PROXIMITY -- every room with
        # cells in the annulus around the door -- so on its own it links a
        # room on the far side of a wall whenever a door sits near a corner,
        # and every pair of the rooms it names becomes a graph edge. The
        # links are therefore vetted against room adjacency, computed once
        # here for the whole tick (0.2 ms on a 413x200 BEV): two rooms are
        # connected only where their regions actually touch. MEASURED on a
        # captured hospital BEV, 23 of its 35 doors see three or more rooms
        # and 3-4 of the 61 pairs that proposes join rooms with a wall
        # between them (the count moves by one with a one-cell shift in
        # where a door rounds). The wire carries the surviving PAIRS, not
        # just the rooms: the rooms at a door are not all connected to each
        # other, so a consumer cannot re-derive the pairs from the list.
        discovered_idx = [i for i in range(len(self._door_xy))
                          if self._discovered[i]]
        links = link_doors(pid_lbl,
                           [self._door_cells[i] for i in discovered_idx],
                           self._door_cut_cells, self._door_match_cells)
        vetted = door_room_pairs(links, room_adjacency(pid_lbl))
        door_rooms = {i: [] for i in range(len(self._door_xy))}
        door_pairs = {i: [] for i in range(len(self._door_xy))}
        for i, pairs in zip(discovered_idx, vetted):
            # Labels are pid+1; the wire carries pids.
            door_pairs[i] = [(a - 1, b - 1) for a, b in pairs]
            door_rooms[i] = sorted({pid for pair in door_pairs[i]
                                    for pid in pair})
        self._door_rooms = door_rooms
        self._door_pairs = door_pairs
        rooms_doors = {pid: [] for pid in rooms}
        for i, pids in door_rooms.items():
            for pid in pids:
                if pid in rooms_doors:
                    rooms_doors[pid].append(i)

        # Confirmed object landmarks -> rooms.
        objects_by_room = {pid: [] for pid in rooms}
        assigned = 0
        for obj in self._objects:
            pid = self._pid_at_xy(pid_lbl, obj["xy"][0], obj["xy"][1])
            if pid is not None and pid in objects_by_room:
                objects_by_room[pid].append(obj)
                assigned += 1
        self._last_objects_assigned = assigned
        self._objects_by_room = objects_by_room

        payload = scene_graph_payload(
            stamp=now_s,
            resolution=self._res,
            origin_xy=self._origin,
            rooms=[room_entry(pid, room.centroid, room.n_cells,
                              self._time_in_room.get(pid, 0.0),
                              frontier_by_pid.get(pid, 0),
                              room_color(pid),
                              objects_by_room[pid],
                              rooms_doors[pid])
                   for pid, room in rooms.items()],
            doors=[door_entry(i, self._door_xy[i],
                              bool(self._discovered[i]), door_rooms[i],
                              door_pairs[i])
                   for i in range(len(self._door_xy))],
            drone_xy=self._drone_xy,
            drone_room_id=self._last_room_pid,
            grid_values=self._grid_values)
        # JSON first, then the grid it describes: a cell value that reaches
        # the viz before its grid_pid_map entry would be tinted by the raw
        # value, i.e. with some other room's colour, for one frame.
        self._pub_sg.publish(String(data=json.dumps(payload)))
        self._publish_room_grid(rooms)

    def _publish_room_grid(self, rooms):
        """Publish the per-cell room values, geometry-identical to the BEV.

        Args:
            rooms: The registry's ``{pid: TrackedRoom}`` for this tick.
        """
        msg = OccupancyGrid()
        # Geometry is INHERITED and the stamp is NOT. The info must stay the
        # BEV's cell for cell or the viz tints the wrong cells; the BEV's
        # header time, though, is when the BEV was BUILT, and its publisher
        # skips unchanged republishes — so that stamp can be minutes old while
        # this segmentation is from this tick. /scene_graph beside it carries
        # the tick time, and the viz needs both halves to agree.
        msg.info = self._bev_info
        msg.header.frame_id = self._bev_header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        # Shape taken from the info being published, so the data length can
        # never disagree with it; a mask of another shape raises here rather
        # than shipping a malformed grid.
        data = room_value_grid((self._bev_info.height, self._bev_info.width),
                               {pid: room.mask for pid, room in rooms.items()},
                               self._grid_values)
        msg.data = data.flatten().tolist()
        self._pub_room_grid.publish(msg)

    # ── the RViz view ────────────────────────────────────────────────
    def _marker_state(self) -> SceneMarkerState:
        """The latest tick, in the plain-data terms ``scene_markers`` draws."""
        rooms = self._registry.rooms
        return SceneMarkerState(
            resolution=self._res,
            origin_xy=self._origin,
            room_masks={pid: room.mask for pid, room in rooms.items()},
            room_centroids={pid: room.centroid
                            for pid, room in rooms.items()},
            skeleton=self._skeleton,
            dwell_s=self._time_in_room,
            frontier_counts=self._frontier_by_pid,
            doors=[door_entry(i, self._door_xy[i], bool(self._discovered[i]),
                              self._door_rooms.get(i, []),
                              self._door_pairs.get(i, []))
                   for i in range(len(self._door_xy))],
            door_cut_m=self._door_cut_m,
            room_types=self._room_types,
            room_probs=self._room_probs,
            objects=self._objects,
            objects_by_room=self._objects_by_room,
            draw_cut_disks=self._marker_cut_disks)

    def _publish_markers(self) -> None:
        """Rebuild and publish the whole RViz view.

        Viz is never allowed to be fatal: /scene_graph and the room-label
        grid are what the mission flies on, so a marker build that raises is
        logged loudly and retried next period instead of taking the mapper
        down. The array is latched, so a late-joining RViz still gets the
        last complete picture.
        """
        if self._grid is None:
            return
        frame = self._bev_header.frame_id if self._bev_header else ""
        if not frame:
            self.get_logger().error(
                "/falcon/bev_2d carries an empty frame_id — RViz drops "
                "markers with no frame, so none are published",
                throttle_duration_sec=10.0)
            return
        try:
            array = scene_marker_array(self._marker_state(), frame,
                                       self.get_clock().now().to_msg())
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().error("marker build failed: %s" % (exc,),
                                    throttle_duration_sec=5.0)
            return
        self._pub_markers.publish(array)

    def _heartbeat(self):
        c = self._counts
        if self._grid is None:
            self.get_logger().warning(
                "hb tick=%d grid=0 — no /falcon/bev_2d yet (is the FALCON "
                "bev publisher/bridge up?)" % (c["tick"],))
        else:
            self.get_logger().info(
                "hb grid=%d tick=%d rooms=%d(tinted %d) doors=%d/%d "
                "cur_room=%s frontiers=%d objs_in_rooms=%d/%d" % (
                    c["grid"], c["tick"], len(self._registry.rooms),
                    len(self._grid_values),
                    int(self._discovered.sum()), len(self._door_xy),
                    self._last_room_pid, self._last_frontier_total,
                    self._last_objects_assigned, len(self._objects)))
        self._counts = dict(grid=0, tick=0)


def main():
    rclpy.init()
    node = SemanticMapperNode()
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
