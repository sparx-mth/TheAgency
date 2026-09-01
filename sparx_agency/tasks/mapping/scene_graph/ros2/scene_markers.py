"""scene_markers -- the RViz MarkerArray picture of the scene graph.

Every function here takes plain data (masks, arrays, dicts, tuples) and
returns ``visualization_msgs`` messages, so the whole view is unit-testable
without a running node. ``semantic_mapper_node`` gathers a
:class:`SceneMarkerState` from its own tick and calls
:func:`scene_marker_array`; nothing else in the pipeline needs to know how
a room is drawn.

The array is rebuilt from scratch on every publish: it is LED by a single
``Marker(action=DELETEALL)`` and carries no lifetimes, so a room that
vanished (or a pid that restarted after a BEV reshape) leaves nothing
stale behind. Every marker id comes from ONE counter shared across all
namespaces, starting at 1 so that no ADD marker collides with the id of the
leading DELETEALL.

The z-stack over a room centroid, from the floor up::

    0.05  room_fill        translucent room-coloured cells
    0.08  room_polygons    the room's outline, opaque, in the room's hue
    0.10  objects          per-class landmark patches
    0.25  skeleton         the Voronoi spine ("open space"), room-coloured
    2.00  rooms            centroid sphere
    2.50  room_room_edges  gold centroid -> door -> centroid dog-legs
    3.00  room_labels      "R7  t=42s  F=3"
    3.70  room_type_labels    "office (0.82)"   <- room_labels topic
    4.50  room_probabilities  "P=0.41"          <- probabilities topic

That stack (3.0 / 3.7 / 4.5) is the one flown in sjtu_project, where the
three lines came from three different nodes; here one node draws all of
them, which is why the probability text drops the ``(label)`` suffix the
old oracle node appended -- the label is already on the line below it.

``room_object_edges`` is the one marker that spans that stack: a translucent
green line from each object up to the centroid of the room it belongs to.
That link is what turns a scatter of coloured cells into a readable *graph*
-- rooms joined to each other through their doors, and every object hanging
off the room that contains it.

One namespace of the original view is deliberately absent. It also published
``voronoi_mesh``: the adjacency polylines of a Voronoi decomposition, drawn
as a LINE_LIST at z=0.15, and it was OFF by default (``publish_voronoi_mesh``
defaulted to false). This pipeline keeps no Voronoi adjacency object to draw
-- rooms come from a *grid* skeleton, and that skeleton is already on screen
as the ``skeleton`` namespace, opaque and in each room's hue. It carries the
same information (where the free-space spine runs, and therefore where two
rooms meet along it), so nothing is missing from the picture and no
adjacency structure is invented here to fill the gap.

The dwell-time label is **ASCII**: the flown text used a Greek tau, which
depends on the glyph coverage of whatever font RViz resolves for marker
text, and a missing glyph costs the whole line. ``t=`` says the same thing
and cannot fail.

Colours are not chosen here: rooms use
:func:`sparx_agency.core.mapping.topology.room_color` and objects use
:func:`sparx_agency.core.mapping.objects.landmarks.class_color`, the same
two functions the PNG/MP4 dashboard uses, so every view of a run agrees.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from sparx_agency.core.mapping.objects.landmarks import class_color
from sparx_agency.core.mapping.topology import room_color

# Room outlines need marching squares. skimage is already a dependency of
# core/mapping/topology and this module is host-side only (RViz), so the
# import belongs up here with the rest -- but a missing skimage must cost
# the outlines, never the mission: the mapper's /scene_graph and room-label
# grid are what the drone flies on.
try:
    from skimage.measure import find_contours
except ImportError:  # pragma: no cover - the venv ships skimage
    find_contours = None

_LOG = logging.getLogger(__name__)
_WARNED_NO_SKIMAGE = []
"""One-shot latch for the missing-skimage warning: a list so the warning
function can mark it without a module-level ``global``. Once, not once per
marker period, or the log is unreadable at 1 Hz."""

# -- geometry, as flown (the old node's viz_* parameters) ----------------
FILL_ALPHA = 0.32
FILL_CAP = 6000
"""Point cap per room_fill marker -- a hospital room is ~10k cells."""
SKELETON_CAP = 1500
SKELETON_SCALE_FACTOR = 1.2
DOOR_RADIUS_M = 0.25
DOOR_HEIGHT_M = 1.40
EDGE_WIDTH_M = 0.18
ROOM_OUTLINE_WIDTH_M = 0.08
"""room_polygons line width. The original flew 0.15 m, at marker scales
~1.6x these (its room edges were 0.25 m against our 0.18); 0.08 m keeps the
outline in the same proportion -- crisp enough to read a room's shape from
the top-down view without becoming a band that hides the fill under it."""
OBJECT_EDGE_WIDTH_M = 0.05
"""room_object_edges line width -- the original's ``viz_obj_edge_w``, kept
as flown: these lines are many and only have to be followable, not loud."""
TEXT_HEIGHT_M = 0.55
TYPE_TEXT_HEIGHT_M = 0.45
DOOR_TEXT_HEIGHT_M = 0.35
OBJECT_TEXT_HEIGHT_M = 0.28
OBJECT_PATCH_CELLS = 2

# -- the z-stack ---------------------------------------------------------
Z_FILL = 0.05
Z_OUTLINE = 0.08
"""Just above room_fill, so the outline always wins the depth test against
its own room's cells instead of z-fighting them -- and still *below*
Z_OBJECT, so an object sitting on a room boundary is never striped by the
line through it. (The original used 0.20, which in this port's stack is
occupied by nothing but sits above the door cut disks; low and tight to the
fill reads better on the top-down view this config opens with.)"""
Z_OBJECT = 0.10
Z_CUT_DISK = 0.15
Z_SKELETON = 0.25
Z_OBJECT_LABEL = 1.0
Z_DOOR_LABEL = DOOR_HEIGHT_M + 0.35
Z_CENTROID = 2.0
Z_EDGE = 2.5
Z_ROOM_LABEL = 3.0
Z_ROOM_TYPE = 3.7
Z_ROOM_PROB = 4.5

# -- fixed colours -------------------------------------------------------
WHITE = (1.0, 1.0, 1.0)
GOLD = (1.0, 0.85, 0.0)
TYPE_LABEL_COLOR = (1.0, 0.85, 0.3)
DOOR_OPEN_COLOR = (1.0, 0.55, 0.0)
DOOR_PENDING_COLOR = (0.55, 0.58, 0.65)
DOOR_PENDING_TEXT_COLOR = (0.75, 0.78, 0.82)
CUT_DISK_COLOR = (0.7, 0.05, 0.05)
OBJECT_EDGE_COLOR = (0.2, 0.7, 0.2)
OBJECT_EDGE_ALPHA = 0.55
"""Semi-transparent, as flown: one edge per object crosses the whole room,
and at full alpha a busy ward turns into a green cage over the map."""


@dataclass
class SceneMarkerState:
    """One tick's worth of scene graph, in the terms the drawing needs.

    Attributes:
        resolution: BEV cell size in meters.
        origin_xy: World ``(x, y)`` of the BEV grid's cell (0, 0) corner.
            Row 0 is minimum y, as in ``nav_msgs/OccupancyGrid`` -- there
            is no row flip anywhere in this module.
        room_masks: ``{pid: (H, W) bool}`` room membership this tick.
        room_centroids: ``{pid: (x, y)}`` world centroids.
        skeleton: ``(H, W) bool`` cut Voronoi skeleton, or None.
        dwell_s: ``{pid: seconds}`` accumulated drone time per room.
        frontier_counts: ``{pid: n}`` frontier clusters per room.
        doors: ``/scene_graph`` door entries -- dicts with ``index``,
            ``xy``, ``discovered``, ``rooms`` (the pids it links) and
            ``room_pairs`` (those pids paired into the edges the door
            actually carries -- the room list is not a clique, so the
            pairs cannot be re-derived from it).
        door_cut_m: Skeleton cut radius, drawn as the cut disk.
        room_types: ``{pid: {"label", "confidence"}}`` from the classifier.
        room_probs: ``{pid: {"prob"}}`` from the LLM oracle.
        objects: Confirmed landmarks -- dicts with ``class``, ``xy``,
            ``count``.
        objects_by_room: ``{pid: [object, ...]}`` -- the SAME grouping the
            node ships as ``/scene_graph``'s ``rooms[].objects``, not a
            second assignment made here. It is the parent-room link the
            ``room_object_edges`` lines draw, so the picture and the JSON
            can never disagree about which room an object is in. An object
            in no room simply appears in no list.
        draw_cut_disks: Whether to draw the door cut disks at all.
    """

    resolution: float
    origin_xy: Tuple[float, float]
    room_masks: Dict[int, np.ndarray] = field(default_factory=dict)
    room_centroids: Dict[int, Tuple[float, float]] = field(
        default_factory=dict)
    skeleton: Optional[np.ndarray] = None
    dwell_s: Dict[int, float] = field(default_factory=dict)
    frontier_counts: Dict[int, int] = field(default_factory=dict)
    doors: List[Dict] = field(default_factory=list)
    door_cut_m: float = 0.60
    room_types: Dict[int, Dict] = field(default_factory=dict)
    room_probs: Dict[int, Dict] = field(default_factory=dict)
    objects: List[Dict] = field(default_factory=list)
    objects_by_room: Dict[int, List[Dict]] = field(default_factory=dict)
    draw_cut_disks: bool = True


class MarkerFactory:
    """One frame, one stamp and one id sequence for a whole MarkerArray.

    Args:
        frame_id: World frame every marker is expressed in.
        stamp: Header stamp shared by every marker; a zero ``Time`` (i.e.
            "latest") when omitted.

    Raises:
        ValueError: If ``frame_id`` is empty -- RViz silently drops such a
            marker, which reads exactly like a dead publisher.
    """

    def __init__(self, frame_id: str, stamp: Optional[Time] = None) -> None:
        if not frame_id:
            raise ValueError("marker frame_id must not be empty")
        self.frame_id = str(frame_id)
        self.stamp = Time() if stamp is None else stamp
        self._next_id = 1  # id 0 belongs to the leading DELETEALL

    def new(self,
            ns: str,
            marker_type: int,
            scale: Sequence[float],
            color: Sequence[float],
            alpha: float = 1.0) -> Marker:
        """An ADD marker in this array's frame, stamp and id sequence."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.stamp
        marker.ns = str(ns)
        marker.id = self._next_id
        self._next_id += 1
        marker.type = int(marker_type)
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = (
            float(v) for v in scale)
        marker.color = ColorRGBA(r=float(color[0]), g=float(color[1]),
                                 b=float(color[2]), a=float(alpha))
        return marker


def _at(marker: Marker, x: float, y: float, z: float) -> Marker:
    """Place ``marker`` at a world point and return it."""
    marker.pose.position = Point(x=float(x), y=float(y), z=float(z))
    return marker


def _cell_points(mask: np.ndarray,
                 origin_xy: Sequence[float],
                 resolution: float,
                 z: float,
                 cap: int,
                 rng: np.random.Generator) -> List[Point]:
    """Cell centres of ``mask`` as world points, subsampled above ``cap``.

    The subsample is without replacement, so a capped room is drawn as a
    thinned-out version of itself rather than as a dense blob in one corner.
    """
    ys, xs = np.nonzero(mask)
    if xs.size > cap:
        keep = rng.choice(xs.size, size=int(cap), replace=False)
        xs, ys = xs[keep], ys[keep]
    ox, oy, res = float(origin_xy[0]), float(origin_xy[1]), float(resolution)
    return [Point(x=ox + (cx + 0.5) * res, y=oy + (cy + 0.5) * res,
                  z=float(z))
            for cx, cy in zip(xs.tolist(), ys.tolist())]


def probability_color(prob: float,
                      prob_max: float) -> Tuple[float, float, float]:
    """The flown oracle ramp: cool grey-blue (low) to hot red (high).

    Normalised by the highest probability on show, not by 1.0, so the best
    room is always fully hot even when the distribution is flat.

    Args:
        prob: This room's probability.
        prob_max: The largest probability among the rooms being drawn.

    Returns:
        ``(r, g, b)`` floats in [0, 1].
    """
    t = 0.0 if prob_max <= 0.0 else max(0.0, min(1.0, float(prob) / prob_max))
    return (0.50 + 0.50 * t, 0.55 - 0.40 * t, 0.75 - 0.55 * t)


def room_fill_markers(factory: MarkerFactory,
                      state: SceneMarkerState,
                      rng: np.random.Generator) -> List[Marker]:
    """ns ``room_fill``: one translucent CUBE_LIST per room."""
    out = []
    for pid in sorted(state.room_masks):
        marker = factory.new("room_fill", Marker.CUBE_LIST,
                             (state.resolution, state.resolution, 0.02),
                             room_color(pid), FILL_ALPHA)
        marker.points = _cell_points(state.room_masks[pid], state.origin_xy,
                                     state.resolution, Z_FILL, FILL_CAP, rng)
        out.append(marker)
    return out


def room_polygon_markers(factory: MarkerFactory,
                         state: SceneMarkerState) -> List[Marker]:
    """ns ``room_polygons``: each room's outline, one LINE_STRIP per contour.

    Marching squares (``skimage.measure.find_contours`` at level 0.5 on the
    boolean mask) gives one contour per connected boundary, so a room with a
    hole in it -- a pillar, an un-mapped island -- draws that hole too. Each
    contour is closed by repeating its first point: skimage already closes a
    contour that lies wholly inside the grid, but one that runs off the edge
    of the BEV comes back OPEN, and an open outline reads as a room with a
    missing wall.

    Without skimage the outlines are dropped and the rest of the view is
    drawn as usual, after one loud log line.
    """
    if find_contours is None:
        if not _WARNED_NO_SKIMAGE:
            _WARNED_NO_SKIMAGE.append(True)
            _LOG.error(
                "skimage is not importable, so the RViz 'room_polygons' "
                "outlines are NOT drawn (every other namespace still is). "
                "Install scikit-image in the interpreter running "
                "semantic_mapper_node to get them back.")
        return []

    ox, oy = float(state.origin_xy[0]), float(state.origin_xy[1])
    res = float(state.resolution)
    out = []
    for pid in sorted(state.room_masks):
        color = room_color(pid)
        for contour in find_contours(
                state.room_masks[pid].astype(np.float32), 0.5):
            # find_contours yields (row, col) = (cy, cx) in CELL space; the
            # same +0.5 cell-centre convention as room_fill, so an outline
            # lands on the cells it encloses rather than half a cell off.
            points = [Point(x=ox + (float(cx) + 0.5) * res,
                            y=oy + (float(cy) + 0.5) * res, z=Z_OUTLINE)
                      for cy, cx in contour]
            if not points:
                continue
            first = points[0]
            points.append(Point(x=first.x, y=first.y, z=first.z))
            marker = factory.new("room_polygons", Marker.LINE_STRIP,
                                 (ROOM_OUTLINE_WIDTH_M, 0.0, 0.0),
                                 color, 1.0)
            marker.points = points
            out.append(marker)
    return out


def skeleton_markers(factory: MarkerFactory,
                     state: SceneMarkerState,
                     rng: np.random.Generator) -> List[Marker]:
    """ns ``skeleton``: the room's share of the Voronoi spine, opaque.

    Raises:
        ValueError: If the skeleton and a room mask disagree on shape --
            that means the two came from different segmentation ticks, and
            drawing it would put the spine in the wrong place.
    """
    if state.skeleton is None:
        return []
    scale = state.resolution * SKELETON_SCALE_FACTOR
    out = []
    for pid in sorted(state.room_masks):
        mask = state.room_masks[pid]
        if mask.shape != state.skeleton.shape:
            raise ValueError(
                "skeleton %s and room %d mask %s are from different ticks"
                % (state.skeleton.shape, pid, mask.shape))
        in_room = state.skeleton & mask
        if not in_room.any():
            continue
        marker = factory.new("skeleton", Marker.CUBE_LIST,
                             (scale, scale, 0.05), room_color(pid), 1.0)
        marker.points = _cell_points(in_room, state.origin_xy,
                                     state.resolution, Z_SKELETON,
                                     SKELETON_CAP, rng)
        out.append(marker)
    return out


def room_node_markers(factory: MarkerFactory,
                      state: SceneMarkerState) -> List[Marker]:
    """ns ``rooms`` + ``room_labels``: centroid sphere and its stats line."""
    out = []
    for pid in sorted(state.room_centroids):
        cx, cy = state.room_centroids[pid]
        out.append(_at(factory.new("rooms", Marker.SPHERE, (0.6, 0.6, 0.6),
                                   room_color(pid)), cx, cy, Z_CENTROID))
        text = factory.new("room_labels", Marker.TEXT_VIEW_FACING,
                           (0.0, 0.0, TEXT_HEIGHT_M), WHITE)
        text.text = "R%d  t=%.0fs  F=%d" % (
            pid, float(state.dwell_s.get(pid, 0.0)),
            int(state.frontier_counts.get(pid, 0)))
        out.append(_at(text, cx, cy, Z_ROOM_LABEL))
    return out


def room_text_markers(factory: MarkerFactory,
                      state: SceneMarkerState) -> List[Marker]:
    """ns ``room_type_labels`` + ``room_probabilities`` over each centroid."""
    out = []
    for pid in sorted(state.room_types):
        if pid not in state.room_centroids:
            continue
        entry = state.room_types[pid]
        text = factory.new("room_type_labels", Marker.TEXT_VIEW_FACING,
                           (0.0, 0.0, TYPE_TEXT_HEIGHT_M), TYPE_LABEL_COLOR)
        text.text = "%s (%.2f)" % (entry.get("label", "?"),
                                   float(entry.get("confidence", 0.0)))
        cx, cy = state.room_centroids[pid]
        out.append(_at(text, cx, cy, Z_ROOM_TYPE))

    probs = [float(e.get("prob", 0.0)) for e in state.room_probs.values()]
    prob_max = max(probs) if probs else 0.0
    for pid in sorted(state.room_probs):
        if pid not in state.room_centroids:
            continue
        prob = float(state.room_probs[pid].get("prob", 0.0))
        text = factory.new("room_probabilities", Marker.TEXT_VIEW_FACING,
                           (0.0, 0.0, TEXT_HEIGHT_M),
                           probability_color(prob, prob_max))
        text.text = "P=%.2f" % (prob,)
        cx, cy = state.room_centroids[pid]
        out.append(_at(text, cx, cy, Z_ROOM_PROB))
    return out


def door_markers(factory: MarkerFactory,
                 state: SceneMarkerState) -> List[Marker]:
    """ns ``doors_*`` + ``door_labels*`` + ``cut_disks``: one pillar per door.

    A discovered door is a bright orange pillar with a white tag; a door the
    exploration has not reached yet is the same pillar in translucent grey,
    so the map shows what is still unopened as well as what is open.
    """
    out = []
    disk_d = 2.0 * float(state.door_cut_m)
    for door in state.doors:
        dx, dy = float(door["xy"][0]), float(door["xy"][1])
        found = bool(door.get("discovered", False))
        ns_pillar = "doors_discovered" if found else "doors_pending"
        ns_label = "door_labels" if found else "door_labels_pending"
        color, alpha = ((DOOR_OPEN_COLOR, 0.95) if found
                        else (DOOR_PENDING_COLOR, 0.35))
        text_color, text_alpha = ((WHITE, 1.0) if found
                                  else (DOOR_PENDING_TEXT_COLOR, 0.55))
        pillar = factory.new(ns_pillar, Marker.CYLINDER,
                             (DOOR_RADIUS_M * 2, DOOR_RADIUS_M * 2,
                              DOOR_HEIGHT_M), color, alpha)
        out.append(_at(pillar, dx, dy, DOOR_HEIGHT_M * 0.5))
        label = factory.new(ns_label, Marker.TEXT_VIEW_FACING,
                            (0.0, 0.0, DOOR_TEXT_HEIGHT_M), text_color,
                            text_alpha)
        label.text = "d%d" % (int(door["index"]),)
        out.append(_at(label, dx, dy, Z_DOOR_LABEL))
        if found and state.draw_cut_disks:
            disk = factory.new("cut_disks", Marker.CYLINDER,
                               (disk_d, disk_d, 0.03), CUT_DISK_COLOR, 0.55)
            out.append(_at(disk, dx, dy, Z_CUT_DISK))
    return out


def room_edge_markers(factory: MarkerFactory,
                      state: SceneMarkerState) -> List[Marker]:
    """ns ``room_room_edges``: gold centroid -> door -> centroid dog-legs.

    One LINE_LIST per edge the door carries, never a straight
    centroid-to-centroid line: the bend at the door is the whole point, because
    it says *where* the two rooms connect and therefore where a searcher has to
    fly through.

    The pairs are read off the door's ``room_pairs``, never enumerated from its
    ``rooms``: a door often sits where three or four regions meet and only some
    of them touch, so pairing up the room list would draw an edge straight
    through a wall.
    """
    out = []
    for door in state.doors:
        if not bool(door.get("discovered", False)):
            continue
        dx, dy = float(door["xy"][0]), float(door["xy"][1])
        for pair in door.get("room_pairs") or []:
            if len(pair) != 2 or pair[0] == pair[1]:
                continue
            if any(pid not in state.room_centroids for pid in pair):
                continue
            ax, ay = state.room_centroids[pair[0]]
            bx, by = state.room_centroids[pair[1]]
            edge = factory.new("room_room_edges", Marker.LINE_LIST,
                               (EDGE_WIDTH_M, 0.0, 0.0), GOLD)
            edge.points = [
                Point(x=float(px), y=float(py), z=Z_EDGE)
                for px, py in ((ax, ay), (dx, dy), (dx, dy), (bx, by))]
            out.append(edge)
    return out


def object_markers(factory: MarkerFactory,
                   state: SceneMarkerState) -> List[Marker]:
    """ns ``objects:<class>`` + ``object_labels``: confirmed landmarks.

    One CUBE_LIST per *class* rather than per object, following the flown
    object mapper's ``obj:<class>`` convention: a namespace per class is what
    lets RViz switch a noisy detector class off without losing the rest.
    """
    res = float(state.resolution)
    offset = (OBJECT_PATCH_CELLS - 1) * 0.5
    by_class = {}  # type: Dict[str, List[Dict]]
    for obj in state.objects:
        by_class.setdefault(str(obj["class"]), []).append(obj)

    out = []
    for cname in sorted(by_class):
        marker = factory.new("objects:%s" % (cname,), Marker.CUBE_LIST,
                             (res, res, 0.06), class_color(cname), 1.0)
        for obj in by_class[cname]:
            ox, oy = float(obj["xy"][0]), float(obj["xy"][1])
            for i in range(OBJECT_PATCH_CELLS):
                for j in range(OBJECT_PATCH_CELLS):
                    marker.points.append(
                        Point(x=ox + (j - offset) * res,
                              y=oy + (i - offset) * res, z=Z_OBJECT))
        out.append(marker)
    for obj in state.objects:
        text = factory.new("object_labels", Marker.TEXT_VIEW_FACING,
                           (0.0, 0.0, OBJECT_TEXT_HEIGHT_M), WHITE)
        text.text = "%s x%d" % (obj["class"], int(obj.get("count", 1)))
        out.append(_at(text, float(obj["xy"][0]), float(obj["xy"][1]),
                       Z_OBJECT_LABEL))
    return out


def room_object_edge_markers(factory: MarkerFactory,
                             state: SceneMarkerState) -> List[Marker]:
    """ns ``room_object_edges``: object -> parent-room centroid, one per link.

    A 2-point LINE_LIST rather than one long LINE_LIST of every link: RViz
    draws them identically, but one marker per edge means a single object
    that moved cannot silently drag the whole bundle, and the id counter
    still makes each one addressable.

    The membership is ``state.objects_by_room``, i.e. the node's own tick
    assignment -- an object whose room is unknown (outside every mask, or
    detected since the last segmentation tick) draws its cube and its label
    but no edge, which is the honest picture: it is on the map, not yet in
    the graph.
    """
    out = []
    for pid in sorted(state.objects_by_room):
        if pid not in state.room_centroids:
            continue
        rx, ry = state.room_centroids[pid]
        for obj in state.objects_by_room[pid]:
            edge = factory.new("room_object_edges", Marker.LINE_LIST,
                               (OBJECT_EDGE_WIDTH_M, 0.0, 0.0),
                               OBJECT_EDGE_COLOR, OBJECT_EDGE_ALPHA)
            edge.points = [
                Point(x=float(obj["xy"][0]), y=float(obj["xy"][1]),
                      z=Z_OBJECT),
                Point(x=float(rx), y=float(ry), z=Z_CENTROID)]
            out.append(edge)
    return out


def scene_marker_array(state: SceneMarkerState,
                       frame_id: str,
                       stamp: Optional[Time] = None,
                       rng: Optional[np.random.Generator] = None
                       ) -> MarkerArray:
    """The whole RViz view of one tick, as one self-clearing MarkerArray.

    Args:
        state: The tick's rooms, doors, labels, probabilities and objects.
        frame_id: World frame -- the BEV's own, so map and markers overlay.
        stamp: Header stamp for every marker (zero, i.e. latest, if None).
        rng: Generator for the room_fill / skeleton subsampling; a fresh
            default generator when None.

    Returns:
        A MarkerArray led by ``DELETEALL``, so the display is rebuilt rather
        than accumulated -- room pids restart on every BEV reshape, and a
        left-over marker from a retired pid would be indistinguishable from
        a live room.
    """
    factory = MarkerFactory(frame_id, stamp)
    rng = np.random.default_rng() if rng is None else rng
    array = MarkerArray()
    array.markers.append(Marker(action=Marker.DELETEALL))
    array.markers.extend(room_fill_markers(factory, state, rng))
    array.markers.extend(room_polygon_markers(factory, state))
    array.markers.extend(skeleton_markers(factory, state, rng))
    array.markers.extend(room_node_markers(factory, state))
    array.markers.extend(room_text_markers(factory, state))
    array.markers.extend(door_markers(factory, state))
    array.markers.extend(room_edge_markers(factory, state))
    array.markers.extend(object_markers(factory, state))
    array.markers.extend(room_object_edge_markers(factory, state))
    return array
