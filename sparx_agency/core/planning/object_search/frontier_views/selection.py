"""Choose the one to three past frames that best show a frontier node -- Strategy B: coverage of its cells, greedily.

A frontier node is a stretch of the map's boundary (``nodes.py``); a frame
is a picture with the pose it was taken from (``frame_log``). The frames to
show a model about a node are chosen from geometry alone, before any model
is asked:

* every frame of the node's storey is asked what it shows of every cell of
  the node (``sight.py``): nothing, the cell, or the cell and what lies
  beyond it -- a **quality** in ``[0, 1]`` per frame and cell;
* a set of frames is worth, to the node, the average over its cells of the
  best quality any frame of the set gives the cell -- so a second frame
  counts only for what the first does not already show as well;
* frames are added greedily while the set's worth grows by a fifth
  (:attr:`ViewSettings.min_gain_ratio`), three at most. The first frame is
  the best single view; with qualities of 0 and 1 this is the textbook
  greedy set cover over the node's cells.

Between frames worth the same -- and frames taken while turning on one spot
are worth exactly the same, which is most of a look-around -- the one that
sees past more of the node is taken, then the one with the node nearest the
middle of the picture (what stands beside a doorway is half of what a
picture of it says), then the latest.

A node whose best frame shows too little of it has **no usable view** yet,
and the answer says which gate every frame failed -- too far, not in any
picture, occluded, or seen but too thin -- so a replay can tell a node no
frame could ever show from one a furniture-high obstacle hid.

Each node is chosen for on its own: one question per node is asked later,
so two nodes may be handed the same frame, each with its own window in it.

**The window** of a view is where the node lands in the picture: the box of
the pixels its seen cells' height bands project to. It is what a consumer
marks or crops; a picture of a doorway with nothing saying which doorway is
meant is half a question.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameRecord
from sparx_agency.core.planning.object_search.frontier_views.nodes import FrontierNode
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.frontier_views.sight import (
    OCCLUDED, OUT_OF_FRAME, OUT_OF_RANGE, SEEN, FrameSight, SightScene)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec

#: Why a node has no usable view, by the furthest gate any frame's sight of any of its cells reached.
NO_FRAMES = "no frame of this storey"
TOO_FAR = "farther than the view range from every frame"
NOT_FRAMED = "in no frame's picture"
HIDDEN = "occluded in every frame that faces it"
TOO_THIN = "seen, but too little of it"
_REASON_OF_STAGE = {OUT_OF_RANGE: TOO_FAR, OUT_OF_FRAME: NOT_FRAMED, OCCLUDED: HIDDEN, SEEN: TOO_THIN}


@dataclass(frozen=True)
class FrameView:
    """One chosen frame and what it shows of the node.

    Attributes:
        name: The frame's name in the frame log.
        step: The step it was captured at.
        gain: What adding this frame raised the node's score by (the first
            frame's gain is its own score).
        seen_fraction: The share of the node's cells this frame sees.
        through_fraction: The share it sees past, into the unknown.
        range_m: The median ground distance from its camera to the cells it sees.
        window: ``(u_min, v_min, u_max, v_max)`` in pixels: the box the
            node's seen cells land in, floor to the top of the height band.
    """

    name: str
    step: int
    gain: float
    seen_fraction: float
    through_fraction: float
    range_m: float
    window: Tuple[float, float, float, float]


@dataclass(frozen=True)
class NodeViews:
    """The frames chosen for one node, or why there are none.

    Attributes:
        node_id: The node's sticky id.
        views: The chosen frames, best first; empty when no view is usable.
        score: The average over the node's cells of the best quality the
            chosen frames give each (0 without views).
        seen_fraction: The share of the node's cells some chosen frame sees.
        through_fraction: The share some chosen frame sees past.
        reason: Empty when usable; else one of :data:`NO_FRAMES`,
            :data:`TOO_FAR`, :data:`NOT_FRAMED`, :data:`HIDDEN`,
            :data:`TOO_THIN`.
        best_score: The best single frame's score, usable or not -- what a
            threshold is tuned against.
        frames: How many frames of the storey were asked.
    """

    node_id: int
    views: Tuple[FrameView, ...]
    score: float
    seen_fraction: float
    through_fraction: float
    reason: str
    best_score: float
    frames: int

    @property
    def usable(self) -> bool:
        return bool(self.views)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(view.name for view in self.views)


def storey_frames(records, reference_z: Optional[float], tolerance_m: float) -> List[FrameRecord]:
    """The frames taken on the storey whose floor is at ``reference_z``, in step order; all of them when it is None.

    ``records`` is the frame log's own lookup (``EpisodeFrameLog.records``:
    frame name to record) or any iterable of records.

    Frames taken from one pose show one picture (a blocked step, a hold):
    only the latest is kept, so a tie between identical views goes to the
    most recent and no pose is cast twice.
    """
    if isinstance(records, abc.Mapping):
        records = records.values()
    kept: Dict[Tuple, FrameRecord] = {}
    for record in records:
        pose = record.pose
        if reference_z is not None and abs(float(pose.z) - float(reference_z)) > tolerance_m:
            continue
        key = (round(float(pose.x), 3), round(float(pose.y), 3), round(float(pose.z), 2),
               round(float(pose.yaw), 4), round(float(pose.camera_pitch), 4))
        held = kept.get(key)
        if held is None or record.step > held.step:
            kept[key] = record
    return sorted(kept.values(), key=lambda record: record.step)


def select_views(world: OccupancyGrid2D, nodes: Sequence[FrontierNode], records, camera: CameraSpec,
                 settings: Optional[ViewSettings] = None, settled: Optional[np.ndarray] = None,
                 reference_z: Optional[float] = None) -> Dict[int, NodeViews]:
    """The frames that best show each node, over the map of this decision.

    Args:
        world: The occupancy grid now -- not as it stood when a frame was
            taken: what has been mapped since is as true of the old frame.
        nodes: The nodes to choose frames for; all on ``world``'s storey.
        records: The frames to choose from: the frame log's lookup
            (``EpisodeFrameLog.records``, name to record) or an iterable of
            its records (name, step, capture pose).
        camera: The camera the frames were captured with.
        settings: The selector's geometry and thresholds.
        settled: Optional ``(H, W)`` bool of settled unknown, as the frontier was read with.
        reference_z: The height of the floor under the agent now; frames
            from another storey are left out. None asks every frame.

    Returns:
        ``{node_id: NodeViews}`` for every node given.
    """
    settings = settings or ViewSettings()
    nodes = list(nodes)
    if not nodes:
        return {}
    frames = storey_frames(records, reference_z, settings.storey_tolerance_m)
    sizes = [int(node.rows.size) for node in nodes]
    ends = np.cumsum(sizes)
    scene = SightScene(world, np.concatenate([node.rows for node in nodes]),
                       np.concatenate([node.cols for node in nodes]), camera, settings, settled)
    sights = [scene.sight(record.pose) for record in frames]
    out = {}
    for node, size, end in zip(nodes, sizes, ends):
        out[node.node_id] = _choose(node, frames, sights, slice(int(end) - size, int(end)), settings,
                                    float(camera.intrinsics.cx))
    return out


def _choose(node: FrontierNode, frames: Sequence[FrameRecord], sights: Sequence[FrameSight], cells: slice,
            settings: ViewSettings, centre_column: float) -> NodeViews:
    """Greedy cover of one node's cells by the frames' qualities; see the module docstring."""
    size = cells.stop - cells.start
    if not frames or size == 0:
        return NodeViews(node.node_id, (), 0.0, 0.0, 0.0, NO_FRAMES, 0.0, len(frames))
    quality = np.stack([sight.quality[cells] for sight in sights]).astype(np.float64)
    through = np.stack([sight.through[cells] for sight in sights]) & (quality > 0.0)
    steps = np.asarray([record.step for record in frames])
    # How far off the middle of each frame the node's seen cells lie, in columns: the tie-break between equals.
    columns = np.stack([sight.u[cells] for sight in sights]).astype(np.float64)
    shown = quality > 0.0
    off_centre = np.where(shown, np.abs(columns - centre_column), 0.0).sum(axis=1) / np.maximum(shown.sum(axis=1), 1)
    best = np.zeros(size, dtype=np.float64)
    chosen: List[int] = []
    views: List[FrameView] = []
    best_score = float(quality.sum(axis=1).max()) / size
    score = 0.0
    while len(chosen) < settings.max_frames:
        # A frame already chosen gains nothing over itself, so it is never the pick of a positive gain.
        gain = np.maximum(quality - best[None, :], 0.0).sum(axis=1) / size
        # The largest gain; between equals the frame that sees past more of the node, then the one with the node
        # nearest the middle of its picture, then the latest.
        pick = int(np.lexsort((steps, -np.round(off_centre, 3), through.sum(axis=1), np.round(gain, 9)))[-1])
        seen = quality[pick] > 0.0
        if gain[pick] <= 0.0:
            break
        if not chosen:
            needed = max(settings.min_seen_cells, int(np.ceil(settings.min_seen_fraction * size)))
            if int(seen.sum()) < needed or gain[pick] < settings.min_score:
                break
        elif gain[pick] < max(settings.min_gain, settings.min_gain_ratio * score):
            break
        sight = sights[pick]
        u, top, bottom = sight.u[cells][seen], sight.v_top[cells][seen], sight.v_bottom[cells][seen]
        views.append(FrameView(
            name=frames[pick].name, step=int(frames[pick].step), gain=float(gain[pick]),
            seen_fraction=float(seen.sum()) / size, through_fraction=float(through[pick].sum()) / size,
            range_m=float(np.median(sight.range_m[cells][seen])),
            window=(float(u.min()), float(top.min()), float(u.max()), float(bottom.max()))))
        chosen.append(pick)
        best = np.maximum(best, quality[pick])
        score += float(gain[pick])
    if not views:
        furthest = max(int(sight.stage[cells].max()) for sight in sights)
        return NodeViews(node.node_id, (), 0.0, 0.0, 0.0, _REASON_OF_STAGE[furthest], best_score, len(frames))
    covered = quality[chosen].max(axis=0) > 0.0
    return NodeViews(node.node_id, tuple(views), float(score), float(covered.sum()) / size,
                     float(through[chosen].any(axis=0).sum()) / size, "", best_score, len(frames))


def views_by_frame(selection: Mapping[int, NodeViews]) -> Dict[str, List[int]]:
    """``{frame name: [node ids]}`` over a selection: which nodes were handed the same picture."""
    shared: Dict[str, List[int]] = {}
    for node_id, chosen in selection.items():
        for view in chosen.views:
            shared.setdefault(view.name, []).append(int(node_id))
    return shared
