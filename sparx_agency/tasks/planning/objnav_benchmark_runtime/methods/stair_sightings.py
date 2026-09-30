"""A perfect stair detector: a bounding box for a staircase only when the staircase is in the frame.

"Ground truth about the stairs" means this and nothing more: when a
staircase's surface is in the current RGB-D frame -- inside the image, within
the depth sensor's range and not hidden behind something nearer -- the policy
receives a bounding box that identifies it exactly, the way a perfect
detector would. When it is not in the frame the policy receives nothing, and
knows nothing: a staircase it has never seen is not a portal, not a node of
the search, and never the explanation of a change of height. The connectors'
geometry (:mod:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.stair_connectors`)
is the detector's INSTRUMENT, not the policy's map; it reaches the policy one
sighting at a time.

The test is geometric, against the same depth image every other detection is
projected with: each of the connector's surface samples is projected into the
camera (:func:`~sparx_agency.core.planning.objnav.camera_geometry.project_to_image`);
a sample counts as visible when its pixel is inside the image, its optical
depth is inside the sensor's range, and the depth image at that pixel is not
nearer than the sample by more than :data:`DEPTH_SLACK_M` (the navmesh sits a
few centimetres above the rendered treads). :data:`MIN_VISIBLE_POINTS`
visible samples make a sighting, whose box is the padded extent of their
pixels and whose footprint is those samples in the world -- what the map
draws where the stairs are.

:class:`StairSightings` remembers every staircase seen so far. It is the one
place the building coordinator asks "do we know these stairs?".
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import Connector, GroundTruthStairs

Xyz = Tuple[float, float, float]

#: The fewest surface samples that must pass the depth test for a staircase to be in the frame.
MIN_VISIBLE_POINTS = 6
#: The depth image may read this much NEARER than a navmesh sample and the sample still shows.
DEPTH_SLACK_M = 0.20
#: Padding around the visible samples' pixels, pixels.
BOX_MARGIN_PX = 4
#: Spacing of the polyline samples stood in for a connector that carries no surface sample, metres.
POLYLINE_SPACING_M = 0.20
#: The most footprint points kept per staircase for the map.
MAX_FOOTPRINT_POINTS = 300
#: Footprint points nearer than this are one point, metres.
FOOTPRINT_SPACING_M = 0.20


@dataclass(frozen=True)
class StairSighting:
    """One staircase seen in one frame.

    Attributes:
        connector_id: Which staircase (the scene's connector id).
        step: The observation it was seen in.
        bbox: ``(x1, y1, x2, y2)`` pixel box around the visible surface.
        visible: Surface samples that passed the depth test.
        tested: Surface samples that were inside the image and in range.
        distance_m: Optical depth of the nearest visible sample, metres.
        points: The visible samples, world ENU.
    """

    connector_id: int
    step: int
    bbox: Tuple[int, int, int, int]
    visible: int
    tested: int
    distance_m: float
    points: Tuple[Xyz, ...]

    def as_detection(self) -> Dict:
        """The sighting as the frame overlay and the trace carry detections: a labelled box."""
        return {"label": "stairs", "confidence": 1.0, "xyxy": [int(v) for v in self.bbox],
                "connector_id": int(self.connector_id), "distance_m": round(float(self.distance_m), 2),
                "visible_points": int(self.visible), "source": "ground_truth"}


def _densified(polyline: Sequence[Xyz], spacing_m: float) -> np.ndarray:
    """Points along an ENU polyline at most ``spacing_m`` apart in plan view; endpoints kept."""
    out: List[np.ndarray] = []
    for a, b in zip(polyline, polyline[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        count = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / spacing_m)))
        for index in range(count):
            out.append(a + (b - a) * (index / count))
    out.append(np.asarray(polyline[-1], dtype=float))
    return np.asarray(out, dtype=float).reshape(-1, 3)


class GroundTruthStairDetector:
    """Says which staircases are in the frame, and where in it.

    Args:
        stairs: The scene's connectors.
        min_visible: Visible surface samples needed for a sighting.
        depth_slack_m: See :data:`DEPTH_SLACK_M`.
    """

    def __init__(self, stairs: GroundTruthStairs, min_visible: int = MIN_VISIBLE_POINTS,
                 depth_slack_m: float = DEPTH_SLACK_M):
        self.min_visible = int(min_visible)
        self.depth_slack_m = float(depth_slack_m)
        self._points: Dict[int, np.ndarray] = {}
        for connector in stairs.connectors:
            self._points[connector.id] = self._surface_of(connector)

    @staticmethod
    def _surface_of(connector: Connector) -> np.ndarray:
        surface = getattr(connector, "surface", ()) or ()
        if len(surface) >= MIN_VISIBLE_POINTS:
            return np.asarray([list(p[:3]) for p in surface], dtype=float).reshape(-1, 3)
        return _densified(connector.polyline, POLYLINE_SPACING_M)

    def detect(self, obs) -> List[StairSighting]:
        """The staircases in this frame, nearest first; empty when none is."""
        depth = np.asarray(obs.depth_m)
        camera = obs.camera
        height, width = depth.shape
        sightings: List[StairSighting] = []
        for connector_id, points in self._points.items():
            if not len(points):
                continue
            pixels, z = project_to_image(points, camera, obs.pose)
            with np.errstate(invalid="ignore"):
                in_range = np.isfinite(z) & (z > camera.min_depth_m) & (z < camera.max_depth_m)
                finite_px = np.all(np.isfinite(pixels), axis=1)
            candidate = in_range & finite_px
            if int(candidate.sum()) < self.min_visible:
                continue
            safe = np.where(finite_px[:, None], pixels, -1.0)     # a non-finite pixel is off the image, not a cast error
            u = np.rint(safe[:, 0]).astype(int)
            v = np.rint(safe[:, 1]).astype(int)
            inside = candidate & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            tested = int(inside.sum())
            if tested < self.min_visible:
                continue
            measured = depth[v[inside], u[inside]]
            with np.errstate(invalid="ignore"):
                visible_mask = np.isfinite(measured) & (measured >= z[inside] - self.depth_slack_m)
            visible = int(visible_mask.sum())
            if visible < self.min_visible:
                continue
            seen_px = pixels[inside][visible_mask]
            seen_z = z[inside][visible_mask]
            seen_points = points[inside][visible_mask]
            x1 = max(0, int(math.floor(seen_px[:, 0].min())) - BOX_MARGIN_PX)
            y1 = max(0, int(math.floor(seen_px[:, 1].min())) - BOX_MARGIN_PX)
            x2 = min(width - 1, int(math.ceil(seen_px[:, 0].max())) + BOX_MARGIN_PX)
            y2 = min(height - 1, int(math.ceil(seen_px[:, 1].max())) + BOX_MARGIN_PX)
            sightings.append(StairSighting(
                connector_id=int(connector_id), step=int(obs.step), bbox=(x1, y1, x2, y2),
                visible=visible, tested=tested, distance_m=float(seen_z.min()),
                points=tuple(tuple(float(c) for c in p) for p in seen_points)))
        sightings.sort(key=lambda s: s.distance_m)
        return sightings


class StairSightings:
    """Every staircase seen so far, and what was seen of it."""

    def __init__(self):
        self.records: Dict[int, Dict] = {}
        self.latest: List[StairSighting] = []

    def note(self, sighting: StairSighting) -> bool:
        """Record a sighting. True when it is the first of that staircase."""
        record = self.records.get(sighting.connector_id)
        first = record is None
        if first:
            record = {"connector_id": int(sighting.connector_id), "first_step": int(sighting.step),
                      "last_step": int(sighting.step), "sightings": 0, "nearest_m": math.inf,
                      "most_visible": 0, "footprint": []}
            self.records[sighting.connector_id] = record
        record["last_step"] = int(sighting.step)
        record["sightings"] += 1
        record["nearest_m"] = min(record["nearest_m"], float(sighting.distance_m))
        record["most_visible"] = max(record["most_visible"], int(sighting.visible))
        self._extend_footprint(record, sighting.points)
        return first

    def observe(self, sightings: Sequence[StairSighting]) -> List[StairSighting]:
        """Record this frame's sightings; the ones that were a staircase's first."""
        self.latest = list(sightings)
        return [s for s in sightings if self.note(s)]

    def mark_seen(self, connector_id: int, step: int, points: Sequence[Xyz] = ()) -> bool:
        """Count a staircase as seen without a frame -- a test, or a policy handed the knowledge outright."""
        return self.note(StairSighting(int(connector_id), int(step), (0, 0, 0, 0), 0, 0, 0.0,
                                       tuple(tuple(float(c) for c in p) for p in points)))

    def seen(self, connector_id: int) -> bool:
        return int(connector_id) in self.records

    def footprint(self, connector_id: int) -> List[Xyz]:
        """The visible surface of a staircase, world ENU, deduplicated; empty when never seen."""
        record = self.records.get(int(connector_id))
        return list(record["footprint"]) if record else []

    def _extend_footprint(self, record: Dict, points: Sequence[Xyz]) -> None:
        kept = record["footprint"]
        for point in points:
            if len(kept) >= MAX_FOOTPRINT_POINTS:
                break
            if all(math.dist(point[:2], q[:2]) > FOOTPRINT_SPACING_M or abs(point[2] - q[2]) > FOOTPRINT_SPACING_M
                   for q in kept):
                kept.append(tuple(float(c) for c in point))

    def diagnostics(self) -> Dict:
        records = []
        for record in self.records.values():
            row = {k: (None if v == math.inf else v) for k, v in record.items() if k != "footprint"}
            row["footprint_points"] = len(record["footprint"])
            records.append(row)
        return {"seen": sorted(self.records), "records": records,
                "latest": [s.as_detection() for s in self.latest]}


