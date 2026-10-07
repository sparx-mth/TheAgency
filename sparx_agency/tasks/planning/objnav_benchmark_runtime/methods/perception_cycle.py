"""Continuous raw perception, conservative semantic fusion, and fresh target evidence."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import time
import numpy as np
import requests

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import DETECTOR
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import DetectorFrameError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import deduplicate_detections
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import clipped_box, observed_objects, STAIR_LABELS


#: Two observations more than this apart in height are different objects
#: stacked in plan view (a television on a cabinet), never one vote.
HEIGHT_TOLERANCE_M = 0.35


def associate_landmark(policy, xyz, radius_m=None, exclude=(), class_name=None):
    """The map landmark an observation at ``xyz`` belongs to, or None.

    The landmark map associates in plan view -- same class only (the
    ObjectNav wiring, ``class_votes=False``: a cup never joins the table it
    stands on), by centroid radius or footprint overlap; this adds the one
    check it cannot make -- the object's height against the landmark's first
    measured height -- and takes the first candidate that passes. Returns
    ``(landmark, co_located)``: ``co_located`` says whether some same-class
    landmark shared the footprint at another height.
    """
    p = policy
    candidates = p.landmarks.matches(tuple(float(v) for v in xyz[:2]), radius_m, class_name=class_name, exclude=exclude)
    for landmark in candidates:
        anchor = p._object_geometry.get(landmark.id)
        if anchor is None or abs(anchor[2] - xyz[2]) <= HEIGHT_TOLERANCE_M:
            return landmark, bool(candidates)
    return None, bool(candidates)


def contradicted_by_map(policy, label, xyz, radius_m=None):
    """A confirmed landmark here whose plurality class the target does not accept.

    Only under class voting: with classes never merging (the ObjectNav
    wiring since 2026-10-07) a box of another class at a landmark's spot is
    a second object, not a vote against the first, so the map has no
    contradiction to make and this returns None. Under voting, a ``sofa``
    box projecting onto a bed the map has confirmed five times is a vote the
    bed outweighs, not a sofa. Returns that landmark, or None when the map
    does not object.
    """
    p = policy
    if not p.landmarks.class_votes:
        return None
    landmark, _ = associate_landmark(p, xyz, radius_m, class_name=label)
    if landmark is None or not p.landmarks.is_confirmed(landmark):
        return None
    if p.target.accepts(landmark.class_name) or landmark.class_name == label:
        return None
    return landmark


class PerceptionCycle:
    def __init__(self, policy):
        self.policy = policy
        self.step = None
        self.raw = self.detections = ()
        self.projections = []
        self.counts = Counter()
        self.detector_evidence = {}

    def observe(self, obs):
        """Exactly one synchronous prediction for this RGB/depth/pose tuple.

        A detector that fails (HTTP timeout, dead service, malformed reply) is
        recorded on the policy's exploration fallback, which arms a doubling
        back-off; until it ends the frame carries no detections and the
        search keeps moving instead of waiting a timeout per action. The
        skipped frames are counted, never passed off as empty scenes.
        """
        if self.step == obs.step:
            return
        p = self.policy
        self.step = obs.step
        self.projections = []
        self.counts["raw_frames"] += 1
        if p.fallback.service_unavailable(DETECTOR, obs.step):
            self.raw = self.detections = ()
            self.detector_evidence = {"skipped": "detector back-off until action %d" % p.fallback.retry_step[DETECTOR]}
            self.counts["detector_skipped"] += 1
            return
        started = time.monotonic()
        try:
            self.raw = tuple(p.detector.detect(obs.rgb))
        except Exception as exc:  # the detector service, not this frame's geometry
            if isinstance(exc, ObjNavInternalError) and not isinstance(
                    exc.__cause__, (requests.RequestException, DetectorFrameError, ValueError, KeyError)):
                # Not a transport or frame failure: the service's vocabulary, model or
                # configuration changed under the evaluation. That ends the run; a back-off
                # would fly on. A malformed reply for ONE frame is a frame failure.
                raise
            retry = p.fallback.note_service_failure(obs, DETECTOR, exc)
            self.raw = self.detections = ()
            self.detector_evidence = {"failed": "%s: %s" % (type(exc).__name__, exc), "retry_step": retry}
            self.counts["detector_failures"] += 1
            return
        finally:
            p.telemetry.latencies["detector_http"].append((time.monotonic() - started) * 1000)
        p.fallback.note_service_success(DETECTOR)
        self.detector_evidence = deepcopy(getattr(p.detector, "last_diagnostics", {}))
        self.detections = tuple(deduplicate_detections(self.raw))
        p._duplicates_removed += len(self.raw) - len(self.detections)
        if p.building:
            p.building.semantic_stair_hint = any(d.cls in STAIR_LABELS and d.conf >= p.settings.detection_confidence for d in self.raw)

    def fuse(self, obs):
        p = self.policy
        self.observe(obs)
        if obs.step - p._target_step > p.target_settings.max_unseen_steps:
            p._target_id = p._target_xy = p._target_floor_id = None
        committed = p.building is not None and p.building.committed
        normal = p.camera_control.normal_view(obs.pose)
        allowed = not committed and normal
        if allowed:
            p.doors.update(obs, self.raw)
        confirmed = False
        used = set()
        self.projections = []
        for label, xyz in observed_objects(obs, self.detections, p.settings.detection_confidence, self.projections):
            row = self.projections[-1]
            if label in DOOR_LABELS or label in STAIR_LABELS:
                row["status"] = "terrain_or_portal_context"
                continue
            reason = "transition_in_progress" if committed else "inspection_view" if not normal else self._floor_support(xyz)
            if reason is not None:
                row["status"] = reason
                continue
            # The landmark map owns the plan-view association -- same class only; the
            # height check here keeps a second chair on a mezzanine off the one below.
            radius = row.get("radius_m")
            landmark, co_located = associate_landmark(p, xyz, radius, exclude=used, class_name=label)
            if landmark is None and co_located:
                row["status"] = "inconsistent_3d_association"      # co-located in plan view, another height
                continue
            if landmark is None and associate_landmark(p, xyz, radius, class_name=label)[0] is not None:
                # The only landmark it belongs to -- class, footprint AND height -- was fed
                # by this frame already: a second box of the same object.
                row["status"] = "same_frame_association"
                continue
            before = landmark.class_name if landmark is not None else None
            landmark = p.landmarks.observe(label, tuple(float(v) for v in xyz[:2]), frame_id=obs.step,
                                           radius_m=radius, landmark=landmark)
            p._object_geometry.setdefault(landmark.id, tuple(float(v) for v in xyz))
            used.add(landmark.id)
            row.update(status="fused", floor_id=p.mapping.floor_id, landmark_id=landmark.id,
                       landmark_class=landmark.class_name, votes=dict(landmark.votes))
            if before is not None and landmark.class_name != before:
                self.counts["landmark_relabels"] += 1
            if landmark.class_name != label:
                self.counts["outvoted_detections"] += 1          # class voting only; never with classes kept apart
            # Evidence for the target is the landmark's class -- the box's own, with
            # classes never merging (under voting, the plurality: a sofa box on a
            # confirmed bed was the misidentification). And a box clipped at the image
            # edge is a partial view (the same gate the takeover applies): the
            # Ranchester upstairs run spent nine actions pursuing a 48 px sliver the
            # takeover had rightly refused.
            closing = getattr(p, "closing", None)
            if closing is not None and closing.active:
                # The takeover owns the target while it is active; the map keeps mapping
                # the frames it sees, but the legacy pursuit records nothing.
                row["target_evidence"] = "takeover_active"
                continue
            if (p.target.accepts(label) and p.target.accepts(landmark.class_name)
                    and row["confidence"] >= p.settings.target_closing.confidence
                    and not p.target_evidence.is_suppressed(landmark.id, obs.step)):
                if clipped_box(row["xyxy"], obs.camera.intrinsics, p.settings.target_closing.border_margin_px):
                    row["target_evidence"] = "border_clipped"
                    self.counts["border_clipped_evidence"] += 1
                    continue
                # What the takeover refuses to start on, the legacy pursuit does not walk
                # after either: a released far candidate, one seen during the release
                # cooldown, one seen from a spot given up for want of a path.
                if closing is not None and closing.refuses_far_candidate(obs, landmark.xy):
                    row["target_evidence"] = "refused_by_takeover"
                    self.counts["refused_evidence"] += 1
                    continue
                supported = p.target_evidence.observe(landmark, obs.pose, obs.step)
                if p._target_id in (None, landmark.id):
                    p._target_id, p._target_xy, p._target_step = landmark.id, landmark.xy, obs.step
                    p._target_floor_id = p.mapping.floor_id
                    confirmed |= supported
        self.counts.update(row["status"] for row in self.projections)
        return bool(confirmed and allowed and p._target_floor_id == p.mapping.floor_id)

    def _floor_support(self, xyz):
        p = self.policy
        if p.building is None:
            return None
        height = p.mapping._anchor
        if height is None:
            return None                       # no frame integrated yet: nothing to judge the floor by
        if not height - 0.30 <= xyz[2] <= height + 2.2:
            return "wrong_floor_height"
        terrain = p.building.terrain
        near = [value[0] for (x, y, _), value in terrain.samples.items()
                if value[2] and math.hypot((x + 0.5) * terrain.resolution - xyz[0],
                                          (y + 0.5) * terrain.resolution - xyz[1]) <= 0.75
                and value[0] <= xyz[2] + 0.1]
        if len(near) >= 6:
            floor = min(near)
            return None if height - 0.40 <= floor <= height + 0.18 else "unsupported_floor_association"
        # A frontal object can occlude the floor itself. Previously observed
        # free cells near its surface support membership; unknown is not free.
        world = p.mapping.worlds[p.mapping.floor_id]
        x, y = world.world_to_grid(*xyz[:2])
        r = int(math.ceil(0.35 / world.resolution))
        if not world.in_bounds(x, y):
            return "outside_observed_map"
        patch = world.grid[max(0, y-r):y+r+1, max(0, x-r):x+r+1]
        return None if np.count_nonzero(patch == world.values.free) >= 3 else "unobserved_floor_support"

    def diagnostics(self):
        return {"observation_step": self.step, "counts": dict(self.counts),
                "projections": list(self.projections), "detector_evidence": self.detector_evidence}

