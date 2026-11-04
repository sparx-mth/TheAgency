"""Continuous raw perception, conservative semantic fusion, and fresh target evidence."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import time
import numpy as np

from sparx_agency.core.mapping.objects.box_extent import clipped_box
from sparx_agency.core.mapping.objects.detection_dedup import deduplicate_detections
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import DETECTOR, SEGMENTER
from sparx_agency.core.planning.object_search.interfaces import (
    RECOVERABLE_DETECTOR_CAUSES, RECOVERABLE_SEGMENTER_CAUSES)
from sparx_agency.core.planning.object_search.perception.doors import DOOR_LABELS
from sparx_agency.core.planning.object_search.perception.projection import observed_objects, STAIR_LABELS
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError


#: Two observations more than this apart in height are different objects
#: stacked in plan view (a television on a cabinet), never one vote.
HEIGHT_TOLERANCE_M = 0.35
#: Retraction events kept for the record; the counters are complete.
MAX_RETRACTION_EVENTS = 50


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
        # ``detection -> HxW bool mask`` for the target's boxes when the ``segmentation``
        # setting is on (``None`` when it is off): the one input that distinguishes the
        # two pipelines. A box mapped to None was due a mask the service did not deliver.
        self.masks = None
        self.segmenter_evidence = {}
        #: Landmarks retracted on clear views without their class (``_count_misses``), for the record.
        self.retractions = []
        #: landmark id -> the pose ``(x, y, yaw)`` of its last counted miss: the next must be a distinct viewpoint.
        self._miss_poses = {}

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
        self.masks = {} if p.settings.segmentation else None
        self.segmenter_evidence = {}
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
                    exc.__cause__, RECOVERABLE_DETECTOR_CAUSES):
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
        if p.settings.segmentation:
            self._segment(obs)

    def _segment(self, obs):
        """One segmenter call for the frame's target boxes; a failure leaves them due a mask they do not get.

        Only the target's boxes are prompted: theirs is the depth that ends an
        episode. Their masks restrict the projection to the object's pixels
        (``perception.observed_objects``). On a transport or frame failure
        the back-off is armed like the detector's, every target box of the
        frame is mapped to None -- ``segmentation_unavailable``, not a box
        read -- and the frame is counted; identity drift ends the run.
        """
        p = self.policy
        # Every target box the closing may read -- ``track_confidence`` re-sightings included -- gets a mask.
        floor = min(p.settings.detection_confidence, p.settings.target_closing.track_confidence)
        targets = [d for d in self.detections if p.target.accepts(d.cls) and d.conf >= floor]
        if not targets:
            return
        if p.fallback.service_unavailable(SEGMENTER, obs.step):
            self.masks = {d: None for d in targets}
            self.segmenter_evidence = {"skipped": "segmenter back-off until action %d" % p.fallback.retry_step[SEGMENTER]}
            self.counts["segmenter_skipped"] += 1
            return
        started = time.monotonic()
        try:
            masks = p.segmenter.segment(obs.rgb, [d.xyxy for d in targets])
        except Exception as exc:
            if isinstance(exc, ObjNavInternalError) and not isinstance(
                    exc.__cause__, RECOVERABLE_SEGMENTER_CAUSES):
                raise
            retry = p.fallback.note_service_failure(obs, SEGMENTER, exc)
            self.masks = {d: None for d in targets}
            self.segmenter_evidence = {"failed": "%s: %s" % (type(exc).__name__, exc), "retry_step": retry}
            self.counts["segmenter_failures"] += 1
            return
        finally:
            p.telemetry.latencies["segmenter_http"].append((time.monotonic() - started) * 1000)
        p.fallback.note_service_success(SEGMENTER)
        self.masks = dict(zip(targets, masks))
        self.segmenter_evidence = {"boxes": len(targets), "mask_pixels": [int(np.count_nonzero(m)) for m in masks],
                                   "ms": getattr(p.segmenter, "last_inference_ms", None)}
        self.counts["segmented_frames"] += 1
        self.counts["segmented_boxes"] += len(targets)

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
        for label, xyz in observed_objects(obs, self.detections, p.settings.detection_confidence, self.projections,
                                           masks=self.masks):
            row = self.projections[-1]
            if row.get("occluder_depth_m") is not None:
                self.counts["occluder_reads"] += 1               # the box read around a surface in front of the object
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
            self._miss_poses.pop(landmark.id, None)              # seen: the miss ledger starts afresh
            p._object_geometry.setdefault(landmark.id, tuple(float(v) for v in xyz))
            # The best confidence the landmark was ever observed at: a single sighting at the
            # closing's threshold makes a target landmark a node (``opening_nodes.target_landmarks``).
            p._landmark_best_conf[landmark.id] = max(p._landmark_best_conf.get(landmark.id, 0.0), float(row["confidence"]))
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
        if allowed:
            self._count_misses(obs, used)
        return bool(confirmed and allowed and p._target_floor_id == p.mapping.floor_id)

    def _count_misses(self, obs, used):
        """Tell the map which confirmed landmarks stood in a clear view this frame and got no detection.

        The negative evidence the one-way confirmation lacked (2026-10-21):
        a landmark is in a clear view when its anchor projects inside the
        frame by ``landmark_miss_margin_px``, between the miss range bounds,
        and the depth at its pixel reaches at least as far as the landmark
        -- nothing measured in front of it. A depth reading short of it is
        an occluder; an invalid reading (beyond the sensor's range, a hole
        in the mesh) is no view at all. Only in a normal, uncommitted view:
        a frame the detections themselves are not fused from counts for
        nothing against them either. ``used`` holds the landmarks this
        frame's detections fed. A miss counts only from a viewpoint
        distinct from the last counted one -- ``landmark_miss_viewpoint_m``
        of travel or 30 degrees of turn -- so a bad angle held for four
        frames is one miss, not four. The map retracts after ``max_misses``
        or the landmark's own observation count, whichever is more.
        """
        p = self.policy
        landmarks = p.landmarks
        if landmarks.max_misses <= 0:
            return
        s = p.settings
        candidates = [lm for lm in landmarks.confirmed() if lm.id not in used and lm.id in p._object_geometry]
        if not candidates:
            return
        anchors = np.asarray([p._object_geometry[lm.id] for lm in candidates], dtype=float)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or not np.isfinite(anchors).all():
            return
        pixels, depths = project_to_image(anchors, obs.camera, obs.pose)
        k = obs.camera.intrinsics
        margin = int(s.landmark_miss_margin_px)
        depth_image = obs.depth_m
        here = (float(obs.pose.x), float(obs.pose.y), float(obs.pose.yaw))
        for lm, (u, v), z in zip(candidates, pixels, depths):
            last = self._miss_poses.get(lm.id)
            if last is not None and (math.hypot(here[0] - last[0], here[1] - last[1]) < s.landmark_miss_viewpoint_m
                                     and abs(math.remainder(here[2] - last[2], 2 * math.pi)) < math.radians(30.0)):
                continue                          # the same viewpoint as the last counted miss
            if not (np.isfinite(z) and s.landmark_miss_min_range_m <= z <= s.landmark_miss_max_range_m):
                continue
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            col, row = int(round(u)), int(round(v))
            if not (margin <= col < k.width - margin and margin <= row < k.height - margin):
                continue
            # The sensor's word on the line of sight: the median reading around the pixel.
            patch = depth_image[max(0, row - 2):row + 3, max(0, col - 2):col + 3]
            readings = patch[np.isfinite(patch) & (patch > float(obs.camera.min_depth_m))
                             & (patch < float(obs.camera.max_depth_m))]
            if len(readings) < 3:
                continue                          # no measurement there: not a view of the spot
            if float(np.median(readings)) < float(z) - 0.5:
                self.counts["landmark_views_occluded"] += 1
                continue                          # something stands between the camera and the landmark
            before = lm.misses
            retracted = landmarks.note_missed(lm.id, frame_id=obs.step)
            if lm.misses > before:
                self.counts["landmark_misses"] += 1
                self._miss_poses[lm.id] = here
            if retracted:
                self.counts["landmark_retractions"] += 1
                if len(self.retractions) < MAX_RETRACTION_EVENTS:
                    self.retractions.append({"step": int(obs.step), "landmark_id": int(lm.id), "class": lm.class_name,
                                             "count": int(lm.count), "misses": int(lm.misses), "range_m": round(float(z), 2),
                                             "xy": [round(float(c), 2) for c in lm.xy]})

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
            # The samples within 0.75 m of the object. The floor itself when the camera saw it;
            # the object's own base or bowl when it did not (a toilet seen from a metre with the
            # floor in front inside the camera's blind zone and its base under the frame's lower
            # edge read 0.2-0.45 m here, Collierville/000023, 2026-10-14). A storey is 2.5 m and
            # more away: a sample within 0.60 m of this storey's floor level is this storey's floor,
            # and ONE such sample near the object puts the object on it -- the LOWEST sample used
            # to decide, and the stairwell beside the same toilet (its top tread 0.75 m from the
            # bowl) refused it on every frame of the afternoon's run.
            if any(height - 0.40 <= z <= height + 0.60 for z in near):
                return None
            return "unsupported_floor_association"
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
                "projections": list(self.projections), "detector_evidence": self.detector_evidence,
                "segmenter_evidence": self.segmenter_evidence, "retractions": list(self.retractions)}

