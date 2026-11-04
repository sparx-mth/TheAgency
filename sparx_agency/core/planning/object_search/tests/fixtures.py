"""A self-contained rig for the method: the benchmark's camera and actions, fake services, synthetic frames.

Nothing here imports the benchmark runtime. The camera and the action set are
the Gibson ObjectNav protocol's numbers written out (640 x 480 at a 79-degree
horizontal field, 0.88 m up, 0.5-5 m of depth; 0.25 m steps and 30-degree
turns), so the method is exercised on the geometry it is tuned for without
the adapter that owns those numbers. The fakes answer every service role
neutrally: the detector returns one fixed box of one class, the LLM labels
every room ``living_room`` and gives every node an equal share.

Deliberate duplication (house rule): the benchmark runtime's regression rig
(``tasks/planning/objnav_benchmark_runtime/tests/test_method.py``) carries
its own copies of these fakes. Test modules are not a library -- importing
another suite's fixtures couples the two, and a change to theirs would
silently change what this one checks -- so each suite keeps its own.
"""
from __future__ import annotations

import re

import numpy as np

from sparx_agency.core.planning.object_search.interfaces import Detection
from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import ALL_ACTIONS, DiscreteActionSpec, NAVIGATION_ACTIONS
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def benchmark_camera() -> CameraSpec:
    """The registered RGB-D pinhole camera of the Gibson ObjectNav protocol."""
    return CameraSpec(intrinsics_from_hfov(640, 480, 79.0), 0.88, 0.5, 5.0)


def body_actions(tilt: bool = False) -> DiscreteActionSpec:
    """The four body actions (0.25 m, 30 degrees); with ``tilt`` the LOOK pair too (30 degrees, -30..60)."""
    if not tilt:
        return DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0, tilt_angle_deg=30.0,
                                  actions=NAVIGATION_ACTIONS)
    return DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0, tilt_angle_deg=30.0,
                              min_pitch_deg=-30.0, max_pitch_deg=60.0, actions=ALL_ACTIONS)


class FakeLLM:
    """Every model role at once -- room classifier and node oracle -- all neutral."""

    def __init__(self):
        self.reasoning_calls = 0
        self.calls = 0

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        if kwargs.get("reasoning"):
            self.reasoning_calls += 1
        if "Room observed objects" in user:
            return {"label": "living_room", "confidence": 0.8, "reasoning": "objects"}
        ids = [int(nid) for nid in re.findall(r"^id=(\d+)", user, re.MULTILINE)]
        share = max(1, int(round(90 / max(1, len(ids)))))
        return {"nodes": [{"id": nid, "why": "no opinion", "p": share} for nid in ids],
                "elsewhere": max(0, 100 - share * len(ids))}


class FakeDetector:
    """One fixed box of one class on every frame (twice when ``duplicate``)."""

    def __init__(self, label="chair", confidence=0.9, duplicate=False):
        self.label, self.confidence, self.duplicate = label, confidence, duplicate
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        row = Detection(self.label, self.confidence, (280.0, 200.0, 360.0, 280.0))
        return [row, row] if self.duplicate else [row]


def episode(actions=None, metadata=None) -> ObjNavEpisode:
    """One synthetic chair episode on the benchmark camera."""
    return ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", "chair", benchmark_camera(),
                         actions or body_actions(), 500, metadata=dict(metadata or {}))


def make_policy(detector=None, llm=None, actions=None, warmed_up=True, **overrides):
    """A reset policy on a 20 m synthetic map; ``overrides`` are ``RPTSettings`` fields.

    With ``warmed_up`` (the default) the mandatory warm-up rotation is marked
    done and the spawn room needs no discovery peek, so the first ``plan``
    reaches the room reasoning; ``warmed_up=False`` starts the episode as the
    benchmark does, with the look-around.
    """
    overrides.setdefault("doorway_peek", {"enabled": False})
    settings = RPTSettings(**dict({"map_size_m": 20.0}, **overrides))
    policy = RPTSearchPolicy(detector or FakeDetector(), llm or FakeLLM(), settings)
    ep = episode(actions)
    policy.reset(ep, gibson_label_mapper().target_labels("chair"))
    if warmed_up:
        policy.warmup_actions = policy.settings.warmup_steps
        policy._warmup_pending = False
        policy.peek.floor_ready = lambda: True
    return policy, ep


def frame(ep, step, depth=0.7, pose=None) -> ObjNavObservation:
    """A black RGB frame over a flat depth image at ``pose`` (the origin, facing +x, by default)."""
    k = ep.camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8),
                             np.full((k.height, k.width), depth, np.float32),
                             pose or AgentPose(0, 0, 0, 0), ep.camera, "chair", step)
