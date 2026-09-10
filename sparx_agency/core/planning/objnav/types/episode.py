"""What the agent may know about an episode before its first step.

Only public task information crosses this boundary: which scene, which object
category, the camera, the action geometry and the step budget. Goal positions,
geodesic distances and the start-to-goal path length are **privileged** --
they stay inside the environment and reach the harness through
:class:`~sparx_agency.core.planning.objnav.types.measurement.EpisodeMeasurement`,
never the agent. A benchmark number produced by an agent that could see them
would be meaningless, and nothing would flag it.

The typed fields have no place for them, but ``metadata`` is a free-form
mapping handed to the policy unchanged, and nothing here can tell a harmless
extra from a leaked one. Privileged values must never be put in it -- a
dataset row copied wholesale carries ``shortest_path`` and friends. The
adapter is responsible. The harness's reset check refuses the known leak keys,
but it is a tripwire for those keys, not a guarantee.

Python 3.8 syntax; numpy arrives only through the camera's intrinsics type.
"""
from __future__ import annotations

import collections.abc
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.camera import CameraSpec


@dataclass(frozen=True)
class ObjNavEpisode:
    """The public description of one episode.

    Attributes:
        episode_id: Unique within ``(benchmark, split)``.
        scene_id: The scene (building) the episode runs in.
        benchmark: Benchmark key, e.g. ``"hm3d_v2"`` -- it names the dataset
            and its protocol, and selects the label mapper.
        split: Dataset split, e.g. ``"val"``.
        target_category: The goal category in the dataset's own vocabulary,
            verbatim.
        camera: The agent's camera for the whole episode.
        action_spec: The geometry of the discrete actions.
        max_steps: The step budget. The environment ends the episode once
            this many actions have been executed.
        metadata: Non-privileged extras: a mapping, stored as a dict copy.
            Never goal positions, distances or paths -- the adapter keeps them
            out; nothing here can tell.

    Raises:
        ObjNavError: On an identifier that is not a non-blank string (a
            string of spaces is blank: it would merge result rows), a wrong
            type for the camera or action spec, a non-positive step budget,
            or metadata that is not a mapping.
    """

    episode_id: str
    scene_id: str
    benchmark: str
    split: str
    target_category: str
    camera: CameraSpec
    action_spec: DiscreteActionSpec
    max_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("episode_id", "scene_id", "benchmark", "split",
                     "target_category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ObjNavError(
                    "ObjNavEpisode.%s must be a non-blank string, got %r"
                    % (name, value))
        if not isinstance(self.metadata, collections.abc.Mapping):
            raise ObjNavError(
                "ObjNavEpisode.metadata must be a mapping of non-privileged "
                "extras, got %r" % (self.metadata,))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not isinstance(self.camera, CameraSpec):
            raise ObjNavError(
                "ObjNavEpisode.camera must be a CameraSpec, got %r"
                % (self.camera,))
        if not isinstance(self.action_spec, DiscreteActionSpec):
            raise ObjNavError(
                "ObjNavEpisode.action_spec must be a DiscreteActionSpec, got "
                "%r" % (self.action_spec,))
        if (not isinstance(self.max_steps, numbers.Integral)
                or isinstance(self.max_steps, bool) or self.max_steps <= 0):
            raise ObjNavError(
                "ObjNavEpisode.max_steps must be a positive integer, got %r"
                % (self.max_steps,))
