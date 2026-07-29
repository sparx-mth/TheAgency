"""The :class:`NavigationPolicy` contract shared by every VLA in this package.

One request type in, one result type out, both frozen, both carrying an
extension dict -- the same shape as ``PlanRequest``/``PlanResult`` and
``TrackerRequest``/``TrackerResult`` elsewhere in ``core/planning``.

Why an ABC and not a Protocol
-----------------------------
The pluggable *pipeline stages* in this repo (planner, smoother, tracker,
behavior) are ``typing.Protocol``. The swappable *backends* -- ``DepthModel``,
``DetectionModel``, ``PathCorrector``, ``BoxTracker`` -- are ``abc.ABC`` with
``raise NotImplementedError`` bodies. A VLA is the second kind: several concrete
implementations of one job, selected by name at runtime. So it follows that
convention.

What this contract deliberately does NOT do
-------------------------------------------
It does not replace the per-policy clients. ``NavDPPointgoalClient`` and
``FlowNavImageGoalClient`` keep their native, policy-shaped APIs, because the
FALCON ROS1 nodes call them directly and depend on their exact behaviour. This
contract is the *uniform* view on top, for callers that want to be told "here is
a policy, drive it" without knowing which one -- an arbiter switching policies, a
benchmark sweeping all of them, or a new robot's runner node.

Python 3.8 compatible.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.vlas.interfaces.goals import Goal


@dataclass(frozen=True)
class PolicyObservation:
    """One egocentric observation handed to a policy.

    Provide the subset the policy needs -- an image-goal policy ignores ``depth``
    and ``intrinsics`` entirely, a point-goal policy requires both. A policy
    raises if a field it needs is ``None``; it never substitutes a zero array,
    because a silently-zeroed depth channel is a wrong trajectory that looks
    plausible.

    Attributes:
        rgb: HxWx3 uint8 image in **RGB** order (not OpenCV's BGR).
        depth_m: HxW float32 metric depth in metres, aligned to ``rgb``.
        intrinsics: camera model the frames were captured with.
        altitude_m: height above the ground plane, where the policy renders its
            trajectory against one.
        metadata: free-form extras (timestamps, frame ids, ...).
    """
    rgb: Optional[np.ndarray] = None
    depth_m: Optional[np.ndarray] = None
    intrinsics: Optional[Intrinsics] = None
    altitude_m: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyResult:
    """What a policy returns for one observation.

    Attributes:
        trajectory: ``(T, >=2)`` float32 body-frame **FLU** waypoints --
            column 0 forward (+x), column 1 left (+y), optional column 2 yaw.
            ``None`` when the policy declined to act (see ``ok``).
        score: the policy's own confidence in ``trajectory`` where it exposes one
            (NavDP's critic value); ``None`` where it does not (FlowNav).
        stop: the policy is asking to stop rather than continue.
        metadata: free-form extras -- the full sample fan-out, per-sample values,
            an overlay mask, inference latency.
    """
    trajectory: Optional[np.ndarray] = None
    score: Optional[float] = None
    stop: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self):
        """True when a flyable trajectory is present.

        Matches ``PlanResult.ok`` / ``LocalPlanOutput.ok`` elsewhere in
        ``core/planning`` so callers read the same way across stages.
        """
        return self.trajectory is not None and len(self.trajectory) > 0


class NavigationPolicy(abc.ABC):
    """A learned policy: observation + goal -> body-frame trajectory.

    Attributes:
        name: the registry key, e.g. ``"navdp"``.
        accepts: the goal classes this policy understands. Checked by
            :meth:`check_goal` so a wrong modality fails at wire-up with a clear
            message instead of at inference with a shape error.
    """

    name = ""
    accepts = ()

    def check_goal(self, goal):
        """Raise unless ``goal`` is a modality this policy accepts.

        Args:
            goal: the :class:`Goal` about to be passed to :meth:`step`.

        Raises:
            TypeError: the policy cannot consume this goal modality.
        """
        if not isinstance(goal, tuple(self.accepts)):
            raise TypeError(
                "%s cannot take a %s; it accepts %s"
                % (self.name or type(self).__name__, type(goal).__name__,
                   ", ".join(g.__name__ for g in self.accepts)))

    @abc.abstractmethod
    def reset(self, observation=None):
        """Clear any per-episode state (frame history, server-side context).

        Call when the goal changes or a new episode starts.

        Args:
            observation: an optional first observation, for policies that need
                the camera model up front (NavDP resets with the intrinsics).

        Returns:
            ``True`` if the policy is ready to step.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, observation, goal):
        """Produce a trajectory for one observation.

        Args:
            observation: the current :class:`PolicyObservation`.
            goal: a :class:`Goal` of a type in :attr:`accepts`.

        Returns:
            A :class:`PolicyResult`. A dropped inference (transport error at
            video rate) returns a result with ``ok == False`` rather than
            raising -- the caller re-sends on the next frame. A response that
            *arrived* but cannot be flown raises, because returning "no result"
            there would let the caller keep flying a stale path.

        Raises:
            TypeError: ``goal`` is a modality this policy does not accept.
            VlaError: the response was malformed, or the runtime failed.
        """
        raise NotImplementedError
