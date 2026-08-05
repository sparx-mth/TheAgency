"""Goal modalities a VLA policy can be asked to reach.

The policies in this package differ far more in *what they are told to go to*
than in what they emit -- every one of them returns body-frame waypoints, but
NavDP takes a metric point, FlowNav takes a target image, InternVLA-N1 takes a
sentence, and OmniVLA takes any combination. Modelling the goal as a small closed
set of frozen dataclasses is what lets one runner serve all of them: the node
builds a goal, the policy declares which types it accepts, and the mismatch is a
loud error at wire-up time instead of a silent shape bug at inference time.

Frame convention (shared with the whole planning stack)
-------------------------------------------------------
Body **FLU**: ``+x`` forward, ``+y`` left, ``+z`` up, robot at the origin, metres.

Python 3.8 compatible: no PEP 604 unions outside ``from __future__ import
annotations``, no ``dataclass(slots=True)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Goal:
    """Base for every goal modality.

    Attributes:
        metadata: free-form extras a policy or node may attach (e.g. the clicked
            pixel that produced a :class:`PointGoal`, for a server-side overlay).
    """
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PointGoal(Goal):
    """A metric point in the robot body frame. NavDP's native goal.

    Attributes:
        forward_m: ``+x``, metres ahead of the robot.
        left_m: ``+y``, metres to the robot's left.
    """
    forward_m: float = 0.0
    left_m: float = 0.0

    def as_tuple(self):
        """Return ``(forward_m, left_m)``."""
        return (float(self.forward_m), float(self.left_m))


@dataclass(frozen=True)
class ImageGoal(Goal):
    """A target camera view to drive towards. FlowNav's native goal.

    Attributes:
        rgb: HxWx3 uint8 image in **RGB** order (not OpenCV's BGR), or ``None``
            to mean "whatever goal the server already holds" -- FlowNav's server
            can own the goal image so the ROS node never needs the file mounted.
    """
    rgb: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LanguageGoal(Goal):
    """A natural-language instruction. InternVLA-N1's native goal.

    Attributes:
        instruction: the sentence handed to the policy, e.g. ``"go through the
            door on your left and stop at the table"``.
    """
    instruction: str = ""


@dataclass(frozen=True)
class PoseGoal(Goal):
    """A relative pose: a point plus a desired heading change.

    Distinct from :class:`PointGoal` because some policies (OmniVLA) condition on
    the heading difference as well as the displacement.

    Attributes:
        forward_m: ``+x``, metres ahead of the robot.
        left_m: ``+y``, metres to the robot's left.
        heading_delta_rad: desired heading change, radians, CCW-positive.
    """
    forward_m: float = 0.0
    left_m: float = 0.0
    heading_delta_rad: float = 0.0

    def as_tuple(self):
        """Return ``(forward_m, left_m, heading_delta_rad)``."""
        return (float(self.forward_m), float(self.left_m),
                float(self.heading_delta_rad))
