"""What the agent receives at every step: a clean RGB-D frame, a perfect pose, the target.

Ground truth throughout: the simulator's rendered depth, not a network's
estimate, and the simulator's pose, not a localiser's. That is what lets a
benchmark run isolate the search logic from perception and localisation error.

An observation is self-describing -- it carries the camera it was captured
with -- so mapping code never has to reach back into the episode to interpret
a pixel. It is validated on construction, because the cheapest place to catch
an adapter that swapped height and width, forgot to drop an alpha channel or
handed over millimetres as integers is before a single pixel is read.

Treat the arrays as read-only: they are shared, not copied.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObservationError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose


@dataclass(frozen=True, eq=False)
class ObjNavObservation:
    # eq=False: a generated __eq__ would compare the images element-wise and
    # raise on the ambiguous truth value of the result. Identity is the only
    # meaningful equality for one step's frame.
    """One step's input to an ObjectNav agent.

    Attributes:
        rgb: ``(H, W, 3)`` uint8 image in **RGB** order.
        depth_m: ``(H, W)`` floating-point depth in metres, optical-frame
            ``z``; ``NaN`` for no reading, ``+inf`` for no surface in range
            (see :mod:`~sparx_agency.core.planning.objnav.types.camera`).
        pose: The ground-truth pose at the moment of capture.
        camera: The camera both images were captured with.
        target_category: The episode's goal category, in the dataset's own
            vocabulary (``"tv_monitor"``, ``"Television"``).
        step: 0 for the observation ``reset`` returns, then one more per
            executed action.

    Raises:
        ObservationError: If an image's shape or type disagrees with the
            camera, the target is empty, or the step index is negative.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    pose: AgentPose
    camera: CameraSpec
    target_category: str
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.pose, AgentPose):
            raise ObservationError(
                "observation pose must be an AgentPose, got %r" % (self.pose,))
        if not isinstance(self.camera, CameraSpec):
            raise ObservationError(
                "observation camera must be a CameraSpec, got %r"
                % (self.camera,))
        size = (self.camera.intrinsics.height, self.camera.intrinsics.width)
        rgb = self.rgb
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.shape != size + (3,)):
            raise ObservationError(
                "rgb must be a %r uint8 array in RGB order, got %s"
                % (size + (3,), _describe(rgb)))
        depth = self.depth_m
        if (not isinstance(depth, np.ndarray) or depth.shape != size
                or not np.issubdtype(depth.dtype, np.floating)):
            raise ObservationError(
                "depth_m must be a %r floating-point array in metres, got %s"
                % (size, _describe(depth)))
        if not isinstance(self.target_category, str) or not self.target_category:
            raise ObservationError(
                "target_category must be a non-empty string, got %r"
                % (self.target_category,))
        if (not isinstance(self.step, numbers.Integral)
                or isinstance(self.step, bool) or self.step < 0):
            raise ObservationError(
                "step must be a non-negative integer, got %r" % (self.step,))


def _describe(array) -> str:
    """Shape and dtype of an array, or its type when it is not one."""
    if isinstance(array, np.ndarray):
        return "shape %r dtype %s" % (array.shape, array.dtype)
    return type(array).__name__
