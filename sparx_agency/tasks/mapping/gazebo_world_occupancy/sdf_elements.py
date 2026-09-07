"""Read the small values out of SDF XML: a pose, and a child element's text.

SDF hides two conventions in these few lines, and both are easy to get subtly
wrong. A ``<pose>`` is ``x y z roll pitch yaw`` with **extrinsic XYZ**
rotations, and an absent or blank pose means identity rather than an error --
most elements in a real world file have no pose at all.

Newer SDF revisions add attributes that re-interpret those same six numbers
(``degrees``, ``rotation_format``, ``relative_to``). None is implemented here,
and each is refused rather than ignored, because ignoring one parses cleanly
and puts the model in the wrong place.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np

POSE_COMPONENTS = 6
UNSUPPORTED_POSE_ATTRIBUTES = ("degrees", "rotation_format", "relative_to")


def pose_to_matrix(values: Sequence[float]) -> np.ndarray:
    """Convert an SDF ``<pose>`` to a 4x4 homogeneous transform.

    SDF poses are ``x y z roll pitch yaw`` with extrinsic XYZ rotations, i.e.
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Args:
        values: Six numbers.

    Returns:
        ``(4, 4)`` float64 transform.

    Raises:
        ValueError: If ``values`` is not six numbers.
    """
    numbers = [float(v) for v in values]
    if len(numbers) != POSE_COMPONENTS:
        raise ValueError("pose must have 6 components, got %d" % (len(numbers),))
    x, y, z, roll, pitch, yaw = numbers
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [
        [
            cos_y * cos_p,
            cos_y * sin_p * sin_r - sin_y * cos_r,
            cos_y * sin_p * cos_r + sin_y * sin_r,
        ],
        [
            sin_y * cos_p,
            sin_y * sin_p * sin_r + cos_y * cos_r,
            sin_y * sin_p * cos_r - cos_y * sin_r,
        ],
        [-sin_p, cos_p * sin_r, cos_p * cos_r],
    ]
    transform[:3, 3] = (x, y, z)
    return transform


def element_pose(element: ET.Element) -> np.ndarray:
    """Return an element's own ``<pose>`` as a transform.

    Args:
        element: Any SDF element that may carry a ``<pose>`` child.

    Returns:
        The transform, or identity when there is no pose or it is blank.

    Raises:
        ValueError: If the ``<pose>`` carries an attribute that changes how
            its numbers must be read; see
            :data:`UNSUPPORTED_POSE_ATTRIBUTES`.
    """
    pose = element.find("pose")
    if pose is None:
        return np.eye(4, dtype=np.float64)
    _reject_unsupported_attributes(pose)
    if not (pose.text or "").strip():
        return np.eye(4, dtype=np.float64)
    return pose_to_matrix((pose.text or "").split())


def _reject_unsupported_attributes(pose: ET.Element) -> None:
    """Refuse the pose attributes this reader would otherwise misread.

    SDF 1.9's ``degrees="true"`` and ``rotation_format="quat_xyzw"``, and SDF
    1.7's ``relative_to``, each change what the numbers mean while leaving six
    of them in the element -- so the parse succeeds, the model lands somewhere
    else in the world, and the map is wrong with nothing to show for it. A
    world that uses them needs the feature implemented, not guessed at.

    Args:
        pose: The ``<pose>`` element.

    Raises:
        ValueError: If any unsupported attribute is present, whatever its
            value.
    """
    present = [
        name for name in UNSUPPORTED_POSE_ATTRIBUTES if pose.get(name) is not None
    ]
    if not present:
        return
    raise ValueError(
        "<pose> carries unsupported attribute(s) %s; this reader implements "
        "only SDF's default six numbers, 'x y z roll pitch yaw' in radians "
        "and relative to the parent frame" % (", ".join(present),)
    )


def child_text(element: ET.Element, tag: str, default: str = "") -> str:
    """Return the stripped text of a child tag.

    Args:
        element: Parent element.
        tag: Child tag name.
        default: Returned when the child is absent or empty.

    Returns:
        The text, or ``default``.
    """
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()
