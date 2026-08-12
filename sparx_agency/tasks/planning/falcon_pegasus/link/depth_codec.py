"""Turn Isaac Sim's depth buffer into the exact bytes FALCON's mapper reads.

Isaac's ``distance_to_image_plane`` annotator returns float32 **metres**, already
the perpendicular distance to the image plane (optical-frame z), which is what a
pinhole back-projection wants -- not ray length. FALCON's mapper reads uint16
**millimetres** and treats 0 as "no return". Converting here, on the side with a
modern numpy and CPU headroom, halves the bytes on the wire and skips a
per-frame ``convertTo`` inside the mapper.

Three classes of pixel need deciding on, and each decision changes the map:

**Non-finite.** A ray that leaves through a window or past the far clip comes
back ``inf``; in one office recording 1186 of 1516 frames contained some, up to
54 % of pixels. ``inf`` is *not* missing data -- it is a measurement that nothing
is there -- so it becomes :data:`FAR_M`, and FALCON carves free space along that
ray. ``NaN`` is genuinely missing and becomes 0, which FALCON skips.

**Too near.** Below the camera's minimum range the value is not trustworthy, and
a spurious obstacle 5 cm in front of the lens is one the planner will never get
past. Zeroed.

**Too far.** Clamped to :data:`FAR_M`, which must stay comfortably **above**
FALCON's ``tsdf/raycast_max``. That is the whole trick: FALCON clips the ray at
``raycast_max`` but computes the signed distance from the *reported* range, so a
point clamped beyond that limit carves free space and never lays down a surface.
Clamp at or below ``raycast_max`` instead and every open doorway grows a wall
across it at exactly that distance.

Kept Python 3.8 compatible -- the ROS1 node imports the decoder half.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.tasks.planning.falcon_pegasus.link.protocol import DEPTH_DTYPE

NEAR_M = 0.15
"""Closest range reported. Nearer returns are CLAMPED to it, not discarded.

They used to be zeroed, as "noise or the airframe's own propellers" -- which is
the right rule for a physical sensor with a genuine blind zone, and the wrong
one here. This camera is a ray-cast against the surveyed map: a 0.09 m reading
is not noise, it is a wall.

Zeroing them cost a whole flight. With the aircraft pinned against a partition
the lens sits ~0.1 m off it, so the ENTIRE frame reads under NEAR_M -- measured
min 0.08 / median 0.09 / max 0.13 m -- and every pixel was sent as 0, which on
this wire means "no measurement". FALCON was handed a blank image while its nose
was on the wall, so it never mapped that wall, the space beyond it stayed
UNKNOWN, and it re-planned the same unreachable viewpoint indefinitely. The
aircraft was backed off three times and returned within ten seconds each time.
Clamping instead reports "a surface at least this close", which is true and is
what lets the mapper finally lay the wall down.
"""
FAR_M = 20.0
"""Ceiling for a reported range, metres.

Must exceed FALCON's ``voxel_mapping/tsdf/raycast_max`` (5.0 m by default) --
see this module's docstring for why. Also well under the 65.535 m where uint16
millimetres would wrap.
"""
_MAX_ENCODABLE_M = 65.0


def encode_depth(depth_m, near_m=NEAR_M, far_m=FAR_M):
    # type: (np.ndarray, float, float) -> np.ndarray
    """Convert a float32 metre depth image to uint16 millimetres.

    Args:
        depth_m: ``(height, width)`` float array in metres. ``inf`` means "no
            surface within range"; ``NaN`` means "no measurement".
        near_m: Closest reportable range; nearer returns clamp to it. See
            :data:`NEAR_M` -- discarding them instead hides an obstacle the
            aircraft is touching.
        far_m: Readings above this are clamped to it.

    Returns:
        A ``(height, width)`` little-endian uint16 array in millimetres, with 0
        wherever there is no usable measurement.

    Raises:
        ValueError: If ``far_m`` is not above ``near_m``, or exceeds what uint16
            millimetres can hold.
    """
    if not far_m > near_m:
        raise ValueError("far_m (%r) must be greater than near_m (%r)" % (far_m, near_m))
    if far_m > _MAX_ENCODABLE_M:
        raise ValueError(
            "far_m %r m does not fit in uint16 millimetres (max %r m)"
            % (far_m, _MAX_ENCODABLE_M))

    values = np.asarray(depth_m, dtype=np.float32)
    # +inf is "nothing out there", which is a measurement; NaN is "no reading",
    # which is not. They must not collapse to the same number.
    finite = np.nan_to_num(values, nan=0.0, posinf=far_m, neginf=0.0)
    np.clip(finite, 0.0, far_m, out=finite)
    # A real return closer than near_m clamps UP to it; only a genuine
    # non-measurement (NaN, already 0 above) stays 0. See NEAR_M.
    near = (finite > 0.0) & (finite < near_m)
    finite[near] = near_m
    return np.rint(finite * 1000.0).astype(DEPTH_DTYPE, copy=False)


def decode_depth(payload, width, height):
    # type: (bytes, int, int) -> np.ndarray
    """Rebuild the uint16 millimetre image from the bytes on the wire.

    Args:
        payload: ``height * width`` little-endian uint16 values.
        width: Image width, pixels.
        height: Image height, pixels.

    Returns:
        A ``(height, width)`` uint16 array, millimetres.

    Raises:
        ValueError: If the payload is not exactly the declared size. A short or
            long buffer means the sender and receiver disagree about the camera,
            and reshaping it anyway would produce a sheared image and a map that
            looks plausible.
    """
    expected = width * height * 2
    if len(payload) != expected:
        raise ValueError(
            "depth payload is %d bytes, expected %d for %dx%d uint16"
            % (len(payload), expected, width, height))
    return np.frombuffer(payload, dtype=DEPTH_DTYPE).reshape(height, width)
