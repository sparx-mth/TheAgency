"""How long ago the aircraft last looked in a given direction.

A forward-facing depth camera observes a wedge of the world, and the aircraft is
free to move in directions that wedge does not cover. Every reflex in this stack
that moves sideways or backwards -- the contact retreat, the map-gate back-out,
the unstick -- does exactly that, at cruise speed, into space nothing has looked
at since before the manoeuvre began. The depth brake cannot help there: it caps
forward speed by the nearest return in the corridor, which is a statement about
what the camera CAN see.

This is the missing half of that: not "what is in front of me" but "how stale is
my knowledge of the direction I am about to move in". It is deliberately a memory
of *looking*, not of *contents* -- a bearing counts as observed when the camera
was pointed along it and returned a frame, whatever that frame contained, because
the thing being bounded is the age of the evidence rather than its verdict.

Bearings are world-frame radians. The aircraft's yaw plus the camera's horizontal
half field of view defines the wedge marked observed on each frame.
"""

import math


class ObservationMemoryConfig(object):
    """Geometry and resolution of the observation memory.

    Args:
        sectors: How many equal bearing bins cover the full circle. 24 gives
            15 degrees each, comfortably finer than the wedge a single frame
            marks and coarser than the yaw noise between frames.
        half_fov_rad: Half the camera's horizontal field of view. The SJTU
            drone's depth camera is 1.3098 rad wide, so 0.655 rad.
        margin_rad: Shrink applied to the wedge before marking it observed, so
            that a bearing is only credited when it was comfortably inside the
            frame rather than clipping its edge.
    """

    def __init__(self, sectors=24, half_fov_rad=0.655, margin_rad=0.1):
        if sectors < 4:
            raise ValueError("sectors must be at least 4, got %r" % (sectors,))
        if half_fov_rad <= 0.0:
            raise ValueError(
                "half_fov_rad must be positive, got %r" % (half_fov_rad,))
        if margin_rad < 0.0:
            raise ValueError("margin_rad must not be negative, got %r"
                             % (margin_rad,))
        if margin_rad >= half_fov_rad:
            raise ValueError(
                "margin_rad %r must be smaller than half_fov_rad %r; a wedge "
                "shrunk to nothing would never mark any bearing observed"
                % (margin_rad, half_fov_rad))
        self.sectors = int(sectors)
        self.half_fov_rad = float(half_fov_rad)
        self.margin_rad = float(margin_rad)


class ObservationMemory(object):
    """Last-observed timestamp per bearing sector.

    Nothing is assumed observed at construction: every bearing starts unseen,
    so an aircraft that has not yet taken a frame is treated as knowing nothing,
    which is what it is.
    """

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or ObservationMemoryConfig()
        self._seen = [None] * self._cfg.sectors  # type: list

    @property
    def config(self):
        # type: () -> object
        return self._cfg

    def _sector(self, bearing_rad):
        # type: (float) -> int
        span = 2.0 * math.pi / self._cfg.sectors
        return int(math.floor((bearing_rad % (2.0 * math.pi)) / span)) \
            % self._cfg.sectors

    def observe(self, yaw_rad, now_s):
        # type: (float, float) -> None
        """Mark the camera's wedge, centred on ``yaw_rad``, seen at ``now_s``."""
        reach = self._cfg.half_fov_rad - self._cfg.margin_rad
        # Step by less than a sector so no sector inside the wedge is skipped.
        span = 2.0 * math.pi / self._cfg.sectors
        step = span * 0.5
        offset = -reach
        while offset <= reach + 1e-9:
            self._seen[self._sector(yaw_rad + offset)] = float(now_s)
            offset += step
        # The wedge edges themselves, so a narrow wedge still marks its ends.
        self._seen[self._sector(yaw_rad - reach)] = float(now_s)
        self._seen[self._sector(yaw_rad + reach)] = float(now_s)

    def age(self, bearing_rad, now_s):
        # type: (float, float) -> object
        """Seconds since ``bearing_rad`` was last observed, or None if never.

        A timestamp in the future -- the clock was reset under us, which happens
        when a simulation restarts -- reads as freshly observed rather than as
        infinitely stale, because the alternative brakes the aircraft for a
        bookkeeping artefact.
        """
        seen = self._seen[self._sector(bearing_rad)]
        if seen is None:
            return None
        age = float(now_s) - seen
        return 0.0 if age < 0.0 else age

    def is_stale(self, bearing_rad, now_s, max_age_s):
        # type: (float, float, float) -> bool
        """Whether knowledge of ``bearing_rad`` is older than ``max_age_s``.

        Never-observed counts as stale: the aircraft has no evidence at all
        about that direction, which is the case this exists for.
        """
        age = self.age(bearing_rad, now_s)
        return True if age is None else age > float(max_age_s)

    def reset(self):
        # type: () -> None
        """Forget everything. For a respawn, which starts with no evidence."""
        self._seen = [None] * self._cfg.sectors
