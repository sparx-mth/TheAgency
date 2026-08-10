"""When the policy will not move, turn until it can see somewhere to go.

A forward-looking RGB-D policy answers about the view it is given. Point it at a
wall, or away from the goal, and "stop" is not a failure -- it is the correct
answer to the question asked. The failure is asking the same question forever.

That is exactly what a well-behaved hold produces: the aircraft stops, so the
view stops changing, so the answer stops changing. Measured in the office scene,
a mission that spawns with the goal 116 degrees off the nose deadlocked both the
pretrained and the fine-tuned policy for the entire flight budget -- 50
consecutive predictions of zero length, from a camera that never once looked at
the goal. Before position-holding was fixed the aircraft used to drift out of it
by accident, which is not a mechanism anyone should rely on.

So: hold the position and sweep the heading. Face the goal first, because that is
where a route is most likely to exist; if the policy still declines after looking
there for a moment, widen the search either side, and keep widening until
something is proposed or the aircraft has looked everywhere. Rotating on the spot
costs nothing, risks nothing, and is what a pilot does.

This is policy-agnostic and ROS-free: it decides where to *look*, and the caller
decides how to get there (rate-limited, in every stack here). Python 3.8 idioms,
and this module itself pulls in no numpy -- though importing it through the
package does, since its siblings need it. The FALCON Noetic adapter can import
it either way; numpy 1.17 is present in that container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, radians
from typing import Optional, Tuple

from sparx_agency.core.common.types.geometry import normalize_angle

QUARTER = pi / 2.0

DEFAULT_OFFSETS = (0.0, radians(45.0), radians(-45.0), radians(90.0),
                   radians(-90.0), radians(135.0), radians(-135.0), pi)
"""Where to look, relative to the bearing to the goal, in order.

Straight at the goal first: on an open route that is the only heading that
matters, and the sweep never leaves it. Then alternating either side so the
search stays as close to the goal direction as it can for as long as it can --
a monotonic sweep would spend its first several steps looking further and
further from where the aircraft wants to go. It ends looking backwards, which is
the right last resort for a goal that can only be reached by going round."""


@dataclass
class YawSearchSpec:
    """How to look around when nothing is flyable.

    Attributes:
        offsets: Headings to try, as offsets from the bearing to the goal.
        dwell_s: How long to sit on a heading before giving up on it. Must cover
            at least one inference or the sweep outruns the policy and never
            actually asks from any of the headings it visits.
        aligned_rad: How close counts as looking that way. The dwell clock only
            runs once the aircraft has arrived, so a slow turn does not consume
            the time meant for asking.
    """

    offsets: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_OFFSETS)
    dwell_s: float = 1.5
    aligned_rad: float = radians(12.0)

    def __post_init__(self):
        if not self.offsets:
            raise ValueError("a yaw search needs at least one heading to try")
        if self.dwell_s <= 0.0:
            raise ValueError("dwell_s must be positive; got %r" % (self.dwell_s,))


class YawSearch(object):
    """Sweeps the heading while the policy has nothing to offer.

    Call :meth:`heading` every control step the aircraft is stuck, and
    :meth:`reset` the moment it is not -- the sweep is a response to being
    stuck, and a route means it worked.
    """

    def __init__(self, spec: Optional[YawSearchSpec] = None) -> None:
        self.spec = spec or YawSearchSpec()
        self.index = 0
        self.sweeps = 0
        self._arrived_s = None      # type: Optional[float]

    def reset(self) -> None:
        """Back to looking at the goal. Call this as soon as a route exists."""
        self.index = 0
        self.sweeps = 0
        self._arrived_s = None

    @property
    def offset(self) -> float:
        """The offset from the goal bearing currently being tried."""
        return self.spec.offsets[self.index]

    def heading(self, goal_bearing: float, current_yaw: float,
                now_s: float) -> float:
        """Where to point, world radians.

        Args:
            goal_bearing: World bearing from the aircraft to its goal.
            current_yaw: Where the aircraft is actually pointing.
            now_s: Clock, same timebase throughout.

        Returns:
            The heading to turn to. Stable until the aircraft has held it for
            ``dwell_s``, then the next one in the sweep.
        """
        target = normalize_angle(float(goal_bearing) + self.offset)
        if abs(normalize_angle(target - float(current_yaw))) > self.spec.aligned_rad:
            # Still turning: the dwell has not started, so a long turn never
            # counts as time spent asking from the new heading.
            self._arrived_s = None
            return target
        if self._arrived_s is None:
            self._arrived_s = float(now_s)
        elif float(now_s) - self._arrived_s >= self.spec.dwell_s:
            self.index += 1
            if self.index >= len(self.spec.offsets):
                self.index = 0
                self.sweeps += 1
            self._arrived_s = None
            target = normalize_angle(float(goal_bearing) + self.offset)
        return target
