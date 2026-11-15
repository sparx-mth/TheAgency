"""The discrete action space every ObjectNav benchmark shares.

Habitat (HM3D, MP3D, Gibson) and AI2-THOR (RoboTHOR) give the agent the same
vocabulary: stop, step forward, turn left or right, and tilt the camera up or
down. The integer values below follow Habitat's ObjectNav action order, so a
Habitat index and a :class:`DiscreteAction` happen to agree -- but an
environment adapter must still map by *name*, never by assuming integer
equality with its simulator. AI2-THOR names and orders its actions
differently, and a benchmark without camera tilt has no LOOK actions at all.

:class:`DiscreteActionSpec` is the geometry of one step as the benchmark
executes it. Its defaults are the values the target papers evaluate with
(0.25 m, 30 degrees, 30 degrees). Every adapter states its own explicitly
rather than inheriting them: a spec that silently disagrees with the
simulator makes every perfect-execution prediction wrong by the same amount,
every step.

Camera pitch follows REP-103: the rotation about the body ``+y`` (left) axis,
so **a positive pitch looks down**. LOOK_DOWN adds ``tilt_angle_deg`` and
LOOK_UP subtracts it. A simulator that signs pitch the other way converts at
its adapter, once: Habitat's sensor pitch is positive *up*; AI2-THOR's camera
horizon is positive down, like this.

Pitch limits model what the simulators do at the edge: Habitat does not clamp
LOOK at all (the limits stay None), while AI2-THOR's LoCoBot *fails* a LOOK
that would pass +/-30 degrees -- the step is spent and the camera does not
move. Either way the pitch never leaves the range, and nothing here ever asks
for a LOOK the simulator would refuse.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

from sparx_agency.core.planning.objnav.errors import ObjNavError


class DiscreteAction(IntEnum):
    """One benchmark action. Values follow Habitat's ObjectNav action order."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


#: Every action, in value order: a benchmark with camera tilt.
ALL_ACTIONS = tuple(DiscreteAction)

#: The four actions every ObjectNav benchmark has.
NAVIGATION_ACTIONS = (
    DiscreteAction.STOP,
    DiscreteAction.MOVE_FORWARD,
    DiscreteAction.TURN_LEFT,
    DiscreteAction.TURN_RIGHT,
)

#: The camera-tilt pair. A benchmark has both or neither.
LOOK_ACTIONS = (DiscreteAction.LOOK_UP, DiscreteAction.LOOK_DOWN)


def _is_real(value) -> bool:
    """A real number that is not a bool (``True`` would otherwise pass as 1)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class DiscreteActionSpec:
    """The geometry of one discrete step, as the benchmark executes it.

    Degrees, because that is what a person writes in a benchmark config; the
    ``*_rad`` properties are what code reads.

    Attributes:
        forward_step_m: How far MOVE_FORWARD advances, metres.
        turn_angle_deg: How far one TURN_LEFT or TURN_RIGHT rotates, degrees.
        tilt_angle_deg: How far one LOOK_UP or LOOK_DOWN tilts the camera,
            degrees.
        min_pitch_deg: The most upward camera pitch the simulator allows
            (REP-103, so upward is negative), or None when it does not clamp.
        max_pitch_deg: The most downward camera pitch the simulator allows, or
            None when it does not clamp.
        actions: The actions the benchmark accepts, stored in value order.
            Nothing in this package ever emits an action outside this set.

    Raises:
        ObjNavError: On a non-positive or non-finite step, an inverted pitch
            range, or an action set that is empty, repeats an action, lacks
            one of the four navigation actions, or has only half the LOOK
            pair.
    """

    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    tilt_angle_deg: float = 30.0
    min_pitch_deg: Optional[float] = None
    max_pitch_deg: Optional[float] = None
    actions: Tuple[DiscreteAction, ...] = ALL_ACTIONS

    def __post_init__(self) -> None:
        for name in ("forward_step_m", "turn_angle_deg", "tilt_angle_deg"):
            value = getattr(self, name)
            if not (_is_real(value) and math.isfinite(value) and value > 0):
                raise ObjNavError(
                    "DiscreteActionSpec.%s must be a positive finite number, "
                    "got %r" % (name, value))
        if self.turn_angle_deg > 180.0:
            raise ObjNavError(
                "DiscreteActionSpec.turn_angle_deg must be at most 180, got %r"
                % (self.turn_angle_deg,))
        if self.tilt_angle_deg > 90.0:
            raise ObjNavError(
                "DiscreteActionSpec.tilt_angle_deg must be at most 90, got %r"
                % (self.tilt_angle_deg,))
        for name in ("min_pitch_deg", "max_pitch_deg"):
            value = getattr(self, name)
            if value is None:
                continue
            if not (_is_real(value) and math.isfinite(value)
                    and abs(value) <= 90.0):
                raise ObjNavError(
                    "DiscreteActionSpec.%s must be None or a finite angle in "
                    "[-90, 90] degrees, got %r" % (name, value))
        if (self.min_pitch_deg is not None and self.max_pitch_deg is not None
                and self.min_pitch_deg >= self.max_pitch_deg):
            raise ObjNavError(
                "DiscreteActionSpec pitch range is inverted: min_pitch_deg=%r "
                "is not below max_pitch_deg=%r (REP-103: up is negative)"
                % (self.min_pitch_deg, self.max_pitch_deg))
        self._check_actions()

    def _check_actions(self) -> None:
        actions = tuple(self.actions)
        for action in actions:
            if not isinstance(action, DiscreteAction):
                raise ObjNavError(
                    "DiscreteActionSpec.actions must hold DiscreteAction "
                    "members, got %r" % (action,))
        if len(set(actions)) != len(actions):
            raise ObjNavError(
                "DiscreteActionSpec.actions repeats an action: %r" % (actions,))
        missing = [a.name for a in NAVIGATION_ACTIONS if a not in actions]
        if missing:
            raise ObjNavError(
                "DiscreteActionSpec.actions lacks %s; every ObjectNav "
                "benchmark has STOP, MOVE_FORWARD, TURN_LEFT and TURN_RIGHT"
                % ", ".join(missing))
        if (DiscreteAction.LOOK_UP in actions) != (
                DiscreteAction.LOOK_DOWN in actions):
            raise ObjNavError(
                "DiscreteActionSpec.actions has only half of the LOOK pair: "
                "%r" % (actions,))
        object.__setattr__(self, "actions", tuple(sorted(actions)))

    @property
    def turn_angle_rad(self) -> float:
        """One TURN, in radians."""
        return math.radians(self.turn_angle_deg)

    @property
    def tilt_angle_rad(self) -> float:
        """One LOOK, in radians."""
        return math.radians(self.tilt_angle_deg)

    @property
    def min_pitch_rad(self) -> Optional[float]:
        """The most upward pitch in radians, or None when unclamped."""
        if self.min_pitch_deg is None:
            return None
        return math.radians(self.min_pitch_deg)

    @property
    def max_pitch_rad(self) -> Optional[float]:
        """The most downward pitch in radians, or None when unclamped."""
        if self.max_pitch_deg is None:
            return None
        return math.radians(self.max_pitch_deg)

    @property
    def has_camera_tilt(self) -> bool:
        """Whether the benchmark has the LOOK_UP / LOOK_DOWN pair."""
        return DiscreteAction.LOOK_UP in self.actions

    def allows(self, action: DiscreteAction) -> bool:
        """Whether the benchmark accepts ``action``.

        A bare integer is never accepted, even one equal to an allowed
        action's value: the integer order is Habitat's, and another
        simulator's ``3`` is a different action. (``IntEnum`` equality would
        otherwise let ``3 in actions`` pass.)
        """
        return isinstance(action, DiscreteAction) and action in self.actions
