"""What a search policy wants the agent to do next, in continuous world geometry.

This is the seam between *where* and *how*. The policy -- the scene graph, the
LLM's room probabilities and RPT* -- decides where to go and says so as a path
of world points. The action converter turns that into one discrete action per
step. So a policy never emits a :class:`DiscreteAction`, and the converter
never decides where the target is.

Four kinds of command, one type:

* **follow** a path of waypoints (the last one is the goal), optionally
  re-aiming the camera and facing a heading at the end -- and optionally
  carrying the exact body actions that walk it (``actions``, from the pose
  ``plan_pose``), which the converter executes verbatim while the agent is
  where the plan expects it and falls back from to the path otherwise;
* **hold** position while re-aiming the camera or turning to a heading;
* **stop** -- the policy believes it is at the target and ends the episode;
* nothing at all (an empty hold): the policy has no opinion this step.

A stop carries nothing else: it ends the episode on this step, so waypoints, a
pitch or a heading would never be executed ("stop, but first walk there" sent
as one stop is how an agent calls STOP a metre short). A policy that wants to
stop once a follow or a hold is done says so in advance with
``stop_on_arrival``: the converter then emits STOP on the step the command is
fully executed, and not one step before.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from sparx_agency.core.planning.objnav.errors import CommandError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.angles import MAX_ANGLE_RAD

#: A world-ENU point ``(x, y)``, metres.
Waypoint = Tuple[float, float]
#: A planned pose ``(x, y, yaw)``: world-ENU metres and radians.
PlanPose = Tuple[float, float, float]
#: The actions a plan may carry: the body moves. A LOOK is the camera pitch's business, a STOP the policy's.
PLAN_ACTIONS = (DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT)


def _finite_real(value) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value))


def _float_pairs(waypoints) -> Tuple[Waypoint, ...]:
    """``waypoints`` as a tuple of finite float pairs, or CommandError naming the bad one."""
    points = []
    for index, point in enumerate(waypoints):
        try:
            x, y = point
        except (TypeError, ValueError):
            raise CommandError(
                "waypoint %d must be an (x, y) pair, got %r" % (index, point))
        if not (_finite_real(x) and _finite_real(y)):
            raise CommandError(
                "waypoint %d must be finite, got %r" % (index, point))
        points.append((float(x), float(y)))
    return tuple(points)


def _check_angle(name: str, value) -> None:
    """Raise unless ``value`` is None or a finite angle within :data:`MAX_ANGLE_RAD`."""
    if value is None:
        return
    if not _finite_real(value):
        raise CommandError("%s must be None or finite, got %r" % (name, value))
    if abs(value) > MAX_ANGLE_RAD:
        raise CommandError(
            "%s=%r rad exceeds %g rad in magnitude; wrap it (normalize_angle) "
            "before building the command -- wrapping an angle this large "
            "would hang the converter" % (name, value, MAX_ANGLE_RAD))


def _info(info: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """A classmethod's ``info`` argument for the constructor, which checks and copies it."""
    return {} if info is None else info


def _plan_actions(actions) -> Tuple[DiscreteAction, ...]:
    """``actions`` as a tuple of body actions, or CommandError naming the bad one."""
    out = []
    for index, action in enumerate(actions):
        if not isinstance(action, DiscreteAction) or action not in PLAN_ACTIONS:
            raise CommandError(
                "planned action %d must be MOVE_FORWARD, TURN_LEFT or TURN_RIGHT, got %r -- a LOOK is the "
                "camera pitch's business and a STOP the policy's" % (index, action))
        out.append(action)
    return tuple(out)


def _plan_pose(pose) -> Optional[PlanPose]:
    """``pose`` as a finite ``(x, y, yaw)`` triple, or None, or CommandError."""
    if pose is None:
        return None
    try:
        x, y, yaw = pose
    except (TypeError, ValueError):
        raise CommandError("plan_pose must be an (x, y, yaw) triple, got %r" % (pose,))
    if not (_finite_real(x) and _finite_real(y) and _finite_real(yaw)):
        raise CommandError("plan_pose must be finite, got %r" % (pose,))
    _check_angle("plan_pose yaw", yaw)
    return (float(x), float(y), float(yaw))


@dataclass(frozen=True)
class NavigationCommand:
    """One step's instruction from the search policy to the action converter.

    Build one with :meth:`follow`, :meth:`hold` or :meth:`stop_here`; the
    constructor is the validating form they all go through.

    Attributes:
        waypoints: Ordered world-ENU ``(x, y)`` points, metres. The last is
            the goal. Empty when the command is a hold or a stop.
        stop: End the episode now. Must be a real ``bool`` -- a string such as
            ``"false"`` is refused, because ``bool("false")`` is True.
        camera_pitch: Desired camera pitch, radians, REP-103 (positive looks
            down), or None to leave the camera where it is.
        final_yaw: Heading to face once the last waypoint is reached (or at
            once, for a hold), radians, or None for no requirement.
        info: Free-form per-step diagnostics from the policy: a mapping,
            stored as a dict copy. Never interpreted. The headless agent
            returns a copy of it with that step's decision to the caller of
            ``act()``; the harness does not store it -- only the agent's
            ``episode_info()`` reaches the results row.
        stop_on_arrival: Emit STOP on the step this command is fully executed
            -- the path walked, the pitch reached, the heading faced -- instead
            of reporting it arrived. A real ``bool``. It exists because the
            step after a satisfied command is spent on the headless agent's
            idle action, a turn, which undoes a facing: on RoboTHOR the object
            must be in view in the STOP frame, so a ``stop_here()`` sent one
            step later stops a turn too late. It is still the policy's
            decision -- asked for in advance, taken on exactly the step the
            command completes.
        actions: The body actions that walk the path from ``plan_pose``, in
            order, as a kinematic planner emitted them (since 2026-10-28):
            MOVE_FORWARD, TURN_LEFT and TURN_RIGHT only. The converter
            executes the first one when the agent stands at ``plan_pose``
            (within its tolerances) and otherwise follows the waypoints as
            if no plan were given; the policy, which sees every executed
            action, sends the remaining suffix next step. Empty for no plan.
        plan_pose: The pose ``(x, y, yaw)`` the actions start from. Required
            with ``actions``; None without.

    Raises:
        CommandError: On a waypoint that is not a finite ``(x, y)`` pair; a
            ``stop`` or ``stop_on_arrival`` that is not a bool; a non-finite
            angle, or one beyond ``MAX_ANGLE_RAD``; an ``info`` that is not a
            mapping; a stop that also carries waypoints, a pitch, a heading or
            ``stop_on_arrival``; ``stop_on_arrival`` on a command that asks
            for nothing else (that is :meth:`stop_here`); a planned action
            that is not a body move; actions without waypoints or without
            ``plan_pose``; or a ``plan_pose`` without actions.
    """

    waypoints: Tuple[Waypoint, ...] = ()
    stop: bool = False
    camera_pitch: Optional[float] = None
    final_yaw: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
    stop_on_arrival: bool = False
    actions: Tuple[DiscreteAction, ...] = ()
    plan_pose: Optional[PlanPose] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", _float_pairs(self.waypoints))
        object.__setattr__(self, "actions", _plan_actions(self.actions))
        object.__setattr__(self, "plan_pose", _plan_pose(self.plan_pose))
        if self.actions and not self.waypoints:
            raise CommandError("planned actions walk a path: a command with actions needs waypoints")
        if bool(self.actions) != (self.plan_pose is not None):
            raise CommandError("actions and plan_pose come together: the actions start from that pose")
        for name in ("stop", "stop_on_arrival"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise CommandError(
                    "%s must be a bool, got %r -- bool('false') is True, so a "
                    "string is never coerced" % (name, value))
        for name in ("camera_pitch", "final_yaw"):
            _check_angle(name, getattr(self, name))
        if not isinstance(self.info, collections.abc.Mapping):
            raise CommandError(
                "NavigationCommand.info must be a mapping of diagnostics, got "
                "%r" % (self.info,))
        object.__setattr__(self, "info", dict(self.info))
        asks = bool(self.waypoints or self.camera_pitch is not None
                    or self.final_yaw is not None)
        if self.stop and asks:
            raise CommandError(
                "a stop command carries nothing else: it ends the episode on "
                "this step, so waypoints, a pitch or a heading would never be "
                "executed")
        if self.stop and self.stop_on_arrival:
            raise CommandError(
                "stop_on_arrival on a stop command: the stop already ends the "
                "episode on this step; send stop_here() alone")
        if self.stop_on_arrival and not asks:
            raise CommandError(
                "stop_on_arrival on a command that asks for nothing else is a "
                "stop; send NavigationCommand.stop_here()")

    @classmethod
    def follow(cls, points, camera_pitch: Optional[float] = None,
               final_yaw: Optional[float] = None,
               info: Optional[Mapping[str, Any]] = None,
               stop_on_arrival: bool = False,
               actions=(), plan_pose=None) -> NavigationCommand:
        """Follow a path.

        Args:
            points: The path, as ``(x, y)`` pairs (tuples, lists, numpy rows),
                as objects with ``.x`` and ``.y`` (``Pose2D``), or as a
                ``Path2D``. World ENU, metres. At least one point.
            camera_pitch: Desired camera pitch, radians, REP-103.
            final_yaw: Heading to face at the end, radians.
            info: Diagnostics returned with the step's decision.
            stop_on_arrival: STOP on the step the path is walked (and the
                pitch and heading reached), rather than idling there.
            actions: The planned body actions that walk the path (see the
                class), or none.
            plan_pose: The ``(x, y, yaw)`` they start from; None without them.

        Returns:
            The command.

        Raises:
            CommandError: On an empty path, a malformed point, or anything the
                constructor refuses.
        """
        raw = getattr(points, "points", points)
        pairs = []
        for index, point in enumerate(raw):
            if hasattr(point, "x") and hasattr(point, "y"):
                pairs.append((point.x, point.y))
                continue
            try:
                # The first two columns: a (N, 3) path with a yaw column is
                # common in this repo, and its heading is not a waypoint.
                pairs.append((point[0], point[1]))
            except (TypeError, IndexError, KeyError):
                raise CommandError(
                    "waypoint %d must be an (x, y) pair or have .x and .y, "
                    "got %r" % (index, point))
        if not pairs:
            raise CommandError(
                "follow() needs at least one waypoint; use hold() to stay")
        return cls(waypoints=tuple(pairs), camera_pitch=camera_pitch,
                   final_yaw=final_yaw, info=_info(info),
                   stop_on_arrival=stop_on_arrival, actions=tuple(actions), plan_pose=plan_pose)

    @classmethod
    def hold(cls, camera_pitch: Optional[float] = None,
             final_yaw: Optional[float] = None,
             info: Optional[Mapping[str, Any]] = None,
             stop_on_arrival: bool = False) -> NavigationCommand:
        """Stay here, optionally re-aiming the camera or turning to a heading.

        With neither argument this is the empty command: the policy has
        nothing to ask for this step. A hold that is already satisfied is
        followed by the headless agent's idle action, a turn, which moves the
        heading off again; to stop facing the heading, pass
        ``stop_on_arrival=True`` (with a pitch or a heading to reach).
        """
        return cls(camera_pitch=camera_pitch, final_yaw=final_yaw,
                   info=_info(info), stop_on_arrival=stop_on_arrival)

    @classmethod
    def stop_here(cls, info: Optional[Mapping[str, Any]] = None
                  ) -> NavigationCommand:
        """End the episode: the policy believes the target is reached."""
        return cls(stop=True, info=_info(info))
