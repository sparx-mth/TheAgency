"""Settings of the narrow-passage navigation layer: the kinematic follower, the scrape guard, the look-down.

Every value here is part of the frozen method configuration
(``RPTSearchPolicy.configuration()``), as the rest of ``RPTSettings`` is.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers


def _check_positive(owner, name, value, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError("%s.%s must be a %s finite number" % (owner, name, "non-negative" if allow_zero else "positive"))


def _check_count(owner, name, value, allow_zero=False):
    if type(value) is not int or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("%s.%s must be a %s integer" % (owner, name, "non-negative" if allow_zero else "positive"))


def _check_bool(owner, name, value):
    if type(value) is not bool:
        raise ValueError("%s.%s must be boolean; no silent fallback" % (owner, name))


@dataclass(frozen=True)
class KinematicSettings:
    """The discrete-action lattice that turns every committed route into the actions that walk it.

    Attributes:
        enabled: Whether routes are refined into action plans at all. Off,
            the converter's waypoint ladder walks them as before 2026-10-28.
        turn_action_cost: What one TURN costs against one MOVE_FORWARD over
            open floor (``LatticePlannerParams.turn_action_cost``). 1.0 is
            the benchmark's own accounting: an action is an action.
        corridor_m: How far from the 2D route the plan may wander
            (``LatticePlannerParams.corridor_m``). 0.4 m: the route is the
            medial-axis path the plan keeps; under the route memory's
            0.5 m cross-track rule.
        route_offset_cost: Actions charged per metre the plan strays from
            the route at every step (``LatticePlannerParams.route_offset_cost``).
        heuristic_weight: The lattice search's heuristic weight
            (``LatticePlannerParams.heuristic_weight``).
        max_expansions: Its expansion cap.
        lattice_floor_m: The last standoff the lattice tries, below the
            planner's ``min_clearance_m``, before the route is left to the
            converter's ladder. The lattice samples the continuous step
            every 5 cm where the 2D planner joins cell centres, so an
            8-connected route that passes an obstacle corner diagonally --
            0.21 m from it -- is infeasible to the lattice at 0.25 m: on
            the recordings of 2026-10-28, 17 of the 24 lattice failures on
            a known map planned at 0.20 m. 0.20 m: the body radius and a
            cell's margin; never below the body radius.
        resync_m: The displacement from the plan's expected pose up to
            which the plan is kept and re-anchored on the true pose (the
            remaining poses move with the agent, the same actions follow)
            rather than re-planned: the simulator's navmesh snap moves a
            step's end by centimetres. The heading must match exactly; a
            heading the plan did not predict is a re-plan.
        strict_tracking: Whether the converter tracks rigidly
            (``ActionConverterParams.strict_plan_tracking``): a one-step
            aim and no heading hysteresis in its fallback ladder, and the
            turn toward the plan's heading before any step off a plan's
            start. On, the ladder can no longer shortcut a planned turn.
        refused_move_actions: How long a refused MOVE_FORWARD excludes that
            pose's forward step from later plans, in actions; 0 for the
            episode (``LatticePlannerParams.refused_move_actions``).
        refused_move_radius_m: The radius around a refused step's start the
            exclusion covers, at the same heading.
        plan_final_yaw: Whether the plan also turns to the command's
            ``final_yaw`` at the goal. Off: the converter's facing rung does
            it, as before, one turn at a time.
        replan_backoff_actions: After a plan failed from a pose, how many
            actions pass before the lattice is asked again from the same
            route (the ladder walks it meanwhile); a route change asks at
            once. 0: asked again every action (a plan costs milliseconds).
    """

    enabled: bool = True
    turn_action_cost: float = 1.0
    corridor_m: float = 0.4
    route_offset_cost: float = 2.0
    heuristic_weight: float = 1.5
    max_expansions: int = 60000
    lattice_floor_m: float = 0.20
    resync_m: float = 0.10
    strict_tracking: bool = True
    refused_move_actions: int = 120
    refused_move_radius_m: float = 0.15
    plan_final_yaw: bool = False
    replan_backoff_actions: int = 0

    def __post_init__(self):
        _check_bool("KinematicSettings", "enabled", self.enabled)
        _check_bool("KinematicSettings", "plan_final_yaw", self.plan_final_yaw)
        _check_bool("KinematicSettings", "strict_tracking", self.strict_tracking)
        for name in ("turn_action_cost", "corridor_m", "heuristic_weight", "lattice_floor_m", "resync_m"):
            _check_positive("KinematicSettings", name, getattr(self, name))
        _check_positive("KinematicSettings", "route_offset_cost", self.route_offset_cost, allow_zero=True)
        _check_positive("KinematicSettings", "refused_move_radius_m", self.refused_move_radius_m, allow_zero=True)
        _check_count("KinematicSettings", "max_expansions", self.max_expansions)
        _check_count("KinematicSettings", "refused_move_actions", self.refused_move_actions, allow_zero=True)
        _check_count("KinematicSettings", "replan_backoff_actions", self.replan_backoff_actions, allow_zero=True)
        if self.heuristic_weight < 1.0:
            raise ValueError("KinematicSettings.heuristic_weight below 1 only slows the search")


@dataclass(frozen=True)
class ScrapeSettings:
    """When a refused MOVE_FORWARD is a doorway scrape and not a wall.

    A step the simulator refuses inside a narrow channel, with the body
    nearly aligned to the channel's axis, is the jamb clipped by a heading
    quantised to 30 degrees -- the doorway is open. Such a refusal is not
    written on the occupancy grid (``ObservedMap.notify_blocked`` paints the
    cell ahead occupied, and a second one a bar across the doorway); it is
    remembered as a refused pose for the lattice, which re-plans the entry
    from a different heading or offset.

    Attributes:
        enabled: Whether the guard applies. Off, every refused step is
            written as before.
        max_heading_deg: The largest angle between the body's heading and
            the channel's axis for which the refusal reads as a scrape. A
            heading more than this off the axis is not walking through the
            doorway, and whatever it hit is written.
        reach_m: How far from the channel's throat (the point one step
            ahead of the body, or the body itself) the refusal must be.
        scrapes_before_mark: How many scrapes at one channel within
            ``window_actions`` are excused before the next refusal is
            written after all: a doorway that refuses every alignment is
            closed, whatever the map says.
        window_actions: The window of that count.
        door_match_m: A confirmed door landmark within this distance of the
            throat is reported with the scrape (a doorway the detector saw,
            not only one the geometry found); it does not change the verdict.
    """

    enabled: bool = True
    max_heading_deg: float = 40.0
    reach_m: float = 0.75
    scrapes_before_mark: int = 3
    window_actions: int = 40
    door_match_m: float = 0.6

    def __post_init__(self):
        _check_bool("ScrapeSettings", "enabled", self.enabled)
        for name in ("max_heading_deg", "reach_m", "door_match_m"):
            _check_positive("ScrapeSettings", name, getattr(self, name))
        _check_count("ScrapeSettings", "scrapes_before_mark", self.scrapes_before_mark)
        _check_count("ScrapeSettings", "window_actions", self.window_actions)
        if self.max_heading_deg > 90.0:
            raise ValueError("ScrapeSettings.max_heading_deg beyond 90 would excuse a step across the doorway")


@dataclass(frozen=True)
class LookDownSettings:
    """The camera pitched down on the approach to a doorway or a narrow channel, and the re-plan it buys.

    A level camera 0.88 m up sees the floor from 1.4 m out and the jambs'
    bases at a few pixels per decimetre; pitched 30 degrees it sees the
    floor from 0.5 m out and the near ground at several times the density,
    and the lintel is lost while the room beyond is not (the upper edge of
    the image stays just above the horizon). The map of the entry is then
    exact where it matters, and the plan through it is re-derived on that
    map before the body commits to a heading.

    Attributes:
        enabled: Whether the look-down applies.
        approach_m: How far before the channel's entry, along the route,
            the pitch is requested.
        exit_margin_m: How far past the channel's exit the pitch is held.
        pitch_deg: The downward pitch, degrees (REP-103, positive down).
        max_hold_actions: After this many actions with the pitch held for
            one channel the request is released: an approach that does not
            get there does not keep the camera down.
    """

    enabled: bool = True
    approach_m: float = 1.5
    exit_margin_m: float = 0.5
    pitch_deg: float = 30.0
    max_hold_actions: int = 30

    def __post_init__(self):
        _check_bool("LookDownSettings", "enabled", self.enabled)
        for name in ("approach_m", "exit_margin_m", "pitch_deg"):
            _check_positive("LookDownSettings", name, getattr(self, name))
        _check_count("LookDownSettings", "max_hold_actions", self.max_hold_actions)
        if self.pitch_deg > 60.0:
            raise ValueError("LookDownSettings.pitch_deg beyond 60 loses the room ahead")
