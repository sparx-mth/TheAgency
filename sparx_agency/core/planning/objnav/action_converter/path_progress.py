"""Where the agent is along the path it follows, and the point it aims at.

The converter's progress is state -- the path it adopted, and the segment and
arc length reached on it -- kept here apart from the decision ladder so that
its two rules live in one place:

* **Progress only runs forward**, in segment *and* arc length, so a path that
  doubles back near itself, nearly reverses at a corner, or even retraces
  itself exactly is walked to its end instead of paced at a corner forever
  (why the arc length matters:
  :mod:`~sparx_agency.core.planning.objnav.action_converter.path_geometry`).
* **Nor does it jump ahead to a leg not yet walked.** A later leg that passes
  nearer the agent than the leg it walks -- the return of an out-and-back a
  few centimetres beside its way out, a tour's last leg crossing its first --
  would take a nearest-wins projection, and the agent would skip everything
  between, or arrive at once. The heading dead band lets the agent walk up to
  ``keep_within_m`` off a leg, so the first leg it stands that near, and has
  not walked past the end of, keeps the progress.
* **Progress belongs to one path.** The same path sent again keeps it; any
  other path, a hold included, starts it over. ``NavigationCommand`` coerces
  waypoints to float tuples, so a policy that re-sends the same route every
  step keeps its progress however it built the points.
* **The aim never sits on the agent's own foot.** Where the path folds back
  exactly onto itself, the point a lookahead past the progress can land on
  the agent's position, and the heading to it is noise: the agent steps
  sideways off the path, and an A-B-A-B path of quarter-metre reversals is
  paced forever. An aim nearer the agent than half a forward step is moved on
  along the path until it is that far away, or the path ends.

Python 3.8 syntax; numpy arrives only through the types it imports.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Tuple

from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    heading_error,
)
from sparx_agency.core.planning.objnav.action_converter.checks import (
    require_positive,
)
from sparx_agency.core.planning.objnav.action_converter.path_geometry import (
    arclength_beyond,
    path_length,
    point_at_arclength,
    project_onto_path,
)
from sparx_agency.core.planning.objnav.types.command import Waypoint
from sparx_agency.core.planning.objnav.types.pose import AgentPose


class PathSample(NamedTuple):
    """Where the agent is on its path this step, and what it aims at.

    Attributes:
        segment_index: The segment the progress projects onto.
        cross_track_m: Distance from the agent to its projection.
        distance_to_goal_m: Straight-line distance to the last waypoint.
        remaining_path_m: Path length from the projection to the last
            waypoint.
        target_xy: The aim point, a lookahead along the path past the
            progress.
        aims_at_goal: Whether the aim point is exactly the last waypoint.
        heading_error_rad: Signed turn from the agent's heading to the aim,
            radians in ``(-pi, pi]``, positive to the left.
    """

    segment_index: int
    cross_track_m: float
    distance_to_goal_m: float
    remaining_path_m: float
    target_xy: Tuple[float, float]
    aims_at_goal: bool
    heading_error_rad: float


class PathProgress:
    """One adopted path and how far along it the agent has come.

    Starts with no path. Use a new one to forget a path and its progress.
    """

    def __init__(self) -> None:
        self._path = ()
        self._length = 0.0
        self._cursor = 0
        self._progress_m = 0.0

    @property
    def path(self) -> Tuple[Waypoint, ...]:
        """The adopted waypoints; ``()`` when there is no path."""
        return self._path

    def adopt(self, waypoints: Tuple[Waypoint, ...]) -> None:
        """Follow ``waypoints``: keep the progress on the same path, else start over.

        Args:
            waypoints: A command's waypoints, a tuple of float pairs as
                ``NavigationCommand`` holds them; ``()`` for no path.
        """
        if waypoints == self._path:
            return
        self._path = waypoints
        self._length = path_length(waypoints) if waypoints else 0.0
        self._cursor = 0
        self._progress_m = 0.0

    def advance(self, pose: AgentPose, lookahead_m: float,
                min_aim_m: float, keep_within_m: float = 0.0) -> PathSample:
        """Project ``pose`` onto the path ahead of the progress, advance it, and aim.

        Args:
            pose: The agent's true pose now.
            lookahead_m: How far along the path past the progress to aim,
                metres.
            min_aim_m: The nearest the aim may sit to the agent unless it is
                the goal, metres; the converter passes half a forward step.
                A nearer aim is moved on along the path until it is this far
                away, or the path ends.
            keep_within_m: How far off a leg the agent may stand and still be
                walking it, metres; the converter passes
                :func:`~sparx_agency.core.planning.objnav.action_converter.ladder.parallel_offset`.
                The first leg this near the agent, and not walked past the
                end of, keeps the progress, however near a later leg passes.
                0 lets the nearest leg win.

        Returns:
            Where the agent is and what it aims at. Past the end of the path
            the aim is exactly the last waypoint.

        Raises:
            ObjNavError: If there is no path, ``min_aim_m`` is not positive
                and finite, or ``keep_within_m`` is negative or not finite.
        """
        require_positive("min_aim_m", min_aim_m)
        segment, arc, cross_track, _ = project_onto_path(
            self._path, pose.x, pose.y, from_segment=self._cursor,
            from_arc_m=self._progress_m, keep_within_m=keep_within_m)
        self._cursor = segment
        # max(): the projection may land an ulp short of its bound.
        self._progress_m = max(self._progress_m, arc)
        reach = self._progress_m + lookahead_m
        target = point_at_arclength(self._path, reach)
        if (reach < self._length and math.hypot(target[0] - pose.x,
                                                target[1] - pose.y)
                < min_aim_m):
            # A fold put the aim on the agent's own foot; aim where the path
            # leaves the disc round the agent instead.
            reach = arclength_beyond(self._path, pose.x, pose.y, min_aim_m,
                                     from_arc_m=reach)
            target = point_at_arclength(self._path, reach)
        goal_x, goal_y = self._path[-1]
        return PathSample(
            segment_index=segment,
            cross_track_m=cross_track,
            distance_to_goal_m=math.hypot(goal_x - pose.x, goal_y - pose.y),
            remaining_path_m=self._length - arc,
            target_xy=target,
            # point_at_arclength returns exactly the last waypoint from here.
            aims_at_goal=reach >= self._length,
            heading_error_rad=heading_error(pose.x, pose.y, pose.yaw,
                                            target[0], target[1]))
