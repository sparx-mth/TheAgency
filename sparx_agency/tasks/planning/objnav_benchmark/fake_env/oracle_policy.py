"""The privileged oracle: walk the fake's geodesic field into the goal region, then stop.

**An upper bound for testing the pipeline, never a baseline.** It reads the
fake environment's ground truth -- the building, the goal region, the
navigable cells -- which no real method sees, so its SR and SPL say only that
the loop *can* score: that the headless agent, the action converter, the
runner and the scoring do not lose a success the policy earned. A number from
it belongs in a test, never in a results table.

Two subtleties decide whether it actually gets there, and both are failures
that end an episode at the step budget, not with an error:

* **Where it aims.** At the goal region eroded from its open side
  (:mod:`.oracle_aim`), so arriving at its path's end puts the agent inside
  the region rather than one cell short of it; it stops the moment the agent
  enters the *full* region.
* **How close to walls it walks.** The converter aims a lookahead ahead along
  the path, so at a corner it cuts inside; next to a wall that step is
  blocked and the fake refuses it atomically. The converter's recovery -- aim
  one forward step ahead until a step moves -- changes the aim but not always
  the action: when the nearer aim still lies inside the heading dead band
  (half a turn), the same pose gives the same blocked MOVE_FORWARD until the
  budget runs out. Measured on the demo building with that recovery in
  place, a descent of the plain geodesic still lost 15 of 90 episodes this
  way (29 of the 90 had a blocked step). So the field it descends is the
  geodesic weighted by :func:`~.geodesics.clearance_costs`, which keeps the
  path a cell off the walls wherever a lane exists and still lets it through
  a doorway or into an aim that has none: 90 of 90, no step blocked. The
  environment's own distances (``l``, ``d0``, ``dT``) stay the true
  geodesic; only the oracle's route bends.

Its map is the ground truth, so a blocked step is never an obstacle it does
not know about: :meth:`OracleSearchPolicy.notify_blocked` only counts, and a
count above 0 means the clearance weighting has stopped keeping it off the
walls -- a regression to report, not to route around.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    ObjNavInternalError,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.target import TargetLabels
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_length
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import (
    clearance_costs,
    geodesic_field,
    legal_moves,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_aim import (
    erode_goal_region,
    erosion_cells,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import Cell


def _every_second_and_last(cells: List[Cell]) -> List[Cell]:
    """Every second cell from the first, and the last one always."""
    picked = cells[::2]
    if (len(cells) - 1) % 2:
        picked.append(cells[-1])
    return picked


class OracleSearchPolicy:
    """A :class:`SearchPolicy` that knows where the goal is. Privileged; never a baseline.

    Args:
        env: The fake environment whose ground truth it reads.
        converter_goal_tolerance_m: The agent's action-converter goal
            tolerance, which decides how deep inside the region to aim.

    Raises:
        TypeError: If ``env`` is not a :class:`FakeObjNavEnv`.
        ObjNavError: On a tolerance that is not positive and finite.
    """

    name = "oracle"

    def __init__(self, env: FakeObjNavEnv,
                 converter_goal_tolerance_m: float = 0.25) -> None:
        if not isinstance(env, FakeObjNavEnv):
            raise TypeError("the oracle reads a FakeObjNavEnv's ground truth; "
                            "got %r" % (env,))
        if not (is_length(converter_goal_tolerance_m)
                and converter_goal_tolerance_m > 0):
            raise ObjNavError("converter_goal_tolerance_m must be a positive "
                              "finite number, got %r"
                              % (converter_goal_tolerance_m,))
        self._env = env
        self._world = env.privileged_world()
        self._navigable = env.privileged_navigable_mask()
        self._costs = clearance_costs(self._navigable)
        self._tolerance = float(converter_goal_tolerance_m)
        self._forget()

    def _forget(self) -> None:
        self._goal: Optional[np.ndarray] = None
        self._field: Optional[np.ndarray] = None
        self._erosion = 0
        self._aim_is_full_region = False
        self._counts = {"plans": 0, "stops": 0, "holds": 0, "follows": 0,
                        "blocked": 0}

    def reset(self, episode: ObjNavEpisode, target: TargetLabels) -> None:
        """Aim at the eroded goal region of the episode's category.

        Raises:
            TypeError: On a wrong type for ``episode`` or ``target``.
            ObjNavError: If ``target`` is another category's, or the
                environment has no goal region for it.
        """
        self._forget()
        if not isinstance(episode, ObjNavEpisode):
            raise TypeError("reset() needs an ObjNavEpisode, got %r" % (episode,))
        if not isinstance(target, TargetLabels):
            raise TypeError("reset() needs TargetLabels, got %r" % (target,))
        if target.category != episode.target_category:
            raise ObjNavError("target %r is not episode %r's category %r"
                              % (target.category, episode.episode_id,
                                 episode.target_category))
        goal = self._env.privileged_goal_mask(episode.target_category)
        erosion = erosion_cells(self._tolerance,
                                episode.action_spec.forward_step_m,
                                self._world.resolution_m)
        aim = erode_goal_region(goal, self._navigable, erosion)
        self._aim_is_full_region = not aim.any()
        if self._aim_is_full_region:
            aim = goal
        self._field = geodesic_field(self._world, aim, self._navigable,
                                     self._costs)
        self._goal = goal
        self._erosion = erosion

    def notify_blocked(self, observation: ObjNavObservation) -> None:
        """Count a MOVE_FORWARD that did not move the agent; the route stays as it is.

        The headless agent calls this before :meth:`plan` whenever its last
        MOVE_FORWARD did not move. The oracle's map is the ground truth, so
        the block is never an obstacle it did not know about -- it is the
        converter's step cutting a corner of the route -- and there is nothing
        to mark or re-plan. :meth:`episode_info` reports the count as
        ``"blocked"``.

        Raises:
            ObjNavError: Before :meth:`reset`.
            TypeError: If ``observation`` is not an :class:`ObjNavObservation`.
        """
        if self._field is None:
            raise ObjNavError("notify_blocked() before reset(); start an "
                              "episode")
        if not isinstance(observation, ObjNavObservation):
            raise TypeError("notify_blocked() needs an ObjNavObservation, got "
                            "%r" % (observation,))
        self._counts["blocked"] += 1

    def plan(self, observation: ObjNavObservation) -> NavigationCommand:
        """STOP inside the goal region; otherwise the geodesic descent toward the aim.

        Returns:
            ``stop_here`` when the agent's cell is in the full goal region;
            ``hold`` (reason ``"no path"``) when the aim is unreachable from
            it; else ``follow`` through the centres of every second cell of
            the descent, and its last.

        Raises:
            ObjNavError: Before :meth:`reset`, or when the agent stands on a
                cell the fake would never put it on.
            TypeError: If ``observation`` is not an :class:`ObjNavObservation`.
        """
        if self._field is None:
            raise ObjNavError("plan() before reset(); start an episode")
        if not isinstance(observation, ObjNavObservation):
            raise TypeError("plan() needs an ObjNavObservation, got %r"
                            % (observation,))
        row, col = self._world.cell_of(observation.pose.x, observation.pose.y)
        if not (self._world.in_bounds(row, col) and self._navigable[row, col]):
            raise ObjNavError(
                "the agent stands in cell %r, which is not navigable; the fake "
                "never puts it there, so this observation is not the fake's"
                % ((row, col),))
        self._counts["plans"] += 1
        where = [row, col]
        if self._goal[row, col]:
            self._counts["stops"] += 1
            return NavigationCommand.stop_here(
                info={"reason": "in goal region", "cell": where})
        remaining = float(self._field[row, col])
        if math.isinf(remaining):
            self._counts["holds"] += 1
            return NavigationCommand.hold(info={"reason": "no path",
                                                "cell": where})
        cells = self._descend((row, col))
        self._counts["follows"] += 1
        return NavigationCommand.follow(
            [self._world.cell_center(r, c)
             for r, c in _every_second_and_last(cells)],
            info={"reason": "descend", "cell": where, "geodesic_m": remaining,
                  "path_cells": len(cells)})

    def episode_info(self) -> Dict[str, Any]:
        """How deep the aim sat, whether erosion emptied it, what each plan was, and how many steps were blocked.

        Returns:
            ``erosion_cells``, ``aim_is_full_region``, and the counts
            ``plans``, ``stops``, ``holds``, ``follows`` and ``blocked`` (calls
            of :meth:`notify_blocked`).
        """
        info: Dict[str, Any] = {"erosion_cells": self._erosion,
                                "aim_is_full_region": self._aim_is_full_region}
        info.update(self._counts)
        return info

    def _descend(self, cell: Cell) -> List[Cell]:
        """Cells from ``cell`` down the field to 0, each the lowest legal neighbour.

        Ties go to the first move in ``MOVES`` order. The field strictly falls
        along a legal move out of every reachable non-target cell, so the
        descent ends; a cell with no lower neighbour is this package's bug.
        """
        cells = [cell]
        value = self._field[cell]
        while value > 0.0:
            row, col = cells[-1]
            best, best_value = None, value
            for r, c, _ in legal_moves(self._world, row, col, self._navigable):
                if self._field[r, c] < best_value:
                    best, best_value = (r, c), self._field[r, c]
            if best is None:
                raise ObjNavInternalError(
                    "the geodesic field has no lower legal neighbour at cell "
                    "%r (value %r)" % ((row, col), value))
            cells.append(best)
            value = best_value
        return cells

    def __repr__(self) -> str:
        return "OracleSearchPolicy(env=%r, converter_goal_tolerance_m=%r)" % (
            self._env, self._tolerance)
