"""Turn a search policy's waypoints into one benchmark action per step.

The policy says *where* -- a path of world points, a camera pitch, a heading
to face, or stop -- and :class:`DiscreteActionConverter` says *how*, one
discrete action per step, deciding as if every action executes perfectly. It
re-reads the true pose each step, so an action that did not execute as
modelled is absorbed on the next step instead of accumulating.

The converter holds the state and answers STOP. Progress along the path is
kept by :mod:`~sparx_agency.core.planning.objnav.action_converter.path_progress`
and every other decision is the stateless ladder in
:mod:`~sparx_agency.core.planning.objnav.action_converter.ladder`. The quiet
failures handled here:

* **A STOP that is not obeyed.** A stop command is STOP from anywhere, before
  any other rung: ending the episode is the policy's call, never
  second-guessed. A ``stop_on_arrival`` command gets STOP from the ladder on
  the step it is fully executed.
* **Lost progress.** The same path sent again keeps its progress, and a
  rejected command leaves the converter exactly as it was.
* **The same blocked step, forever.** The default lookahead, two steps, cuts
  inside corners. Where the cut hits something the policy's collision-free
  path avoids, the step is blocked, the pose does not change, and the same
  pose gives the same decision: the agent repeats the blocked step until the
  step budget runs out (on the fake building 27 of 90 oracle episodes
  livelocked exactly so). So :meth:`DiscreteActionConverter.forward_blocked`
  reports the blocked step, and until a MOVE_FORWARD moves the agent again the
  converter aims one forward step ahead instead, hugging the path the policy
  planned. That breaks the loop only when the nearer aim leaves the heading
  dead band (half a turn) and the ladder turns: where it stays inside, the
  same blocked MOVE_FORWARD still repeats (on the fake building the oracle
  without its clearance penalty loses 15 of 90 episodes so). It never
  improvises a detour: replanning is the policy's job -- ``notify_blocked``
  -- and a converter that steered round the wall would hide it.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
import numbers
from typing import Optional

from sparx_agency.core.planning.objnav.action_converter.ladder import (
    check_executable,
    choose_action,
    parallel_offset,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.path_progress import (
    PathProgress,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    ROLLOUT_LIMIT,
    STATUS_STOP,
    ConversionResult,
    Rollout,
)
from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    ObjNavInternalError,
)
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose


class DiscreteActionConverter:
    """Follows a policy's commands one discrete action at a time.

    Stateful: it holds the adopted path and the progress along it, the last
    emitted action with the pose it was emitted at, and whether it is
    recovering from a blocked step. Use one per episode, or :meth:`reset`
    between episodes.

    Args:
        spec: The benchmark's action geometry and action set.
        params: Tuning; the defaults when None.

    Raises:
        TypeError: If ``spec`` or ``params`` has the wrong type.
        ObjNavError: If ``params.lookahead_m`` is shorter than one forward
            step (the step would overshoot the aim, and the agent turn back),
            or ``params.blocked_epsilon_m`` is not shorter than one (every
            unobstructed step would read as blocked).
    """

    def __init__(self, spec: DiscreteActionSpec,
                 params: Optional[ActionConverterParams] = None) -> None:
        if not isinstance(spec, DiscreteActionSpec):
            raise TypeError("spec must be a DiscreteActionSpec, got %r"
                            % (spec,))
        if params is None:
            params = ActionConverterParams()
        if not isinstance(params, ActionConverterParams):
            raise TypeError("params must be ActionConverterParams, got %r"
                            % (params,))
        _check_fit(spec, params)
        self._spec = spec
        self._params = params
        self.reset()

    @property
    def spec(self) -> DiscreteActionSpec:
        """The benchmark's action geometry and action set."""
        return self._spec

    @property
    def params(self) -> ActionConverterParams:
        """The conversion tuning."""
        return self._params

    @property
    def recovering(self) -> bool:
        """Whether the aim is one forward step ahead because a step was blocked.

        Set when a MOVE_FORWARD turns out blocked, and cleared when a later
        MOVE_FORWARD moves the agent, or by :meth:`reset`. Not by a new path:
        a re-plan usually cuts the same corner.
        """
        return self._recovering

    def reset(self) -> None:
        """Forget everything: the path, its progress, the last action and any recovery.

        A reset starts a new episode somewhere else, where a kept recovery
        would aim one step ahead for a wall that is not there -- and make the
        results depend on episode order.
        """
        self._path_progress = PathProgress()
        self._last_action = None
        self._last_pose = None
        self._recovering = False

    def forward_blocked(self, pose: AgentPose) -> bool:
        """Whether the last MOVE_FORWARD failed to move the agent to ``pose``.

        The one definition of a blocked step: :meth:`step` reports it and
        recovers from it, and the headless agent tells the policy with it.
        Read-only, so asking before :meth:`step` changes nothing.

        Args:
            pose: The agent's true pose now, before this step's decision.

        Returns:
            True when the last action this converter emitted was MOVE_FORWARD
            and the agent is less than ``params.blocked_epsilon_m`` (3-D) from
            the pose it was emitted at; False otherwise, and before the first
            action.

        Raises:
            TypeError: If ``pose`` is not an :class:`AgentPose`.
        """
        if not isinstance(pose, AgentPose):
            raise TypeError("pose must be an AgentPose, got %r" % (pose,))
        if self._last_action != DiscreteAction.MOVE_FORWARD:
            return False
        last = self._last_pose
        moved = math.hypot(pose.x - last.x, pose.y - last.y, pose.z - last.z)
        return moved < self._params.blocked_epsilon_m

    def step(self, pose: AgentPose, command: NavigationCommand
             ) -> ConversionResult:
        """Decide this step's action.

        In order, first match wins:

        1. (Bookkeeping) :meth:`forward_blocked`, which starts a blocked-step
           recovery -- or ends one, when the last MOVE_FORWARD moved.
        2. The command asks to stop: STOP.
        3. (Bookkeeping) Adopt the command's path if it differs from the
           current one, restarting the progress, and project the pose onto
           it -- the first leg within ``parallel_offset`` of it -- aiming
           ``params.lookahead_m`` ahead (one forward step while recovering),
           never within half a step of the agent short of the goal.
        4. The ladder,
           :func:`~sparx_agency.core.planning.objnav.action_converter.ladder.choose_action`:
           tilt, face, arrive (STOP for ``stop_on_arrival``), turn toward the
           aim, or step forward.

        Args:
            pose: The agent's true pose now.
            command: This step's command from the policy.

        Returns:
            The decision and its geometry; path fields are filled whenever
            there is a path, and ``forward_blocked`` on every result.

        Raises:
            TypeError: If ``pose`` or ``command`` has the wrong type.
            CommandError: If the command sets a camera pitch and the spec has
                no LOOK actions. The converter is left exactly as it was.
        """
        if not isinstance(pose, AgentPose):
            raise TypeError("pose must be an AgentPose, got %r" % (pose,))
        if not isinstance(command, NavigationCommand):
            raise TypeError("command must be a NavigationCommand, got %r"
                            % (command,))
        result = self._decide(pose, command)
        if result.action is not None and not self._spec.allows(result.action):
            raise ObjNavInternalError(
                "the converter chose %s, which this spec does not allow"
                % (result.action.name,))
        self._last_action = result.action
        self._last_pose = pose
        return result

    def rollout(self, pose: AgentPose, command: NavigationCommand,
                max_actions: int) -> Rollout:
        """Execute ``command`` to completion under perfect execution.

        A fresh converter of this class, with the same spec and params, steps
        from ``pose``, each action applied with
        :func:`~sparx_agency.core.planning.objnav.action_converter.transition.apply_action`.
        Fresh means no progress and no blocked-step recovery: perfect
        execution never blocks a step. It ends on an idle result (no action
        added; its status is the rollout's), on STOP (included), or once
        ``max_actions`` actions are emitted -- ``ROLLOUT_LIMIT``, unless the
        command is done at exactly that point. This converter's own state is
        untouched.

        Args:
            pose: Where the rollout starts.
            command: The command, held fixed for the whole rollout.
            max_actions: The action budget. At least 1.

        Returns:
            The actions, the poses (start first) and how it ended.

        Raises:
            ObjNavError: If ``max_actions`` is not a positive integer.
            CommandError: As :meth:`step`.
        """
        if (not isinstance(max_actions, numbers.Integral)
                or isinstance(max_actions, bool) or max_actions < 1):
            raise ObjNavError("max_actions must be a positive integer, got %r"
                              % (max_actions,))
        # type(self): a subclass's rollout must predict with its own
        # decisions, not silently with the base class's.
        twin = type(self)(self._spec, self._params)
        actions = []
        poses = [pose]
        while True:
            result = twin.step(poses[-1], command)
            if result.idle:
                return Rollout(actions, poses, result.status)
            if len(actions) == max_actions:
                return Rollout(actions, poses, ROLLOUT_LIMIT)
            actions.append(result.action)
            poses.append(apply_action(poses[-1], result.action, self._spec))
            if result.action == DiscreteAction.STOP:
                return Rollout(actions, poses, STATUS_STOP)

    def _decide(self, pose: AgentPose, command: NavigationCommand
                ) -> ConversionResult:
        """:meth:`step` without its type checks and last-action bookkeeping."""
        # Refused before any state changes, so a rejected command leaves the
        # converter exactly as it was.
        check_executable(command, self._spec)
        blocked = self.forward_blocked(pose)
        if blocked:
            self._recovering = True
        elif self._last_action == DiscreteAction.MOVE_FORWARD:
            self._recovering = False
        if command.stop:
            return ConversionResult(DiscreteAction.STOP, STATUS_STOP,
                                    forward_blocked=blocked)
        self._path_progress.adopt(command.waypoints)
        # Projected before the pitch decision only so that a pitch step
        # reports the path geometry too. A LOOK does not move the agent, so
        # committing the progress here equals committing it next step.
        sample = None
        if self._path_progress.path:
            lookahead = self._params.lookahead_m
            if self._recovering:
                lookahead = self._spec.forward_step_m
            sample = self._path_progress.advance(
                pose, lookahead, self._spec.forward_step_m / 2.0,
                parallel_offset(self._spec, self._params.lookahead_m))
        return choose_action(pose, command, sample, blocked, self._spec,
                             self._params)


def _check_fit(spec: DiscreteActionSpec,
               params: ActionConverterParams) -> None:
    """Raise unless the tuning fits one forward step of ``spec``."""
    step = spec.forward_step_m
    if params.lookahead_m < step:
        raise ObjNavError(
            "lookahead_m=%r is shorter than one forward step (%r m): the "
            "step would overshoot the aim point and the agent would turn "
            "back; use a lookahead of at least one step"
            % (params.lookahead_m, step))
    if params.blocked_epsilon_m >= step:
        raise ObjNavError(
            "blocked_epsilon_m=%r is not shorter than one forward step (%r m): "
            "every unobstructed MOVE_FORWARD would read as blocked; use an "
            "epsilon well below the step" % (params.blocked_epsilon_m, step))
