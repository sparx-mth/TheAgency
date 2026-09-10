"""What the headless agent records: why each step's action was chosen, and what its episode added up to.

Every decision carries its reason, returned to the caller of ``act()`` in
``AgentDecision.info``; every episode has a report, ``episode_info()``, which
the harness stores beside its score (the results row's ``agent_info``). The
harness does not store the per-step reasons: a caller that wants a step trace
keeps the decisions ``act()`` returns. The quiet failures designed out here:

* **Reports whose rows do not line up.** The per-status counts always carry
  every converter status, zeros included, so two episodes' reports have the
  same keys in the same order.
* **A record that rewrites the command.** The policy's per-step ``info`` and
  its episode report are copied, never shared, so a policy that reuses its
  dict next step cannot change what was returned or reported.
* **A report that does not serialise.** A policy's ``episode_info()`` that is
  not a mapping is refused, not stringified; everything built here is plain
  JSON types.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
from typing import Any, Dict, Mapping

from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUSES,
    ConversionResult,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


def step_info(result: ConversionResult,
              command: NavigationCommand) -> Dict[str, Any]:
    """Why one step's action was chosen, in plain JSON types.

    Args:
        result: The converter's decision.
        command: The policy's command it was decided from.

    Returns:
        The converter's ``status``, whether the step was ``idle``, the path
        geometry (``segment_index``, ``heading_error_rad``,
        ``distance_to_goal_m``, ``remaining_path_m``, ``cross_track_m``),
        ``forward_blocked``, and a copy of the command's ``info`` under
        ``"policy"``.
    """
    return {
        "status": result.status,
        "idle": result.idle,
        "segment_index": int(result.segment_index),
        "heading_error_rad": float(result.heading_error_rad),
        "distance_to_goal_m": float(result.distance_to_goal_m),
        "remaining_path_m": float(result.remaining_path_m),
        "cross_track_m": float(result.cross_track_m),
        "forward_blocked": result.forward_blocked,
        "policy": dict(command.info),
    }


def policy_report(policy) -> Dict[str, Any]:
    """The policy's own episode report, or ``{}`` when it keeps none.

    ``episode_info()`` is an optional, feature-detected method of a
    :class:`~sparx_agency.core.planning.objnav.interfaces.policy.SearchPolicy`:
    an attribute of that name that is not callable counts as absent.

    Args:
        policy: The search policy.

    Returns:
        A copy of what its ``episode_info()`` returned.

    Raises:
        TypeError: If ``episode_info()`` returns something that is not a
            mapping.
    """
    report = getattr(policy, "episode_info", None)
    if not callable(report):
        return {}
    info = report()
    if not isinstance(info, collections.abc.Mapping):
        raise TypeError(
            "policy %r returned %r from episode_info(); return a dict"
            % (getattr(policy, "name", policy), info))
    return dict(info)


class EpisodeLog:
    """The headless agent's counters for one episode, all starting at zero.

    Attributes:
        steps: Decisions returned.
        idle_steps: Decisions the converter had no action for.
        blocked_forward: Decisions made after a MOVE_FORWARD that did not move
            the agent.
        blocked_notifications: Calls of the policy's ``notify_blocked`` hook.
        statuses: Decisions per converter status, every status present.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.idle_steps = 0
        self.blocked_forward = 0
        self.blocked_notifications = 0
        self.statuses = {status: 0 for status in STATUSES}

    def count(self, result: ConversionResult) -> None:
        """Add one decision.

        Args:
            result: The converter's decision behind it.
        """
        self.steps += 1
        self.statuses[result.status] += 1
        if result.idle:
            self.idle_steps += 1
        if result.forward_blocked:
            self.blocked_forward += 1

    def count_notification(self) -> None:
        """Add one call of the policy's ``notify_blocked`` hook."""
        self.blocked_notifications += 1

    def report(self, agent: str,
               policy_info: Mapping[str, Any]) -> Dict[str, Any]:
        """The episode's report, JSON-serialisable.

        Args:
            agent: The name the agent's results are logged under.
            policy_info: The policy's own report (:func:`policy_report`).

        Returns:
            ``agent``, ``steps``, ``idle_steps``, ``blocked_forward``,
            ``blocked_notifications``, ``statuses`` and ``policy``, the last
            two copied.
        """
        return {
            "agent": agent,
            "steps": self.steps,
            "idle_steps": self.idle_steps,
            "blocked_forward": self.blocked_forward,
            "blocked_notifications": self.blocked_notifications,
            "statuses": dict(self.statuses),
            "policy": dict(policy_info),
        }
