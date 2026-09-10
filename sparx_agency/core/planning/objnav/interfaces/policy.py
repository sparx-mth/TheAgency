"""The :class:`SearchPolicy` contract: where a search method plugs into the agent.

The method -- the scene graph, the LLM's room probabilities, RPT*, or any
baseline -- implements two calls. It is told the target once per episode, in
our vision vocabulary, and asked each step where to go, in world geometry. It
never emits a discrete action and never sees the simulator; the headless agent
converts its answer into the benchmark's action space.

A ``Protocol``, following the repo's rule for pipeline stages: any object with
these two methods and a ``name`` is a policy, no base class required.

Two optional methods are feature-detected rather than declared, so a policy
without them still satisfies the protocol:

* ``notify_blocked(observation) -> None`` -- called by the headless agent just
  before :meth:`SearchPolicy.plan` whenever the previous MOVE_FORWARD did not
  move the agent: something the policy's map does not show is in the way (a
  low obstacle below the camera's view, a corner the path hugged too tightly).
  A policy that ignores it keeps sending the same path into the same wall
  until the step budget runs out, so a real method should mark the cell ahead
  as blocked and replan. It must return None: a command returned from it would
  be silently ignored, so the agent refuses anything else with ``TypeError``.
* ``episode_info() -> Mapping`` -- per-episode diagnostics, JSON-serialisable,
  logged beside the score.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchPolicy(Protocol):
    """Decides where the agent goes next.

    Attributes:
        name: Identifier logged with every result.
    """

    name: str

    def reset(self, episode, target) -> None:
        """Start an episode.

        Args:
            episode: The public
                :class:`~sparx_agency.core.planning.objnav.types.ObjNavEpisode`.
            target: The goal as
                :class:`~sparx_agency.core.planning.objnav.types.TargetLabels`:
                the LLM query, the detector prompts and the accept set.
        """

    def plan(self, observation):
        """Decide this step's command.

        Args:
            observation: The current
                :class:`~sparx_agency.core.planning.objnav.types.ObjNavObservation`.

        Returns:
            A :class:`~sparx_agency.core.planning.objnav.types.NavigationCommand`.
        """
