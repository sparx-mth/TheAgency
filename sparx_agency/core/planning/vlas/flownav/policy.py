"""FlowNav behind the uniform :class:`NavigationPolicy` contract.

A thin adapter over :class:`FlowNavImageGoalClient`; see
``core/planning/vlas/navdp/policy.py`` for why the native client keeps its own
shape and this exists alongside it.

Python 3.8 compatible; numpy-only at import.
"""
from __future__ import annotations

from sparx_agency.core.planning.vlas.flownav.client import FlowNavImageGoalClient
from sparx_agency.core.planning.vlas.interfaces.goals import ImageGoal
from sparx_agency.core.planning.vlas.interfaces.policy import (
    NavigationPolicy,
    PolicyResult,
)


class FlowNavPolicy(NavigationPolicy):
    """Image-goal flow-matching policy served over HTTP.

    Args:
        url: FlowNav server base URL, e.g. ``"http://127.0.0.1:8889"``.
        timeout_s: per-request timeout (seconds).
        logger: optional ``logger(fmt, *args)`` for transport warnings.
    """

    name = "flownav"
    accepts = (ImageGoal,)

    def __init__(self, url, timeout_s=10.0, logger=None):
        self.client = FlowNavImageGoalClient(url, timeout_s=timeout_s, logger=logger)

    def reset(self, observation=None):
        """POST ``/reset`` to clear the server's rolling frame-context buffer.

        ``observation`` is accepted and ignored: FlowNav needs no camera model
        (it is RGB-only and does all preprocessing server-side).
        """
        return self.client.reset()

    def step(self, observation, goal):
        """Run one image-goal inference.

        A :class:`ImageGoal` with ``rgb=None`` means "use the goal the server
        already holds" -- that is how the in-container ROS node avoids needing
        the goal image file mounted.

        Raises:
            TypeError: ``goal`` is not an :class:`ImageGoal`.
            ValueError: the observation is missing RGB.
            FlowNavClientError: the server answered with a malformed trajectory.
        """
        self.check_goal(goal)
        if observation.rgb is None:
            raise ValueError("FlowNav needs rgb; it is an RGB-only policy.")
        result = self.client.step(observation.rgb, goal_rgb=goal.rgb)
        if result is None:
            # Transport drop at video rate: report "no result", do not raise.
            return PolicyResult(metadata={"transport_failed": True})
        distance = result.get("distance")
        return PolicyResult(
            trajectory=self.client.best_trajectory(result),
            metadata={"raw": result,
                      "temporal_distance": float(distance) if distance is not None else None})
