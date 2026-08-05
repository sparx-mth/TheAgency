"""NavDP behind the uniform :class:`NavigationPolicy` contract.

A thin adapter over :class:`NavDPPointgoalClient`. The client keeps its native,
point-goal-shaped API because the five FALCON ROS1 nodes call it directly and
depend on its exact behaviour; this class is the *uniform* view for callers that
want to drive any policy without knowing which one.

Adding nothing but translation is the point: no retry logic, no smoothing, no
frame conversion. Geometry lives in
:mod:`~sparx_agency.core.planning.vlas.navdp.geometry`.

Python 3.8 compatible; numpy-only at import.
"""
from __future__ import annotations

from sparx_agency.core.planning.vlas.interfaces.goals import PointGoal
from sparx_agency.core.planning.vlas.interfaces.policy import (
    NavigationPolicy,
    PolicyResult,
)
from sparx_agency.core.planning.vlas.navdp.client import NavDPPointgoalClient


class NavDPPolicy(NavigationPolicy):
    """Point-goal diffusion policy served over HTTP.

    Args:
        url: NavDP server base URL, e.g. ``"http://127.0.0.1:8888"``.
        timeout_s: per-request timeout (seconds).
        depth_max_m: depth clip before the uint16 wire encoding.
        logger: optional ``logger(fmt, *args)`` for transport warnings.
    """

    name = "navdp"
    accepts = (PointGoal,)

    def __init__(self, url, timeout_s=30.0, depth_max_m=5.0, logger=None):
        self.client = NavDPPointgoalClient(url, timeout_s=timeout_s,
                                           depth_max_m=depth_max_m, logger=logger)

    def reset(self, observation=None):
        """POST ``/navigator_reset`` with the observation's camera model.

        Args:
            observation: must carry ``intrinsics`` -- NavDP needs the camera
                model the frames were captured with before it can step.

        Raises:
            ValueError: no observation, or it has no intrinsics.
        """
        if observation is None or observation.intrinsics is None:
            raise ValueError(
                "NavDP.reset needs a PolicyObservation carrying intrinsics; the "
                "server projects its trajectory with them.")
        return self.client.reset(observation.intrinsics)

    def step(self, observation, goal):
        """Run one point-goal inference.

        Raises:
            TypeError: ``goal`` is not a :class:`PointGoal`.
            ValueError: the observation is missing RGB or depth.
            NavDPError: the server answered with a malformed trajectory.
        """
        self.check_goal(goal)
        if observation.rgb is None or observation.depth_m is None:
            raise ValueError("NavDP needs both rgb and depth_m; it is an RGB-D policy.")
        forward_m, left_m = goal.as_tuple()
        result = self.client.pointgoal_step(
            observation.rgb, observation.depth_m, forward_m, left_m,
            click_px=goal.metadata.get("click_px", -1),
            click_py=goal.metadata.get("click_py", -1),
            altitude=observation.altitude_m)
        if result is None:
            # Transport drop at video rate: report "no result", do not raise.
            return PolicyResult(metadata={"transport_failed": True})
        values = result.get("all_values")
        return PolicyResult(
            trajectory=self.client.best_trajectory(result),
            score=float(values[0][0]) if values else None,
            metadata={"raw": result})
