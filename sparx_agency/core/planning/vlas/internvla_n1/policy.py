"""InternVLA-N1 behind the uniform :class:`NavigationPolicy` contract.

A thin adapter over :class:`~sparx_agency.core.planning.vlas.internvla_n1.client.ModelClient`.
The client keeps its native, action-shaped API (``step`` returns a discrete VLN
action and a System-2 pixel goal) because the existing Rooster/Gazebo bridge
depends on it; this class is the *uniform* view -- observation + language goal in,
a **body-frame trajectory** out -- so N1 can be driven exactly like NavDP: a
runner anchors that trajectory at the pose it was asked from and flies it as a
route (see :mod:`~sparx_agency.core.planning.vlas.common.plan_commit`).

The trajectory is InternVLA-N1's own. Where the server exposes System 1's
continuous prediction, that curve *is* the result. Where it exposes only the
discrete action it decided on -- which the deployed VLN-CE agent server does --
the action is rendered as one short followable step, so the aircraft keeps
moving. Both are shaped in
:mod:`~sparx_agency.core.planning.vlas.internvla_n1.geometry`; this class adds
nothing but translation, the same discipline as :class:`NavDPPolicy`.

``requests`` (via the client) is imported inside ``__init__``, not at module
scope, so importing this module stays numpy-only -- the rule every policy here
holds to.

Python 3.8 compatible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sparx_agency.core.planning.vlas.interfaces.goals import LanguageGoal
from sparx_agency.core.planning.vlas.interfaces.policy import (
    NavigationPolicy,
    PolicyResult,
)
from sparx_agency.core.planning.vlas.internvla_n1 import geometry


class InternVLAN1Policy(NavigationPolicy):
    """Dual-system vision-language navigation policy served over HTTP.

    Args:
        url: full server base URL, e.g. ``"http://127.0.0.1:8087"``. Overrides
            ``host``/``port`` when given.
        host: server host, used when ``url`` is not given.
        port: server port, used when ``url`` is not given.
        timeout_s: per-request timeout, seconds.
        step_m: forward reach of the step rendered from a discrete action.
        turn_deg: heading offset a discrete turn action bends its step by.
        model_name: variant passed to ``/agent/init``.
        ckpt_path: checkpoint path passed to ``/agent/init`` (server-side path).
        model_settings: overrides merged into the server's model settings on
            init -- crucially the camera intrinsics and frame size the frames
            were captured with, so the server projects its pixel goal correctly.
        logger: optional logger object (``.info``/``.warn``/...) for transport
            and server messages.
    """

    name = "internvla_n1"
    accepts = (LanguageGoal,)

    def __init__(self, url=None, host="127.0.0.1", port=8087, timeout_s=30.0,
                 step_m=geometry.STEP_SIZE_M, turn_deg=geometry.TURN_ANGLE_DEG,
                 model_name="internvla_n1", ckpt_path="", model_settings=None,
                 logger=None):
        # Lazy: the client pulls `requests`, which the FALCON Noetic container
        # does not have. Importing this module must not.
        from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient

        if url is not None:
            protocol, _, rest = str(url).partition("://")
            if not rest:
                protocol, rest = "http", protocol
            hostpart, _, portpart = rest.partition(":")
            host = hostpart or host
            port = int(portpart) if portpart else port
        else:
            protocol = "http"
        self.client = ModelClient(host=host, port=int(port), timeout=timeout_s,
                                  protocol=protocol, logger=logger)
        self.step_m = float(step_m)
        self.turn_deg = float(turn_deg)
        self._model_name = model_name
        self._ckpt_path = ckpt_path
        self._model_settings = dict(model_settings) if model_settings else {}

    def reset(self, observation=None):
        """Initialise the server agent (idempotent) and clear its episode state.

        Args:
            observation: unused; accepted for contract parity with policies that
                need the camera model up front. N1 takes its intrinsics from
                ``model_settings`` at construction instead.

        Returns:
            ``True`` once the agent exists and its per-episode state is cleared.
        """
        self.client.init_agent(model_name=self._model_name,
                               ckpt_path=self._ckpt_path,
                               model_settings=self._model_settings)
        return self.client.reset()

    def step(self, observation, goal):
        """Run one language-goal inference and return a body-frame trajectory.

        Args:
            observation: current :class:`PolicyObservation`; ``rgb`` is required,
                ``depth_m`` is used when present.
            goal: a :class:`LanguageGoal` carrying the instruction.

        Returns:
            A :class:`PolicyResult`. A transport drop returns a not-``ok`` result
            (the caller re-sends next frame); a STOP with no curve returns a
            not-``ok`` result with ``stop=True``.

        Raises:
            TypeError: ``goal`` is not a :class:`LanguageGoal`.
            ValueError: the observation has no ``rgb``.
        """
        self.check_goal(goal)
        if observation.rgb is None:
            raise ValueError("InternVLA-N1 needs an rgb frame to step.")

        bgr = self._to_bgr(observation.rgb)
        depth = self._to_depth(observation.depth_m)
        result = self.client.step(bgr, goal.instruction, depth)

        if not result.success:
            # Transport / server error at video rate: report "no result", the
            # caller re-sends. Matches NavDPPolicy's dropped-inference contract.
            return PolicyResult(metadata={"transport_failed": True,
                                          "error": result.error})

        trajectory = geometry.trajectory_from_response(result.raw_response)
        if trajectory is None:
            trajectory = geometry.trajectory_from_action(
                result.action_index, step_m=self.step_m, turn_deg=self.turn_deg)

        s1_ms, s2_ms = self._timings(result.raw_response)
        return PolicyResult(
            trajectory=trajectory,
            stop=(trajectory is None),
            metadata={
                "action": result.action,
                "action_index": result.action_index,
                "waypoint_px": result.waypoint,
                "inference_ms": result.inference_time_ms,
                "s1_ms": s1_ms,
                "s2_ms": s2_ms,
                "raw": result.raw_response,
            },
        )

    @staticmethod
    def _timings(raw):
        # type: (Optional[dict]) -> tuple
        """Pull (s1_ms, s2_ms) from the response, top-level or inside action[0].

        The trajectory-patched server reports System-1 and System-2 inference
        time per step; either may be absent (an unpatched server, or a step
        where System 2 did not run), in which case that half is ``None``.
        """
        if not isinstance(raw, dict):
            return (None, None)
        holder = raw
        inner = raw.get("action")
        if isinstance(inner, list) and inner and isinstance(inner[0], dict):
            holder = inner[0]
        s1 = holder.get("s1_ms", raw.get("s1_ms"))
        s2 = holder.get("s2_ms", raw.get("s2_ms"))
        return (s1, s2)

    @staticmethod
    def _to_bgr(rgb):
        # type: (np.ndarray) -> np.ndarray
        """RGB (the observation contract) to BGR (what the server pipeline reads).

        The deployed bridge fed the server ``bgr8``; a policy handed the uniform
        contract's RGB must flip the channels or every colour cue is inverted.
        Kept to a numpy view reversal so no ``cv2`` import is pulled in.
        """
        frame = np.asarray(rgb)
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and float(np.nanmax(frame)) <= 1.0:
                frame = (frame * 255.0).clip(0, 255)
            frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        return np.ascontiguousarray(frame)

    @staticmethod
    def _to_depth(depth_m):
        # type: (Optional[np.ndarray]) -> Optional[np.ndarray]
        """Shape metric depth to the ``(H, W, 1)`` float32 the client sends."""
        if depth_m is None:
            return None
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[:, :, None]
        return np.ascontiguousarray(depth)


