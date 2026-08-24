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
from sparx_agency.core.planning.vlas.internvla_n1.types import (
    NON_TERMINAL_IDLE_INDICES,
)

# Metres that map to 1.0 on the wire. Set by the agent, which multiplies the
# array it receives by 10 before clipping -- see `_to_depth`.
DEPTH_RANGE_M = 10.0


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
        self._last_waypoint_step = None
        self._waypoint_age = 0

    def reset(self, observation=None):
        """Initialise the server agent (idempotent) and clear its episode state.

        Args:
            observation: unused; accepted for contract parity with policies that
                need the camera model up front. N1 takes its intrinsics from
                ``model_settings`` at construction instead.

        Returns:
            ``True`` once the agent exists and its per-episode state is cleared.
        """
        self._last_waypoint_step = None
        self._waypoint_age = 0
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
            A :class:`PolicyResult`. A transport drop returns a not-``ok``
            result (the caller re-sends next frame); a genuine STOP returns a
            not-``ok`` result with ``stop=True``; a tick that simply carries no
            new decision -- a look-down, or a System-1 call that produced no
            actions -- returns a not-``ok`` result with ``stop=False`` and
            ``idle=True`` in its metadata, which a runner must read as "keep
            flying what you already committed to".

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
        from_curve = trajectory is not None
        if trajectory is None:
            trajectory = geometry.trajectory_from_action(
                result.action_index, step_m=self.step_m, turn_deg=self.turn_deg)

        # An idle tick is NOT a stop, and the difference is the whole point of
        # the distinction. The agent reports -1 while a look-down is in progress
        # and when System 1 returns no actions; both mean "ask me again", not
        # "the task is done". Folding them into STOP -- which is what any
        # `index not in the map -> STOP` default does -- makes a runner abandon
        # a route it is halfway through and hold station until something else
        # happens. System 2 emitted a real STOP zero times across five hospital
        # flights; the agent emitted -1 seventeen times.
        idle = result.action_index in NON_TERMINAL_IDLE_INDICES

        # Is this pixel goal NEW, or the same one the agent has been holding on
        # to since the last System-2 call? A goal is a pixel in the frame System
        # 2 saw; once the aircraft has moved, it no longer points where it
        # meant, so a consumer that draws it has to know how old it is.
        fresh = (result.waypoint is not None
                 and result.waypoint_step is not None
                 and result.waypoint_step != self._last_waypoint_step)
        if result.waypoint_step is not None:
            if result.waypoint_step != self._last_waypoint_step:
                self._last_waypoint_step = result.waypoint_step
                self._waypoint_age = 0
            else:
                self._waypoint_age += 1

        s1_ms, s2_ms = self._timings(result.raw_response)
        return PolicyResult(
            trajectory=trajectory,
            stop=(trajectory is None and not idle),
            metadata={
                "action": result.action,
                "action_index": result.action_index,
                "idle": idle,
                "look_down": bool(result.look_down),
                "from_curve": from_curve,
                "waypoint_px": result.waypoint,
                "waypoint_step": result.waypoint_step,
                "waypoint_fresh": fresh,
                "waypoint_age_steps": (self._waypoint_age
                                       if result.waypoint is not None else None),
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
    def _to_depth(depth_m, range_m=DEPTH_RANGE_M):
        # type: (Optional[np.ndarray], float) -> Optional[np.ndarray]
        """Shape metric depth to the ``(H, W, 1)`` float32 the wire expects.

        **The wire is normalised, not metric.** The agent's System-1 path does
        ``depth * 10.0`` and then clips at its 5 m threshold, with the source
        comment ``# should be 0-10m`` -- so the array it wants is ``0..1`` over
        a 10 m range, and metres are off by exactly that factor.

        Sending metres is not a scale error that shows up as a wrong number; it
        destroys the channel. A wall at 3 m arrives as 30 and is clipped to 5,
        and so is everything else beyond 0.5 m, so System 1 sees a **flat plane
        at the clip distance** and plans its trajectory with no depth
        information at all. Nothing errors, nothing looks wrong, and the curve
        is simply computed blind.

        Args:
            depth_m: Metric depth, ``(H, W)`` or ``(H, W, 1)`` float metres.
            range_m: Metres that map to 1.0.

        Returns:
            ``(H, W, 1)`` float32 in ``[0, 1]``, or ``None``. Non-finite
            samples (the sky, and a Gazebo depth camera's misses) become 1.0 --
            "as far as this sensor can see" -- rather than a NaN the model
            would propagate into its own latents.
        """
        if depth_m is None:
            return None
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[:, :, None]
        depth = depth / float(range_m)
        depth = np.where(np.isfinite(depth), depth, 1.0)
        return np.ascontiguousarray(np.clip(depth, 0.0, 1.0).astype(np.float32))


