#!/usr/bin/env python3
"""HTTP Client for InternNav Agent Server.

The transport -- URL, timeout, pooled session, lazy ``requests`` import, logging
-- is :class:`~sparx_agency.core.planning.vlas.common.http_client.HttpPolicyClient`,
shared with NavDP and FlowNav. What is N1's own, and stays here, is the wire
contract: an agent that must be *created* before it can be stepped, an
observation that travels as base64 pickle rather than PNG, and a response whose
action index has to be read out of a nest of lists.

The session routes exist because of that create step: 201, 409 and a timeout
mean three different things to ``/agent/init``, so it uses the shared
``_attempt`` and judges the status itself, while ``step`` is an ordinary
policy call where anything but 200 is "no answer this frame".
"""

import base64
import pickle
import time
from typing import Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.vlas.common.http_client import HttpPolicyClient
from sparx_agency.core.planning.vlas.internvla_n1.errors import InternVlaError
from sparx_agency.core.planning.vlas.internvla_n1.types import (
    INDEX_TO_ACTION,
    StepResponse,
)


class ModelClient(HttpPolicyClient):
    """HTTP client for InternNav Agent Server API."""

    #: Malformed content raises this; transport failures do not raise at all.
    error_cls = InternVlaError

    DEFAULT_MODEL_SETTINGS = {
        "policy_name": "InternVLAN1_Policy",
        "state_encoder": None,
        "env_num": 1,
        "sim_num": 1,
        "model_path": "checkpoints/InternVLA-N1-DualVLN",
        "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        "width": 640,
        "height": 480,
        "hfov": 79,
        "resize_w": 384,
        "resize_h": 384,
        "max_new_tokens": 1024,
        "num_frames": 32,
        "num_history": 8,
        "num_future_steps": 4,
        "device": "cuda:0",
        "predict_step_nums": 32,
        "continuous_traj": True,
        "infer_mode": "partial_async",
        "vis_debug": False,
    }

    def __init__(self, host: str = "localhost", port: int = 8000,
                 timeout: float = 30.0, protocol: str = "http", logger=None):
        # Pooled: the observation is a base64 pickle of an RGB-D pair, several
        # times a second, to the same host.
        HttpPolicyClient.__init__(self, f"{protocol}://{host}:{port}",
                                  timeout_s=timeout, logger=logger, pooled=True)
        self.agent_name = "internvla_n1"
        self.initialized = False
        #: What the last :meth:`init_agent` asked for, replayed by :meth:`step`.
        self._init_args = None

    @property
    def base_url(self) -> str:
        """The server root. Spelled ``url`` on the shared client."""
        return self.url

    @property
    def timeout(self) -> float:
        """Per-request timeout. Spelled ``timeout_s`` on the shared client."""
        return self.timeout_s

    def check_health(self) -> bool:
        attempt = self._attempt("get", "/openapi.json", "health",
                                timeout_s=5.0, quiet=True)
        if not attempt.arrived:
            self._say("warn", f"Health check failed: {attempt.error}")
            return False
        return attempt.status == 200

    def check_agent_exists(self) -> bool:
        # Quiet: before the agent is created this is *expected* to fail, and
        # once a flight is running it is asked on every step that finds the
        # client uninitialised. Logging it would bury the one line that matters.
        attempt = self._attempt("post", f"/agent/{self.agent_name}/reset",
                                "agent probe", timeout_s=5.0, quiet=True,
                                json={"reset_index": None})
        if attempt.status == 200:
            self._say("info", f"Agent '{self.agent_name}' already exists")
            return True
        return False

    def init_agent(self, model_name: str = "InternVLA-N1",
                   ckpt_path: str = "", model_settings: Optional[Dict] = None) -> bool:
        # Remember what we were asked for, so a later re-init asks for the same
        # thing. `step()` re-initialises when the agent is missing, and it has no
        # arguments to pass -- so without this it falls back to the signature
        # default above, which is NOT the name any caller uses. InternNav
        # registers the agent as `internvla_n1`; `InternVLA-N1` is a 500 with
        # `KeyError` server-side, forever, on every frame. The failure looks like
        # a dead server rather than a wrong argument, and the one correct attempt
        # (the policy's `reset()`) is buried a hundred lines above the retries.
        self._init_args = (model_name, ckpt_path, model_settings)
        if self.check_agent_exists():
            self._say("info", "Agent already initialized, skipping")
            self.initialized = True
            return True

        final_settings = self.DEFAULT_MODEL_SETTINGS.copy()
        if model_settings:
            final_settings.update(model_settings)

        payload = {
            "agent_config": {
                "model_name": model_name,
                "ckpt_path": ckpt_path,
                "model_settings": final_settings,
            }
        }

        self._say("info", f"Initializing agent '{model_name}'...")
        # x3: the first init is where the checkpoint actually loads.
        attempt = self._attempt("post", "/agent/init", "init", quiet=True,
                                timeout_s=self.timeout_s * 3, json=payload)
        if not attempt.arrived:
            if attempt.timed_out():
                # Treated as success on purpose: the server is loading the
                # model and will have the agent by the next step. See the
                # upstream README -- pre-warm rather than rely on this.
                self._say("warn", "Init timeout (model loading)")
                self.initialized = True
                return True
            self._say("error", f"Init error: {attempt.error}")
            return False

        response = attempt.response
        if response.status_code == 201:
            self.initialized = True
            data = response.json()
            self.agent_name = data.get("agent_name", self.agent_name)
            self._say("info", f"Agent initialized: {self.agent_name}")
            return True
        self._say("error", f"Init failed: HTTP {response.status_code}")
        if "already" in response.text.lower() or response.status_code == 409:
            self.initialized = True
            return True
        return False

    def _reinit(self) -> bool:
        """Re-run the last :meth:`init_agent`, or the default if there was none.

        The server can lose its agent under a running client -- it restarted, or
        it was never up when the policy first called ``reset()``. Recovering is
        right; recovering with *different arguments* is not, and that is what a
        bare ``init_agent()`` here did.
        """
        if self._init_args is None:
            return self.init_agent()
        model_name, ckpt_path, model_settings = self._init_args
        return self.init_agent(model_name=model_name, ckpt_path=ckpt_path,
                               model_settings=model_settings)

    def reset(self, reset_index: Optional[List[int]] = None) -> bool:
        attempt = self._attempt("post", f"/agent/{self.agent_name}/reset", "reset",
                                quiet=True, json={"reset_index": reset_index})
        if not attempt.arrived:
            self._say("warn", f"Reset error: {attempt.error}")
            return False
        return attempt.status == 200

    def step(self, rgb: np.ndarray, instruction: str,
             depth: Optional[np.ndarray] = None) -> StepResponse:
        if not self.initialized:
            if self.check_agent_exists():
                self.initialized = True
            elif not self._reinit():
                return StepResponse(success=False, error="Agent not initialized")

        obs = [{
            'rgb': rgb,
            'depth': depth if depth is not None else np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32),
            'instruction': instruction
        }]
        encoded_obs = base64.b64encode(pickle.dumps(obs)).decode('utf-8')

        start_time = time.time()
        attempt = self._attempt("post", f"/agent/{self.agent_name}/step", "step",
                                quiet=True, json={"observation": encoded_obs},
                                headers={'Content-Type': 'application/json'})
        inference_time = (time.time() - start_time) * 1000

        if not attempt.arrived:
            if attempt.timed_out():
                return StepResponse(success=False,
                                    error=f"Timeout after {self.timeout_s}s")
            return StepResponse(success=False, error=str(attempt.error))
        if attempt.status == 200:
            return self._parse_response(attempt.response.json(), inference_time)
        return StepResponse(success=False, error=f"HTTP {attempt.status}",
                            inference_time_ms=inference_time)

    def _parse_response(self, data: Dict, inference_time: float) -> StepResponse:
        # `None`, NOT 0. Starting at 0 means every response this parser cannot
        # read -- a missing `action`, an empty inner list, a null element, a
        # body from a server that answered 200 with something else entirely --
        # decodes as index 0, which is STOP, which the caller reads as "the
        # policy has completed the task" and acts on by abandoning its route.
        # An unreadable answer is a transport failure, not a decision.
        action = None
        action_index = None
        waypoint = None

        # Debug: log response structure (use debug level to avoid flooding)
        self._say("debug", f"Server response keys: {list(data.keys())}")
        for k, v in data.items():
            if k != "action":
                self._say("debug", f"  response['{k}'] = {repr(v)[:200]}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                self._say("debug", f"  response['action'][0] keys: {list(v[0].keys())}")

        try:
            if "action" in data:
                action_data = data["action"]
                if isinstance(action_data, list) and len(action_data) > 0:
                    first = action_data[0]
                    if isinstance(first, dict) and "action" in first:
                        inner = first["action"]
                        if isinstance(inner, list) and len(inner) > 0:
                            action_index = int(inner[0])
                        elif isinstance(inner, (int, float)):
                            action_index = int(inner)
                    elif isinstance(first, (int, float)):
                        action_index = int(first)
                elif isinstance(action_data, (int, float)):
                    action_index = int(action_data)

            if action_index is not None:
                action = INDEX_TO_ACTION.get(action_index)
                if action is None:
                    # A real index the map has never heard of. Say so rather
                    # than folding it into STOP; the caller can then treat it
                    # as "no usable decision" instead of "we have arrived".
                    self._say("warn", f"unknown action index {action_index}")
                    action = "UNKNOWN"
        except Exception as e:
            self._say("warn", f"Parse error: {e}")
            action_index = None

        if action_index is None:
            return StepResponse(
                success=False, raw_response=data, inference_time_ms=inference_time,
                error="response carried no readable action index")

        # --- Extract S2 waypoint pixel coordinates ---
        waypoint = self._extract_waypoint(data)
        waypoint_step = data.get("pixel_goal_step")
        try:
            waypoint_step = int(waypoint_step) if waypoint_step is not None else None
        except (TypeError, ValueError):
            waypoint_step = None
        if waypoint_step is not None and waypoint_step < 0:
            waypoint_step = None      # the agent's "never set" sentinel

        look_down = bool(data.get("look_down"))
        if not look_down:
            action_data = data.get("action")
            if isinstance(action_data, list) and action_data and isinstance(action_data[0], dict):
                look_down = bool(action_data[0].get("look_down"))
        if waypoint:
            self._say("debug", f"S2 waypoint: ({waypoint[0]}, {waypoint[1]})")
        else:
            self._say("debug", "No S2 waypoint this step")

        return StepResponse(action=action, action_index=action_index, waypoint=waypoint,
                           waypoint_step=waypoint_step, look_down=look_down,
                           raw_response=data, inference_time_ms=inference_time,
                           success=True)

    def _extract_waypoint(self, data: Dict):
        """Extract S2 waypoint pixel coords from server response.

        The server returns the waypoint as [y, x] (numpy row,col convention).
        We convert to (x, y) for OpenCV drawing.
        Searches multiple possible locations in the response JSON.
        """
        # Keys to search for waypoint data (pixel_goal is the key from patched server)
        wp_keys = ("pixel_goal", "waypoint", "pixel_point", "target_point", "s2_output", "subgoal", "pixel")

        # Check top-level
        for key in wp_keys:
            val = data.get(key)
            if val is not None:
                return self._parse_waypoint_value(val)

        # Check inside action[0] dict
        action_data = data.get("action")
        if isinstance(action_data, list) and len(action_data) > 0:
            first = action_data[0]
            if isinstance(first, dict):
                for key in wp_keys:
                    val = first.get(key)
                    if val is not None:
                        return self._parse_waypoint_value(val)

        return None

    @staticmethod
    def _parse_waypoint_value(val):
        """Parse waypoint value into (x, y) tuple.

        Server returns [y, x] (numpy convention). We return (x, y) for drawing.
        """
        try:
            if isinstance(val, (list, tuple)):
                coords = val
                # If nested: [[y, x]] -> take first
                if len(coords) > 0 and isinstance(coords[0], (list, tuple)):
                    coords = coords[0]
                if len(coords) >= 2:
                    y, x = float(coords[0]), float(coords[1])
                    return (int(x), int(y))
            elif isinstance(val, dict):
                if 'x' in val and 'y' in val:
                    return (int(val['x']), int(val['y']))
        except (ValueError, TypeError, IndexError):
            pass
        return None