#!/usr/bin/env python3
"""HTTP Client for InternNav Agent Server."""

import base64
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import requests

from .types import StepResponse, INDEX_TO_ACTION


class ModelClient:
    """HTTP client for InternNav Agent Server API."""

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
        self.base_url = f"{protocol}://{host}:{port}"
        self.timeout = timeout
        self.logger = logger
        self.session = requests.Session()
        self.agent_name = "internvla_n1"
        self.initialized = False

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def check_health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/openapi.json", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self._log("warn", f"Health check failed: {e}")
            return False

    def check_agent_exists(self) -> bool:
        try:
            url = f"{self.base_url}/agent/{self.agent_name}/reset"
            response = self.session.post(url, json={"reset_index": None}, timeout=5.0)
            if response.status_code == 200:
                self._log("info", f"Agent '{self.agent_name}' already exists")
                return True
            return False
        except Exception:
            return False

    def init_agent(self, model_name: str = "InternVLA-N1",
                   ckpt_path: str = "", model_settings: Optional[Dict] = None) -> bool:
        if self.check_agent_exists():
            self._log("info", "Agent already initialized, skipping")
            self.initialized = True
            return True

        url = f"{self.base_url}/agent/init"
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

        try:
            self._log("info", f"Initializing agent '{model_name}'...")
            response = self.session.post(url, json=payload, timeout=self.timeout * 3)

            if response.status_code == 201:
                self.initialized = True
                data = response.json()
                self.agent_name = data.get("agent_name", self.agent_name)
                self._log("info", f"Agent initialized: {self.agent_name}")
                return True
            else:
                self._log("error", f"Init failed: HTTP {response.status_code}")
                if "already" in response.text.lower() or response.status_code == 409:
                    self.initialized = True
                    return True
                return False
        except requests.exceptions.Timeout:
            self._log("warn", "Init timeout (model loading)")
            self.initialized = True
            return True
        except Exception as e:
            self._log("error", f"Init error: {e}")
            return False

    def reset(self, reset_index: Optional[List[int]] = None) -> bool:
        url = f"{self.base_url}/agent/{self.agent_name}/reset"
        try:
            response = self.session.post(url, json={"reset_index": reset_index}, timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            self._log("warn", f"Reset error: {e}")
            return False

    def step(self, rgb: np.ndarray, instruction: str,
             depth: Optional[np.ndarray] = None) -> StepResponse:
        if not self.initialized:
            if self.check_agent_exists():
                self.initialized = True
            elif not self.init_agent():
                return StepResponse(success=False, error="Agent not initialized")

        url = f"{self.base_url}/agent/{self.agent_name}/step"

        obs = [{
            'rgb': rgb,
            'depth': depth if depth is not None else np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32),
            'instruction': instruction
        }]
        encoded_obs = base64.b64encode(pickle.dumps(obs)).decode('utf-8')

        try:
            start_time = time.time()
            response = self.session.post(
                url, json={"observation": encoded_obs},
                timeout=self.timeout, headers={'Content-Type': 'application/json'}
            )
            inference_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return self._parse_response(response.json(), inference_time)
            else:
                return StepResponse(success=False, error=f"HTTP {response.status_code}",
                                   inference_time_ms=inference_time)
        except requests.exceptions.Timeout:
            return StepResponse(success=False, error=f"Timeout after {self.timeout}s")
        except Exception as e:
            return StepResponse(success=False, error=str(e))

    def _parse_response(self, data: Dict, inference_time: float) -> StepResponse:
        action = "STOP"
        action_index = 0
        waypoint = None

        # Debug: log response structure (use debug level to avoid flooding)
        self._log("debug", f"Server response keys: {list(data.keys())}")
        for k, v in data.items():
            if k != "action":
                self._log("debug", f"  response['{k}'] = {repr(v)[:200]}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                self._log("debug", f"  response['action'][0] keys: {list(v[0].keys())}")

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

            action = INDEX_TO_ACTION.get(action_index, "STOP")
        except Exception as e:
            self._log("warn", f"Parse error: {e}")

        # --- Extract S2 waypoint pixel coordinates ---
        waypoint = self._extract_waypoint(data)
        if waypoint:
            self._log("debug", f"S2 waypoint: ({waypoint[0]}, {waypoint[1]})")
        else:
            self._log("debug", "No S2 waypoint this step")

        return StepResponse(action=action, action_index=action_index, waypoint=waypoint,
                           raw_response=data, inference_time_ms=inference_time, success=True)

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