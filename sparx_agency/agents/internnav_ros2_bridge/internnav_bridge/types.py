#!/usr/bin/env python3
"""Core types for the InternNav Bridge."""

from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque
from enum import Enum
import numpy as np


class ActionType(Enum):
    """Navigation action types."""
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    UNKNOWN = "UNKNOWN"


# Action index mapping (VLN-CE standard)
ACTION_INDEX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}
INDEX_TO_ACTION = {v: k for k, v in ACTION_INDEX.items()}


@dataclass
class BridgeState:
    """Current state of the bridge."""
    current_rgb: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    current_instruction: str = ""
    current_odometry: Optional[Dict] = None
    rgb_timestamp: float = 0.0
    depth_timestamp: float = 0.0
    last_inference_time: float = 0.0
    is_navigating: bool = False
    image_history: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class StepResponse:
    """Response from model inference."""
    action: str = "STOP"
    action_index: int = 0
    waypoint: Optional[tuple] = None  # (x, y) pixel coords from S2, in input image space
    raw_response: Dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    success: bool = True
    error: str = ""