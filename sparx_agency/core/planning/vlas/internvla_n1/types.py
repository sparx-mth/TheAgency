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
    LOOK_DOWN = "LOOK_DOWN"
    NO_ACTION = "NO_ACTION"
    UNKNOWN = "UNKNOWN"


# Action index mapping (VLN-CE standard), plus the two indices InternVLA-N1's
# agent adds on top of it.
#
# 5 is LOOK_DOWN -- System 2 answering with the down arrow, which tilts the
# camera for one turn of conversation and then decides again.
#
# -1 is NOT AN ACTION AT ALL. The agent emits it while a look-down is in
# progress and whenever System 1 returns an empty action list. It means "no new
# decision this tick", and the difference matters: mapping it to STOP -- which
# is what a plain `.get(idx, "STOP")` does -- tells the caller the policy has
# finished the task. A caller that believes it throws away a route it was
# halfway through flying, brakes, and waits for an instruction that has already
# been given. Measured over five hospital flights: System 2 said STOP zero
# times and the agent emitted -1 seventeen times.
STOP_INDEX = 0
NO_ACTION_INDEX = -1
LOOK_DOWN_INDEX = 5

ACTION_INDEX = {
    "STOP": STOP_INDEX,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
    "LOOK_DOWN": LOOK_DOWN_INDEX,
    "NO_ACTION": NO_ACTION_INDEX,
}
INDEX_TO_ACTION = {v: k for k, v in ACTION_INDEX.items()}

# The indices that carry no motion and are NOT a request to stop. A tick
# reporting one of these should leave any existing commitment alone.
NON_TERMINAL_IDLE_INDICES = frozenset((NO_ACTION_INDEX, LOOK_DOWN_INDEX))


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
    # The episode step the waypoint was computed at, straight off the wire.
    #
    # Without it a waypoint cannot be told apart from the SAME waypoint eight
    # steps later: the patched agent deliberately keeps the last pixel goal
    # alive between System-2 calls, so `waypoint` is non-null almost always and
    # says nothing about whether it is current. It is a pixel in the frame
    # System 2 saw, so the moment the aircraft moves it no longer points at the
    # world position it meant, and drawing it on the live frame as though it
    # did is a lie that looks like telemetry.
    waypoint_step: Optional[int] = None
    # System 2 has asked to look down, and expects the NEXT frame to be that
    # lower view -- the pixel goal it then gives is a pixel in that frame. The
    # action index cannot say so: the agent overwrites the look-down action
    # with -1, which is also what an empty System-1 list reports.
    look_down: bool = False
    raw_response: Dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    success: bool = True
    error: str = ""