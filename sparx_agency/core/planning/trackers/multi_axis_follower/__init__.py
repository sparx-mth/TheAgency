"""Multi-axis waypoint follower (combined forward + lateral + yaw, fixed altitude).

The multi-axis counterpart of ``waypoint_follower``: reaches each waypoint by
translating directly toward it (forward + lateral / "crab"), engaging yaw only
past a deadband so the noisy yaw axis is used sparingly, and station-keeping with
a decisive deadband at the goal. Drop-in compatible public API.
"""
from .params import MultiAxisFollowerParams
from .types import MultiAxisCommand, MultiAxisState
from .follower import MultiAxisFollower
from .predictor import (
    MotionModelParams,
    PredictionResult,
    predict_trajectory,
    prediction_score,
)

__all__ = [
    "MultiAxisFollowerParams",
    "MultiAxisFollower",
    "MultiAxisCommand",
    "MultiAxisState",
    "MotionModelParams",
    "PredictionResult",
    "predict_trajectory",
    "prediction_score",
]
