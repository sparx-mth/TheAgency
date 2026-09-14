"""Explicit synthetic geometry and method settings, not a dataset profile."""
from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSettings


def camera():
    return CameraSpec(Intrinsics(640, 480, 400.0, 400.0, 319.5, 239.5),
                      height_m=1.0, min_depth_m=0.1, max_depth_m=5.0)


def actions():
    return DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0,
                              actions=(DiscreteAction.STOP, DiscreteAction.MOVE_FORWARD,
                                       DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT))


def settings(**overrides):
    values = dict(body_height_m=1.0, body_radius_m=0.2,
                  preferred_clearance_m=0.3, stop_distance_m=0.75, map_size_m=20.0)
    values.update(overrides)
    return RPTSettings(**values)

