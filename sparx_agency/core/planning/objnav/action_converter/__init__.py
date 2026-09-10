"""Continuous waypoints in, one benchmark action out, assuming perfect execution.

* :mod:`.path_geometry` -- pure path geometry: length, projection onto the
  path, and the point a distance along it.
* :mod:`.action_choice` -- the heading error, and the one-step turn and tilt
  choices that close it.
* :mod:`.checks` -- the number checks both of those refuse bad input with.
* :mod:`.path_progress` -- the adopted path and the forward-only progress
  along it.
* :mod:`.transition` -- what one action does to the pose.
* :mod:`.ladder` -- the stateless decision ladder: tilt, face, arrive, turn
  or step forward.
* :mod:`.converter` -- :class:`DiscreteActionConverter`, the state around
  the ladder: STOP, the path adopted, the blocked-step recovery, and the
  rollout.
* :mod:`.params` / :mod:`.types` -- its tuning and its results.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    ANGLE_EPS_RAD,
    PITCH_LIMIT_EPS_RAD,
    heading_error,
    pitch_action,
    pitch_within_limits,
    turn_action,
)
from sparx_agency.core.planning.objnav.action_converter.converter import (
    DiscreteActionConverter,
)
from sparx_agency.core.planning.objnav.action_converter.ladder import (
    parallel_offset,
    reach_floor,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.path_geometry import (
    arclength_beyond,
    path_length,
    point_at_arclength,
    project_onto_path,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    ACTING_STATUSES,
    IDLE_STATUSES,
    ROLLOUT_ENDINGS,
    ROLLOUT_LIMIT,
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_FACE,
    STATUS_FORWARD,
    STATUS_PITCH,
    STATUS_STOP,
    STATUS_TURN,
    STATUSES,
    ConversionResult,
    Rollout,
)

__all__ = [
    # the converter
    "DiscreteActionConverter",
    "ActionConverterParams",
    "apply_action",
    # what it returns
    "ConversionResult",
    "Rollout",
    "STATUS_STOP",
    "STATUS_PITCH",
    "STATUS_TURN",
    "STATUS_FORWARD",
    "STATUS_FACE",
    "STATUS_ARRIVED",
    "STATUS_EMPTY",
    "ACTING_STATUSES",
    "IDLE_STATUSES",
    "STATUSES",
    "ROLLOUT_LIMIT",
    "ROLLOUT_ENDINGS",
    # the geometry it decides with
    "ANGLE_EPS_RAD",
    "PITCH_LIMIT_EPS_RAD",
    "reach_floor",
    "parallel_offset",
    "path_length",
    "project_onto_path",
    "point_at_arclength",
    "arclength_beyond",
    "heading_error",
    "turn_action",
    "pitch_action",
    "pitch_within_limits",
]
