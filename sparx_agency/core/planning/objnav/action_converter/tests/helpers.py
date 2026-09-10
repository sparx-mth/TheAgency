"""The benchmark specs, the paths and the perfect-execution drive loop the converter tests share.

A test module, not a library: nothing outside these tests imports it.

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.types.actions import (
    NAVIGATION_ACTIONS,
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: Habitat's geometry: 0.25 m steps, 30-degree turns and tilts, no pitch limits.
HABITAT = DiscreteActionSpec()
#: AI2-THOR's: the same steps, the camera pitch limited to +-30 degrees.
THOR = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)
#: A benchmark without camera tilt.
NO_TILT = DiscreteActionSpec(actions=NAVIGATION_ACTIONS)

FORWARD = DiscreteAction.MOVE_FORWARD
LEFT = DiscreteAction.TURN_LEFT
RIGHT = DiscreteAction.TURN_RIGHT

#: 4 m east, 1 m north, 4 m back west: the return leg runs 1 m from the way out.
HAIRPIN = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0))
#: The origin, facing east.
ORIGIN = AgentPose(0.0, 0.0)


def drive(converter, pose, command, limit=300):
    """Step ``converter`` from ``pose`` under perfect execution until it idles or stops.

    Returns:
        ``(results, final_pose)``.
    """
    results = []
    for _ in range(limit):
        result = converter.step(pose, command)
        results.append(result)
        if result.idle or result.action == DiscreteAction.STOP:
            return results, pose
        pose = apply_action(pose, result.action, converter.spec)
    raise AssertionError("the converter did not finish in %d steps" % limit)
