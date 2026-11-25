"""The recording wrapper must not hide the checker-only hooks the runner reads off the environment."""
import inspect

import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark import runner
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import MP3D_V1
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import RecordingEnv

# oLBMNvg9in8/000093, actions 84-85 of the 2026-11-24 MP3D v1 val campaign, as habitat-sim 0.2.4 reproduces them.
BEFORE = AgentPose(0.4136446714401245, 2.8959312438964844, -0.7118054628372192, -0.13951241356045846)
PLACED = AgentPose(0.323266, 2.933589, -0.590003, -0.13951241356045846)
AFTER = AgentPose(0.4136446416378021, 2.8959312438964844, -0.5047420263290405, -0.6631112118904312)


class _Env:
    def __init__(self, placed):
        self._placed = placed

    def placed_pose(self):
        return self._placed

    def pose_confirmed_on_navmesh(self):
        return True


def test_the_recording_wrapper_forwards_every_checker_hook_the_runner_reads():
    hooks = set(name for name in ("placed_pose", "pose_confirmed_on_navmesh") if '"%s"' % name in inspect.getsource(runner._drive))
    assert hooks == {"placed_pose", "pose_confirmed_on_navmesh"}
    wrapped = RecordingEnv(_Env(PLACED), recorder=None)
    assert wrapped.placed_pose() is PLACED and wrapped.pose_confirmed_on_navmesh() is True
    assert RecordingEnv(object(), recorder=None).placed_pose() is None


def test_the_turn_that_ended_a_recorded_mp3d_shard_passes_once_the_placement_is_reported():
    spec, tolerance = MP3D_V1.actions(), MP3D_V1.kinematics()
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(DiscreteAction.TURN_RIGHT, BEFORE, AFTER, spec, tolerance, placed=None)
    check_motion(DiscreteAction.TURN_RIGHT, BEFORE, AFTER, spec, tolerance,
                 placed=RecordingEnv(_Env(PLACED), recorder=None).placed_pose())
    # A reported placement excuses the filter's own move, nothing else: a turn that also slides away is refused.
    slid = AgentPose(AFTER.x + 0.30, AFTER.y, AFTER.z, AFTER.yaw)
    with pytest.raises(EnvContractError):
        check_motion(DiscreteAction.TURN_RIGHT, BEFORE, slid, spec, tolerance, placed=PLACED)

