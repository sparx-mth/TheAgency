"""A simulator must execute the action geometry its episode advertises, in world ENU; each way it can fail to is refused and named.

Neither failure shows anywhere else: a simulator turning another angle than
the spec dithers to the step budget or scores normally under another
protocol, and a rotated or mirrored pose stream keeps every path length. The
tolerances must still pass real simulators -- AI2-THOR's actuation noise,
Habitat's clipped steps and stairs -- and the reference environments.
"""
from __future__ import annotations

import math
import random

import pytest

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import (
    fake_label_mapper,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_policy import (
    OracleSearchPolicy,
)
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance,
    check_motion,
)
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.smoke import build_demo_env
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    STOP,
    A,
    CorridorEnv,
    F,
    ScriptedAgent,
    one_episode,
)

SPEC = DiscreteActionSpec()
THOR = DiscreteActionSpec(min_pitch_deg=-30.0, max_pitch_deg=30.0)
TOLERANCE = KinematicTolerance()
#: Every kind of action once: a step, a turn, both LOOKs, a turn back, STOP.
EVERY_ACTION = (F, A.TURN_LEFT, A.LOOK_DOWN, A.LOOK_UP, A.TURN_RIGHT, F, F, STOP)
#: Turns between steps, so a frame bug shows in both.
FRAME_SCRIPT = (F, A.TURN_LEFT, F, F, A.TURN_RIGHT, F, F, STOP)


class MotionEnv(CorridorEnv):
    """The corridor, executing each action as ``motion(pose, action)`` says while its episodes advertise the default spec."""

    def __init__(self, motion, **options):
        super().__init__({"e1": 1.0}, **options)
        self.motion = motion

    def step(self, action):
        before = self.pose
        self.stopped = self.stopped or action == STOP
        self.pose = self.motion(before, action)
        self.path += math.dist(before.position(), self.pose.position())
        self.steps += 1
        return self.observation(self.steps, "step")


class ReportedPoseEnv(CorridorEnv):
    """The corridor, moving correctly but reporting every pose through ``report``: an adapter's frame bug."""

    def __init__(self, report, **options):
        super().__init__({"e1": 1.0}, **options)
        self.report = report

    def observation(self, step, phase):
        true_pose = self.pose
        self.pose = self.report(true_pose)
        try:
            return super().observation(step, phase)
        finally:
            self.pose = true_pose


def executes(spec):
    """A motion that executes ``spec`` perfectly."""
    return lambda pose, action: apply_action(pose, action, spec)


def y_up_unconverted(pose):
    """World ENU ``(x, y, z)`` read as Habitat's Y-up ``(-y, z, -x)`` and stored as if it were ENU."""
    return AgentPose(-pose.y, pose.z, -pose.x, pose.yaw, pose.camera_pitch)


def mirrored(pose):
    """AI2-THOR's left-handed frame with the flip left out: y and yaw mirrored."""
    return AgentPose(pose.x, -pose.y, pose.z, -pose.yaw, pose.camera_pitch)


def moved(pose, dx=0.0, dy=0.0, dz=0.0, dyaw_deg=0.0, dpitch_deg=0.0):
    """``pose`` displaced by the given amounts."""
    return AgentPose(pose.x + dx, pose.y + dy, pose.z + dz,
                     pose.yaw + math.radians(dyaw_deg),
                     pose.camera_pitch + math.radians(dpitch_deg))


def noisy_thor(pose, action, rng):
    """One LoCoBot action with AI2-THOR's documented actuation noise; a fifth of the moves blocked."""
    if action == F:
        yaw = normalize_angle(pose.yaw + math.radians(rng.gauss(0.0, 0.25)))
        if rng.random() < 0.2:  # a blocked MoveAhead still turns by the noise
            return AgentPose(pose.x, pose.y, pose.z, yaw, pose.camera_pitch)
        step = 0.25 + rng.gauss(0.001, 0.005)
        return AgentPose(pose.x + step * math.cos(yaw), pose.y + step * math.sin(yaw),
                         pose.z, yaw, pose.camera_pitch)
    if action in (A.TURN_LEFT, A.TURN_RIGHT):
        turn = math.radians(30.0 + rng.gauss(0.0, 0.5))
        if action == A.TURN_RIGHT:
            turn = -turn
        return AgentPose(pose.x, pose.y, pose.z, normalize_angle(pose.yaw + turn),
                         pose.camera_pitch)
    return apply_action(pose, action, THOR)  # a LOOK has no noise; STOP none


# -- what real simulators do passes ----------------------------------------------

@pytest.mark.parametrize("spec", [
    SPEC, THOR, DiscreteActionSpec(forward_step_m=0.5, turn_angle_deg=90.0,
                                   tilt_angle_deg=15.0)],
    ids=["habitat", "robothor", "coarse"])
def test_exact_execution_of_every_action_passes_from_any_pose(spec):
    """A simulator doing exactly what its spec says must never be refused; LOOKs at a limit included."""
    rng = random.Random(0)
    for _ in range(300):
        pose = AgentPose(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-1, 1),
                         rng.uniform(-10, 10), math.radians(rng.choice((-30.0, 0.0, 30.0))))
        for action in spec.actions:
            check_motion(action, pose, apply_action(pose, action, spec), spec, TOLERANCE)


def test_a_seeded_ai2thor_like_noisy_stream_passes():
    """AI2-THOR perturbs every turn and move; thousands of noisy actions must all pass."""
    rng = random.Random(3)
    pose = AgentPose(0.0, 0.0)
    for _ in range(5000):
        action = rng.choice(THOR.actions)
        after = noisy_thor(pose, action, rng)
        check_motion(action, pose, after, THOR, TOLERANCE)
        pose = after


@pytest.mark.parametrize("advance", [0.0, 0.004, 0.12, 0.25],
                         ids=["blocked", "noise", "clipped", "full"])
def test_a_habitat_like_partial_or_blocked_forward_step_passes(advance):
    """Habitat without sliding clips a step at a wall, and AI2-THOR fails one outright; both are legal."""
    pose = AgentPose(1.0, 2.0, 0.0, math.radians(40.0))
    after = moved(pose, advance * math.cos(pose.yaw), advance * math.sin(pose.yaw))
    check_motion(F, pose, after, SPEC, TOLERANCE)


def test_a_step_too_short_to_have_a_direction_is_not_checked_for_one():
    """A blocked step moves by noise alone; refusing its direction would refuse AI2-THOR."""
    pose = AgentPose(0.0, 0.0)
    check_motion(F, pose, moved(pose, dy=0.03), SPEC, TOLERANCE)


def test_a_stair_riser_on_a_forward_step_passes_and_a_jump_does_not():
    """Habitat's stairs climb up to 0.2 m on one step; a bigger height change on a short step is a frame bug."""
    pose = AgentPose(0.0, 0.0)
    check_motion(F, pose, moved(pose, dx=0.25, dz=0.18), SPEC, TOLERANCE)
    check_motion(F, pose, moved(pose, dx=0.25, dz=-0.18), SPEC, TOLERANCE)
    with pytest.raises(EnvContractError, match="height by at most 0.200 m.*Y-up"):
        check_motion(F, pose, moved(pose, dx=0.1, dz=0.3), SPEC, TOLERANCE)


def test_a_look_the_simulator_refused_at_its_pitch_limit_passes():
    """AI2-THOR fails a LOOK past +/-30 degrees and spends the step; that is the simulator obeying its limits."""
    pose = AgentPose(0.0, 0.0, camera_pitch=math.radians(30.0))
    check_motion(A.LOOK_DOWN, pose, pose, THOR, TOLERANCE)
    check_motion(A.LOOK_UP, pose, moved(pose, dpitch_deg=-30.0), THOR, TOLERANCE)


def test_the_fake_environment_and_the_corridor_pass_the_default_check():
    """The reference environments must satisfy the contract they exist to demonstrate."""
    record = one_episode(CorridorEnv({"e1": 1.0}), ScriptedAgent(EVERY_ACTION))
    assert record.steps == len(EVERY_ACTION)
    env = build_demo_env(2)
    agent = HeadlessObjNavAgent(OracleSearchPolicy(env), fake_label_mapper())
    summary = run_benchmark(env, agent, kinematics=KinematicTolerance())
    assert summary.overall.success_rate == 1.0


# -- what the spec cannot explain is refused ---------------------------------------

@pytest.mark.parametrize("real_turn_deg", [90.0, 10.0])
def test_a_simulator_turning_another_angle_is_refused_at_its_first_turn(real_turn_deg):
    """90 degrees against a 30-degree spec dithers to the budget; 10 scores under another protocol."""
    env = MotionEnv(executes(DiscreteActionSpec(turn_angle_deg=real_turn_deg)))
    with pytest.raises(EnvContractError,
                       match=r"action 2: TURN_LEFT should turn \+30\.0 deg.*turned "
                             r"\+%.2f deg.*another turn than the spec's 30 deg"
                             % real_turn_deg):
        one_episode(env, ScriptedAgent(EVERY_ACTION))
    assert env.steps == 2


def test_a_simulator_tilting_another_angle_is_refused_at_its_first_look():
    """habitat-sim's own 15-degree tilt against a 30-degree spec leaves the camera at half the pitch asked for."""
    env = MotionEnv(executes(DiscreteActionSpec(tilt_angle_deg=15.0)))
    with pytest.raises(EnvContractError,
                       match=r"action 3: LOOK_DOWN should tilt \+30\.0 deg.*tilted "
                             r"\+15\.00 deg.*another tilt than the spec's 30 deg"):
        one_episode(env, ScriptedAgent(EVERY_ACTION))


@pytest.mark.parametrize("report,cause", [
    (y_up_unconverted, "Y-up frame"), (mirrored, "mirrored handedness")],
    ids=["y_up_unconverted", "mirrored"])
def test_a_pose_stream_in_the_wrong_frame_is_refused_though_its_path_length_agrees(
        report, cause):
    """A rotated or mirrored frame keeps every chord, so the path check passes it; only the motion check sees it."""
    unchecked = one_episode(ReportedPoseEnv(report), ScriptedAgent(FRAME_SCRIPT),
                            kinematics=None)
    assert unchecked.observed_path_length_m == pytest.approx(unchecked.path_length_m)
    with pytest.raises(EnvContractError, match=cause):
        one_episode(ReportedPoseEnv(report), ScriptedAgent(FRAME_SCRIPT))


START = AgentPose(0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("action,after,message", [
    (STOP, moved(START, dx=0.1), "STOP should leave the pose unchanged"),
    (STOP, moved(START, dyaw_deg=5.0), "STOP should keep the heading"),
    (A.TURN_LEFT, moved(START, dx=0.1, dyaw_deg=30.0), "TURN_LEFT should turn in place"),
    (A.TURN_RIGHT, moved(START, dyaw_deg=-30.0, dpitch_deg=5.0),
     "TURN_RIGHT should keep the camera pitch"),
    (A.TURN_RIGHT, moved(START, dyaw_deg=30.0), "mirrored handedness"),
    (A.LOOK_DOWN, moved(START, dpitch_deg=-30.0), "signed the other way"),
    (A.LOOK_DOWN, moved(START, dpitch_deg=30.0, dyaw_deg=5.0),
     "LOOK_DOWN should keep the heading"),
    (A.LOOK_UP, moved(START, dx=0.1, dpitch_deg=-30.0), "LOOK_UP should tilt in place"),
    (F, moved(START, dx=0.4), "advance at most 0.280 m"),
    (F, moved(START, dx=0.18, dy=0.18), "advance along its heading"),
    (F, moved(START, dx=0.25, dyaw_deg=5.0), "MOVE_FORWARD should keep the heading"),
    (F, moved(START, dx=0.25, dpitch_deg=5.0), "MOVE_FORWARD should keep the camera pitch"),
], ids=["stop_moves", "stop_turns", "turn_moves", "turn_tilts", "turn_mirrored",
        "look_sign", "look_turns", "look_moves", "step_too_long", "step_off_heading",
        "step_turns", "step_tilts"])
def test_each_motion_the_spec_cannot_explain_is_refused_and_named(action, after, message):
    """Each symptom points at a different adapter bug; the message must say which."""
    with pytest.raises(EnvContractError, match=message):
        check_motion(action, START, after, SPEC, TOLERANCE)


# -- the option ---------------------------------------------------------------------

def test_kinematics_none_disables_the_check():
    """None trusts a simulator whose motion was verified otherwise: the 90-degree turner then runs to its end."""
    env = MotionEnv(executes(DiscreteActionSpec(turn_angle_deg=90.0)))
    record = one_episode(env, ScriptedAgent(EVERY_ACTION), kinematics=None)
    assert record.steps == len(EVERY_ACTION)


@pytest.mark.parametrize("field,value", [
    ("turn_deg", 0.0), ("position_m", -0.01), ("heading_deg", float("nan")),
    ("climb_m", float("inf")), ("yaw_deg", True), ("pitch_deg", "0.5")])
def test_a_tolerance_that_is_not_a_positive_finite_number_is_refused(field, value):
    """A zero tolerance refuses rounding, a NaN one passes everything; neither may reach a run."""
    with pytest.raises(HarnessError, match="KinematicTolerance.%s" % field):
        KinematicTolerance(**{field: value})


def test_a_kinematics_option_of_the_wrong_type_is_refused_before_the_episode_starts():
    """A dict of tolerances would otherwise fail only at the first step."""
    env = CorridorEnv({"e1": 1.0})
    with pytest.raises(TypeError, match="kinematics must be a KinematicTolerance"):
        one_episode(env, kinematics={"turn_deg": 5.0})
    assert env.resets == []
    with pytest.raises(TypeError, match="action must be a DiscreteAction"):
        check_motion(1, START, START, SPEC, TOLERANCE)
