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


def test_a_first_action_settle_onto_the_mesh_is_excused_only_when_the_protocol_allows_it():
    """Gibson val Darden/000000 starts 0.10 m below the navmesh; habitat-sim lifts the agent on the first
    action. Default tolerance: refused (a turn that moved 0.10 m). With ``settle_m``: the vertical part
    of the FIRST action is the simulator's; the horizontal bounds still hold, and later actions are not
    excused at all."""
    pose = AgentPose(1.0, 2.0, 0.0, math.radians(40.0))
    lifted = moved(pose, dz=0.10)
    turned = AgentPose(lifted.x, lifted.y, lifted.z, pose.yaw + SPEC.turn_angle_rad, pose.camera_pitch)
    with pytest.raises(EnvContractError):
        check_motion(A.TURN_LEFT, pose, turned, SPEC, TOLERANCE, first=True)
    settling = KinematicTolerance(settle_m=0.30)
    check_motion(A.TURN_LEFT, pose, turned, SPEC, settling, first=True)
    check_motion(F, pose, moved(pose, dx=0.25 * math.cos(pose.yaw), dy=0.25 * math.sin(pose.yaw), dz=0.28), SPEC,
                 settling, first=True)
    with pytest.raises(EnvContractError):
        check_motion(A.TURN_LEFT, pose, turned, SPEC, settling, first=False), "only the first action"
    with pytest.raises(EnvContractError):
        check_motion(A.TURN_LEFT, pose, AgentPose(lifted.x, lifted.y, 0.35, turned.yaw), SPEC, settling, first=True)
    with pytest.raises(EnvContractError):
        check_motion(A.TURN_LEFT, pose, AgentPose(lifted.x + 0.05, lifted.y, lifted.z, turned.yaw), SPEC, settling,
                     first=True), "the horizontal bound stands"
    assert KinematicTolerance().settle_m == 0.0, "off by default"
    with pytest.raises(HarnessError):
        KinematicTolerance(settle_m=-0.1)
    with pytest.raises(HarnessError):
        KinematicTolerance(position_m=0.0), "every other tolerance is still strictly positive"


def test_a_first_action_pull_onto_the_mesh_edge_is_excused_only_up_to_settle_xy_m():
    """Gibson val Wiconisco/000118 starts 0.033 m outside the navmesh; habitat-sim's step filter pulls the
    agent onto the mesh edge on the first body action -- the warm-up TURN_LEFT -- which the 0.02 m
    turn-in-place bound refused, ending the 1,000-episode run at episode 919. With ``settle_xy_m`` that
    much horizontal placement is taken off the FIRST action, along its own direction; a larger pull, a
    later action, and the vertical part (``settle_m``'s business) are still judged."""
    pose = AgentPose(1.0, 2.0, 0.0, math.radians(40.0))
    pulled = moved(pose, dx=0.033 * math.cos(pose.yaw + math.radians(-73.3)), dy=0.033 * math.sin(pose.yaw + math.radians(-73.3)))
    turned = AgentPose(pulled.x, pulled.y, pulled.z, pose.yaw + SPEC.turn_angle_rad, pose.camera_pitch)
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(A.TURN_LEFT, pose, turned, SPEC, KinematicTolerance(settle_m=0.30), first=True)
    gibson = KinematicTolerance(settle_m=0.30, settle_xy_m=0.05)
    check_motion(A.TURN_LEFT, pose, turned, SPEC, gibson, first=True)
    # The real episode: pulled 0.033 m sideways AND settled 0.064 m down on the same turn.
    check_motion(A.TURN_LEFT, pose, AgentPose(pulled.x, pulled.y, -0.064, turned.yaw), SPEC, gibson, first=True)
    # A first MOVE_FORWARD carrying the pull: the step is judged on what is left after the excuse.
    step = moved(pose, dx=0.29 * math.cos(pose.yaw), dy=0.29 * math.sin(pose.yaw))
    check_motion(F, pose, step, SPEC, gibson, first=True)
    with pytest.raises(EnvContractError):
        check_motion(F, pose, step, SPEC, KinematicTolerance(settle_m=0.30), first=True), "0.29 m is over the 0.25 + 0.03 reach"
    with pytest.raises(EnvContractError):
        check_motion(A.TURN_LEFT, pose, turned, SPEC, gibson, first=False), "only the first action"
    far = moved(pose, dx=0.08)
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(A.TURN_LEFT, pose, AgentPose(far.x, far.y, far.z, turned.yaw), SPEC, gibson, first=True), "0.08 - 0.05 > 0.02"
    assert KinematicTolerance().settle_xy_m == 0.0, "off by default"
    with pytest.raises(HarnessError):
        KinematicTolerance(settle_xy_m=-0.1)
    assert KinematicTolerance(settle_xy_m=0.0).settle_xy_m == 0.0, "zero is allowed, like settle_m"




def _off_heading(pose, across_m, course_deg, dz=0.0):
    """``pose`` carried ``across_m`` over the floor at ``course_deg`` from its heading (and ``dz`` up), heading kept."""
    return moved(pose, dx=across_m * math.cos(pose.yaw + math.radians(course_deg)),
                 dy=across_m * math.sin(pose.yaw + math.radians(course_deg)), dz=dz)


def test_the_step_filters_placement_is_excused_on_any_body_action_only_up_to_reproject_m():
    """The 2026-10-22 HM3D v1 sweep (1,871 episodes) lost two 99-episode scenes to habitat-sim's navmesh step
    filter carrying an off-mesh agent back onto the mesh: 00878-XB4GS9ShBRE/000001 action 156, a TURN_LEFT that
    "moved" 0.071 m at -52 degrees from its heading; 00873-bxsVRursffK/000068 action 315, a MOVE_FORWARD that
    went 0.094 m at -53.4 degrees and 0.057 m up. With ``reproject_m`` that much placement is taken off any turn
    or step -- across the heading, behind the start, past the reach -- and the rest is judged as before; a LOOK or
    STOP that moves, a larger pull, and a sliding step are still refused."""
    pose = AgentPose(3.0, -1.0, 0.5, math.radians(110.0))
    pulled_turn = moved(_off_heading(pose, 0.071, -52.0), dyaw_deg=30.0)
    pulled_step = _off_heading(pose, 0.094, -53.4, dz=0.057)
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(A.TURN_LEFT, pose, pulled_turn, SPEC, KinematicTolerance(climb_m=0.60))
    with pytest.raises(EnvContractError, match="along its heading"):
        check_motion(F, pose, pulled_step, SPEC, KinematicTolerance(climb_m=0.60))
    hm3d = KinematicTolerance(climb_m=0.60, reproject_m=0.15)
    check_motion(A.TURN_LEFT, pose, pulled_turn, SPEC, hm3d)
    check_motion(F, pose, pulled_step, SPEC, hm3d)
    check_motion(A.TURN_RIGHT, pose, moved(_off_heading(pose, 0.12, 140.0, dz=0.1), dyaw_deg=-30.0), SPEC, hm3d), \
        "a turn's placement is three-dimensional: 0.12 m across and 0.10 m up is 0.156 m, 0.006 m over the slack, under 0.02"
    # Mid-episode, not only the first action; and the heading still has to be kept.
    check_motion(A.TURN_LEFT, pose, pulled_turn, SPEC, hm3d, first=False)
    with pytest.raises(EnvContractError, match="turn [+]30.0 deg"):
        check_motion(A.TURN_LEFT, pose, moved(_off_heading(pose, 0.071, -52.0), dyaw_deg=-30.0), SPEC, hm3d)
    # A full step with the placement across it passes; the placement cannot explain a step beyond reach + slack.
    heading = (math.cos(pose.yaw), math.sin(pose.yaw))
    step = moved(_off_heading(pose, 0.10, -90.0), dx=0.25 * heading[0], dy=0.25 * heading[1])
    check_motion(F, pose, step, SPEC, hm3d)
    with pytest.raises(EnvContractError, match="advance at most"):
        check_motion(F, pose, moved(pose, dx=0.44 * heading[0], dy=0.44 * heading[1]), SPEC, hm3d), "0.44 > 0.28 + 0.15"
    # More than the slack across the heading is still a slide, a mirrored frame, or a wrong-axis read.
    with pytest.raises(EnvContractError, match="along its heading"):
        check_motion(F, pose, _off_heading(pose, 0.25, -70.0), SPEC, hm3d), "0.235 m across: 0.085 m over the slack at 0.09 m along"
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(A.TURN_LEFT, pose, moved(_off_heading(pose, 0.18, -52.0), dyaw_deg=30.0), SPEC, hm3d), "0.18 - 0.15 > 0.02"
    # LOOKs and STOP go to the sensors, never through the step filter: no slack for them.
    with pytest.raises(EnvContractError, match="tilt in place"):
        check_motion(A.LOOK_DOWN, pose, moved(_off_heading(pose, 0.071, -52.0), dpitch_deg=30.0), SPEC, hm3d)
    with pytest.raises(EnvContractError, match="leave the pose unchanged"):
        check_motion(STOP, pose, _off_heading(pose, 0.071, -52.0), SPEC, hm3d)
    assert KinematicTolerance().reproject_m == 0.0, "off by default"
    with pytest.raises(HarnessError):
        KinematicTolerance(reproject_m=-0.1)


def _placed(pose, dx=0.0, dy=0.0, dz=0.0):
    """``pose`` as the step filter placed it: the same heading and pitch, its position carried by the given amounts."""
    return moved(pose, dx=dx, dy=dy, dz=dz)


def test_a_reported_placement_judges_a_body_action_from_where_the_step_filter_started_it():
    """habitat-sim's tryStep projects the agent onto the mesh (``placed``), walks the straight line from there to
    where the action aimed from the reported pose, and with sliding off stops at the first wall: the realised end
    lies on that segment. 00878-XB4GS9ShBRE/000095 action 187 (2026-10-27) was a 0.305 m placement onto the next
    tread and a step blocked at once; an unblocked step from an off-mesh pose still lands a stride ahead of the
    reported pose. Both pass with the placement reported and neither does on the 0.15 m slack."""
    pose = AgentPose(2.0, -1.0, 0.3, math.radians(40.0), 0.0)
    hm3d = KinematicTolerance(climb_m=0.85, reproject_m=0.15)
    placed = _placed(pose, dx=0.233 * math.cos(pose.yaw) + 0.197 * math.sin(pose.yaw),
                     dy=0.233 * math.sin(pose.yaw) - 0.197 * math.cos(pose.yaw))
    with pytest.raises(EnvContractError, match="along its heading"):
        check_motion(F, pose, placed, SPEC, hm3d)
    check_motion(F, pose, placed, SPEC, hm3d, placed=placed)                       # blocked at once: ends where placed
    aimed = moved(pose, dx=SPEC.forward_step_m * math.cos(pose.yaw), dy=SPEC.forward_step_m * math.sin(pose.yaw))
    far = _placed(pose, dx=-0.30 * math.sin(pose.yaw), dy=0.30 * math.cos(pose.yaw))   # 0.30 m off the mesh, sideways
    check_motion(F, pose, aimed, SPEC, hm3d, placed=far)                            # unblocked: lands a stride ahead
    half = AgentPose((far.x + aimed.x) / 2, (far.y + aimed.y) / 2, pose.z, pose.yaw, 0.0)
    check_motion(F, pose, half, SPEC, hm3d, placed=far)                             # blocked half-way along the line
    check_motion(F, pose, pose, SPEC, hm3d, placed=far)                             # refused outright: moved nothing
    check_motion(A.TURN_LEFT, pose, moved(pose, dyaw_deg=30.0), SPEC, hm3d, placed=far)   # turned where it stood
    with pytest.raises(EnvContractError, match="keep the heading"):
        check_motion(F, pose, moved(pose, dyaw_deg=5.0), SPEC, hm3d, placed=far)
    # Off the segment is a slide, past its end is another step length, backwards is a mirrored frame: all refused.
    with pytest.raises(EnvContractError, match="toward where it aimed"):
        check_motion(F, pose, moved(half, dx=0.12 * math.sin(pose.yaw), dy=-0.12 * math.cos(pose.yaw)), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="advance at most"):
        check_motion(F, pose, moved(aimed, dx=0.2 * math.cos(pose.yaw), dy=0.2 * math.sin(pose.yaw)), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="toward where it aimed"):
        check_motion(F, pose, moved(pose, dx=-0.25 * math.cos(pose.yaw), dy=-0.25 * math.sin(pose.yaw)), SPEC, hm3d,
                     placed=_placed(pose, dx=0.02 * math.sin(pose.yaw), dy=-0.02 * math.cos(pose.yaw)))
    # The height is judged from the mesh point the filter started at, so a stale height corrects itself.
    check_motion(F, pose, moved(aimed, dz=-0.35), SPEC, hm3d, placed=_placed(pose, dz=-0.35))
    with pytest.raises(EnvContractError, match="height by at most 0.850 m from the mesh"):
        check_motion(F, pose, moved(aimed, dz=-1.3), SPEC, hm3d, placed=_placed(pose, dz=-0.35))
    # A turn heads back to the reported pose: ending where placed, at the pose, or between is the filter's; beyond is not.
    turned = moved(pose, dyaw_deg=30.0)
    check_motion(A.TURN_LEFT, pose, moved(far, dyaw_deg=30.0), SPEC, hm3d, placed=far)
    check_motion(A.TURN_LEFT, pose, turned, SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="turn in place"):
        check_motion(A.TURN_LEFT, pose, moved(far, dyaw_deg=30.0), SPEC, hm3d)
    with pytest.raises(EnvContractError, match="advance at most"):   # past the pose, on its far side
        check_motion(A.TURN_LEFT, pose, moved(turned, dx=0.30 * math.sin(pose.yaw), dy=-0.30 * math.cos(pose.yaw)), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match=r"turn \+30.0 deg"):
        check_motion(A.TURN_LEFT, pose, moved(far, dyaw_deg=-30.0), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="camera pitch"):
        check_motion(A.TURN_LEFT, pose, moved(far, dyaw_deg=30.0, dpitch_deg=5.0), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="keep the heading"):
        check_motion(F, pose, moved(aimed, dyaw_deg=5.0), SPEC, hm3d, placed=far)
    # LOOKs and STOP go to the sensors: the placement is not theirs, and a body that moves to it is refused.
    check_motion(A.LOOK_DOWN, pose, moved(pose, dpitch_deg=30.0), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="tilt in place"):
        check_motion(A.LOOK_DOWN, pose, moved(far, dpitch_deg=30.0), SPEC, hm3d, placed=far)
    with pytest.raises(EnvContractError, match="leave the pose unchanged"):
        check_motion(STOP, pose, far, SPEC, hm3d, placed=far)
    # A placement at the pose, or none, is the ordinary check, slack and all.
    check_motion(F, pose, _off_heading(pose, 0.094, -53.4, dz=0.057), SPEC, hm3d, placed=pose)
    check_motion(F, pose, _off_heading(pose, 0.094, -53.4, dz=0.057), SPEC, hm3d, placed=None)
    with pytest.raises(TypeError, match="placed"):
        check_motion(F, pose, aimed, SPEC, hm3d, placed=(0.0, 0.0, 0.0))


class PlacingEnv(MotionEnv):
    """The corridor whose simulator keeps the agent ``offset`` off its mesh sideways: every MOVE_FORWARD starts
    from the projection and ends a stride ahead of the reported pose (unblocked), every turn carries the agent back
    onto the mesh -- habitat-sim's step filter, in miniature -- and ``placed_pose`` reports the projection."""

    def __init__(self, offset, report=True, **options):
        super().__init__(executes(SPEC), **options)
        self.offset = offset
        self.report = report

    def placed_pose(self):
        if not self.report:
            return None
        return moved(self.pose, dx=-self.offset * math.sin(self.pose.yaw), dy=self.offset * math.cos(self.pose.yaw))

    def step(self, action):
        if action in (A.TURN_LEFT, A.TURN_RIGHT):
            before = self.pose
            self.pose = moved(apply_action(before, action, SPEC), dx=-self.offset * math.sin(before.yaw),
                              dy=self.offset * math.cos(before.yaw))
            self.path += math.dist(before.position(), self.pose.position())
            self.steps += 1
            self.stopped = self.stopped or action == STOP
            return self.observation(self.steps, "step")
        return super().step(action)


def test_the_runner_hands_the_environments_placement_to_the_check():
    """A turn that carries the agent 0.30 m back onto the mesh ends the run when the environment does not report
    where the step filter started -- and is the filter's own doing when it does."""
    with pytest.raises(EnvContractError, match="turn in place"):
        one_episode(PlacingEnv(0.30, report=False), ScriptedAgent((F, A.TURN_LEFT, F, STOP)), kinematics=TOLERANCE)
    record = one_episode(PlacingEnv(0.30, report=True), ScriptedAgent((F, A.TURN_LEFT, F, STOP)), kinematics=TOLERANCE)
    assert record.steps == 4 and record.agent_error is None
