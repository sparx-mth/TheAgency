"""The standard MP3D agent gets no world frame, including through its converter."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sparx_agency.core.planning.objnav.camera_geometry import intrinsics_from_hfov, world_T_camera_optical
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.core.planning.objnav.relative_height import RelativeHeight
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.standard_inputs import STANDARD_INPUTS, StandardInputAgent
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import run as mp3d_run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d_evaluation import (
    PROTOCOL, StraightToTheGoal, prepared, run_config)


def test_gps_compass_only_and_commanded_tilt(tmp_path):
    dataset, env = prepared(tmp_path, per_scene=1)
    ep, obs = env.reset(dataset.episode_ids()[0])
    seen = []
    delegate = SimpleNamespace(name="spy", reset=seen.append,
                               act=lambda o: seen.append(o) or SimpleNamespace(action=DiscreteAction.LOOK_DOWN))
    adapter = StandardInputAgent(delegate)
    for k in (1, 7):
        adapter.reset(replace(ep, metadata={"extra": k}))
        assert seen[-1].metadata == {} and seen[-1].episode_id != ep.episode_id and seen[-1].scene_id != ep.scene_id
        yaw = math.pi / 2 if k == 1 else -math.pi / 2
        adapter.act(replace(obs, pose=AgentPose(k, -k, 3 * k, yaw, 1.0)))
        assert seen[-1].pose == AgentPose(0, 0)
        adapter.act(replace(obs, step=1, pose=AgentPose(k, -k + math.sin(yaw), -4 * k, yaw + 0.5, -1.0)))
        local = seen[-1]
        assert math.isclose(local.pose.x, 1) and abs(local.pose.y) < 1e-12 and local.pose.z == 0
        assert math.isclose(local.pose.yaw, 0.5) and math.isclose(local.pose.camera_pitch, math.pi / 6)
        assert local.rgb is obs.rgb and local.depth_m is obs.depth_m and local.target_category == ep.target_category
    env.close()


def test_strict_agent_scores_the_real_episode_and_path(tmp_path):
    class LocalStraight(StraightToTheGoal):
        def plan(self, observation):
            if observation.step == 0:
                assert observation.pose == AgentPose(0, 0)
            return NavigationCommand.follow([(observation.pose.x, observation.pose.y), (3., 0.)], stop_on_arrival=True)

    dataset, env = prepared(tmp_path, per_scene=1)
    output = tmp_path / "run"
    cfg = dict(run_config(dataset), policy_inputs=STANDARD_INPUTS)
    run_evaluation(env, LocalStraight(), mp3d_label_mapper(), output=output, config=cfg,
                   settings=PROTOCOL.evaluation_settings())
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["episode_id"] == dataset.episode_ids()[0]
    assert row["success"] and math.isclose(row["spl"], 1.)
    assert math.isclose(row["path_length_m"], 3.)


def test_strict_preflight_pins_the_estimator_and_disables_raw_depth_hole_fill(tmp_path, monkeypatch):
    dataset, rig = prepared(tmp_path, per_scene=1)
    rig.close()
    monkeypatch.setattr(mp3d_run, "runtime_record", lambda issues: {})
    for key in ("LLM_ANSWERS_RECORD", "LLM_ANSWERS_REPLAY", "OBJNAV_FRAME_LOG_DIR"):
        monkeypatch.delenv(key, raising=False)
    args = mp3d_run.parser().parse_args(["--standard-inputs", "--agent", "stop", "--limit", "1",
                                         "--episodes-dir", str(dataset.episodes_dir), "--scenes-dir", str(dataset.scenes_dir),
                                         "--skip-scene-hashes", "--allow-shared-gpu"])
    env, _, _, config, issues = mp3d_run.prepare(args)
    try:
        assert not issues
        assert config["policy_inputs"] == STANDARD_INPUTS and config["policy_height_estimator"] == RelativeHeight.NAME
        assert config["protocol"]["depth_hole_fill"] is False
    finally:
        env.close()


@pytest.mark.parametrize("extra,env_var,reason", [
    (["--detector-backend", "yolo_world"], None, "YOLO"),
    (["--detector-backend", "hybrid"], None, "YOLO"),
    (["--agent", "falcon-yolo"], None, "YOLO"),
    (["--perception", "ground-truth"], None, "ground-truth"),
    (["--record-log-only"], None, "recordings"),
    ([], "LLM_ANSWERS_REPLAY", "replay"),
    ([], "LLM_ANSWERS_RECORD", "recordings"),
    ([], "OBJNAV_FRAME_LOG_DIR", "recordings"),
])
def test_incompatible_strict_inputs_fail_before_services_or_scene_load(monkeypatch, extra, env_var, reason):
    args = mp3d_run.parser().parse_args(["--standard-inputs", "--detector-backend", "grounding_dino", *extra])
    if env_var:
        monkeypatch.setenv(env_var, "forbidden")
    monkeypatch.setattr(mp3d_run, "_method", lambda *args: pytest.fail("strict preflight contacted a service"))
    env, policy, _, _, issues = mp3d_run.prepare(args)
    assert env is policy is None and any(reason in issue for issue in issues)


@pytest.mark.parametrize("backend", ["yolo_world", "hybrid", "ground_truth"])
def test_prepared_strict_run_refuses_incompatible_detector_identity_before_reset(tmp_path, backend):
    dataset, env = prepared(tmp_path, per_scene=1)
    config = dict(run_config(dataset), policy_inputs=STANDARD_INPUTS)
    config["method"]["detector"] = {"metadata": {"backend": backend}}
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="require a verified"):
        run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output, config=config)
    assert not output.exists() and env._simulator.pose is None


@pytest.mark.parametrize("rise", [-0.18, 0.18])
def test_only_observed_rgbd_can_supply_relative_height(tmp_path, rise):
    dataset, env = prepared(tmp_path, per_scene=1)
    ep, obs = env.reset(dataset.episode_ids()[0])
    camera = CameraSpec(intrinsics_from_hfov(160, 120, 75), .88, .5, 5.)
    k = camera.intrinsics
    texture = np.random.RandomState(17).randint(0, 256, (512, 512), dtype=np.uint8)
    v, u = np.indices((k.height, k.width))
    rays = np.stack(((u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u)), axis=-1)

    def wall(x, z):
        t = world_T_camera_optical(AgentPose(x, 0, z), camera)
        d = 3. - x
        xyz = (rays * d) @ t[:3, :3].T + t[:3, 3]
        uv = (xyz[:, :, 1:] * 40 + 256).astype(np.float32)
        gray = cv2.remap(texture, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), np.full(gray.shape, d, np.float32)

    seen = []
    delegate = SimpleNamespace(name="probe", reset=lambda ep: None, episode_info=lambda: {},
                               act=lambda o: seen.append(o.pose.z) or SimpleNamespace(action=DiscreteAction.MOVE_FORWARD))
    agent = StandardInputAgent(delegate)
    agent.reset(replace(ep, camera=camera))
    for step, true_z in enumerate((0., rise, rise, rise, rise, rise)):
        x = .25 * step
        rgb, depth = wall(x, true_z)
        if step == 3:
            depth.fill(np.inf)
        if step >= 4:
            rgb.fill(100)  # untextured flat wall: the last pair cannot resolve any vertical motion
        agent.act(replace(obs, rgb=rgb, depth_m=depth, camera=camera, step=step,
                          pose=AgentPose(x, 0, 10. - step, 0)))  # unrelated evaluator altitude
    assert seen[0] == 0 and seen[1:] == pytest.approx([rise] * 5, abs=.015)
    assert agent.episode_info()["rgbd_height"]["unmeasured_strides"] == 3
    agent.reset(ep)
    assert agent.relative_height.height == 0 and agent.relative_height.previous is None
    env.close()
