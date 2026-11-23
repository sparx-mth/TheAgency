"""The frame log in a run: the policy factory hands it over, a whole run fills it, and its failure is never the method's.

``core/planning/object_search/frame_log`` proves the log on its own, and the
method package that the policy feeds it. What only this tree can prove: the
one policy builder turns ``OBJNAV_FRAME_LOG_DIR`` into the policy's log (and
refuses it for an agent that keeps none), the real runner / headless agent /
policy loop logs every frame a decision was taken on under the pose the
environment reported for it, and the harness stops on the log's error where
it would record an ordinary agent error as a failed episode.
"""
from __future__ import annotations

import json

import pytest

from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.frame_log.disk_store import (
    DiskFrameStore, episode_directory_name, read_frame_log)
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import FRAME_LOG_DIR_ENV, EpisodeFrameLog
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameLogError
from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy, RPTSettings
from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import ON_AGENT_ERROR_RECORD, decide
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import FakeObjNavEnv
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import fake_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.rendering import RGB_GREY
from sparx_agency.tasks.planning.objnav_benchmark.smoke import build_demo_episodes, build_demo_world
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import build_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_ground_truth_perception import (
    MAPPER, arguments, source_with)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM

#: Actions per episode of the run below: the warm-up look-around (ten turns) and thirty actions of walking after it.
STEPS = 40


class SeesNothing:
    """A detector that reports no object: the search explores for its whole budget."""

    def detect(self, rgb):
        return []


class Remembering(ObjNavEnv):
    """A pass-through environment that keeps every observation it hands out, per episode."""

    def __init__(self, env):
        self.env, self.name, self.handed_out, self.resets = env, env.name, {}, []

    def episode_ids(self):
        return self.env.episode_ids()

    def reset(self, episode_id):
        episode, observation = self.env.reset(episode_id)
        self.resets.append(episode_id)
        self.handed_out[episode_id] = [observation]
        return episode, observation

    def step(self, action):
        observation = self.env.step(action)
        self.handed_out[self.resets[-1]].append(observation)
        return observation

    @property
    def episode_over(self):
        return self.env.episode_over

    def measure(self):
        return self.env.measure()


def building(episodes=2):
    """The harness's demo building behind a 160 x 120 camera, ``STEPS`` actions an episode."""
    world = build_demo_world()
    camera = CameraSpec(intrinsics_from_hfov(160, 120, 79.0), 0.88, 0.1, 5.0)
    return Remembering(FakeObjNavEnv(world, build_demo_episodes(world, episodes, 0), camera=camera, max_steps=STEPS))


def searching(frame_log):
    return RPTSearchPolicy(SeesNothing(), FakeLLM(), RPTSettings(map_size_m=20.0, body_height_m=0.88, body_radius_m=0.18),
                           frame_log=frame_log)


# -- a whole run ------------------------------------------------------------------------------------------------------
def test_a_run_logs_every_frame_a_decision_was_taken_on_under_the_pose_the_environment_reported(tmp_path):
    env, frames = building(), tmp_path / "frames"
    summary = run_evaluation(env, searching(EpisodeFrameLog(DiskFrameStore(frames))), fake_label_mapper(),
                             output=tmp_path / "run", config={"purpose": "frame log"})
    rows = [json.loads(line) for line in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()]
    assert summary.overall.n_episodes == 2 and [row["episode_id"] for row in rows] == ["demo-000", "demo-001"]
    assert {path.name for path in frames.iterdir()} == {episode_directory_name(row["episode_id"]) for row in rows}
    for row in rows:
        assert row["agent_error"] is None and row["steps"] == STEPS
        directory = frames / episode_directory_name(row["episode_id"])
        header, records = read_frame_log(directory)
        handed_out = env.handed_out[row["episode_id"]]
        assert (header["episode_id"], header["target_category"]) == (row["episode_id"], row["target_category"])
        # One frame per action taken: steps 0 .. STEPS-1, none missing. The frame the last action returned is the
        # one observation no decision is taken on, and the only one not logged.
        assert list(records) == ["step_%06d.jpg" % step for step in range(STEPS)]
        assert len(handed_out) == STEPS + 1 and sorted(p.name for p in directory.glob("step_*")) == list(records)
        for observation, record in zip(handed_out, records.values()):
            assert record.step == observation.step and record.pose == observation.pose
        positions = {(record.pose.x, record.pose.y) for record in records.values()}
        headings = {record.pose.yaw for record in records.values()}
        assert len(positions) >= 5 and len(headings) >= 5, "the agent turned and walked: the poses are not one pose"
        for name in (list(records)[0], list(records)[-1]):
            image = FrameCodec().decode((directory / name).read_bytes())
            assert image.shape == (120, 160, 3) and abs(float(image.mean()) - RGB_GREY) < 1, "the fake's grey frame"


def test_a_resumed_run_leaves_the_frames_of_finished_episodes_as_they_were(tmp_path):
    frames, config = tmp_path / "frames", {"purpose": "frame log"}
    run_evaluation(building(), searching(EpisodeFrameLog(DiskFrameStore(frames))), fake_label_mapper(),
                   output=tmp_path / "run", config=config)
    before = {path: path.stat().st_mtime_ns for path in sorted(frames.rglob("*")) if path.is_file()}
    assert len(before) == 2 * (STEPS + 2), "per episode: its frames, the index and the header"
    env = building()
    resumed = run_evaluation(env, searching(EpisodeFrameLog(DiskFrameStore(frames))), fake_label_mapper(),
                             output=tmp_path / "run", config=config, resume=True)
    assert resumed.overall.n_episodes == 2 and env.resets == [], "nothing is run again"
    assert {path: path.stat().st_mtime_ns for path in sorted(frames.rglob("*")) if path.is_file()} == before


# -- the policy factory -----------------------------------------------------------------------------------------------
@pytest.fixture
def stand_in_llm(monkeypatch):
    """The factory's verified LLM client replaced by the neutral fake: no service is asked."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.services import llm

    class Verified(FakeLLM):
        def __init__(self, client):
            super().__init__()

        def health(self):
            return {"model": "fake", "digest": "0"}

    monkeypatch.setattr(llm, "VerifiedLLMClient", Verified)


def test_the_factory_hands_the_room_search_policy_the_frame_log_the_variable_names(monkeypatch, tmp_path, stand_in_llm):
    monkeypatch.setenv(FRAME_LOG_DIR_ENV, str(tmp_path / "frames"))
    policy, method = build_policy(arguments(agent="rpt"), MAPPER, ground_truth=source_with({2: "chair"}))
    assert isinstance(policy.frame_log, EpisodeFrameLog) and isinstance(policy.frame_log.store, DiskFrameStore)
    assert policy.frame_log.store.root == tmp_path / "frames"
    assert method["frame_log"] == {"schema": "objnav_frame_log/1", "image": {"format": "jpg", "jpeg_quality": 95},
                                   "store": {"kind": "disk", "root": str(tmp_path / "frames")}}
    json.dumps(method)
    assert policy.configuration()["frame_log"] == method["frame_log"]
    assert not (tmp_path / "frames").exists(), "nothing is written before an episode begins"


def test_without_a_disk_override_the_factory_records_the_live_ram_log(monkeypatch, stand_in_llm):
    monkeypatch.delenv(FRAME_LOG_DIR_ENV, raising=False)
    policy, method = build_policy(arguments(agent="rpt"), MAPPER, ground_truth=source_with({2: "chair"}))
    assert policy.frame_log.describe() == method["frame_log"]
    assert method["frame_log"]["store"] == {"kind": "memory"}
    monkeypatch.setenv(FRAME_LOG_DIR_ENV, "  ")
    assert build_policy(arguments(agent="rpt"), MAPPER, ground_truth=source_with({2: "chair"}))[1]["frame_log"] == method["frame_log"]


def test_factory_runs_the_image_client_and_real_planners_through_warmup(monkeypatch):
    import base64
    import math

    import requests
    from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_runtime_boundaries import Response
    from dataclasses import replace

    model = "fixture-vl"
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv(FRAME_LOG_DIR_ENV, raising=False)
    images = []

    def get(session, url, **kwargs):
        assert url.endswith("/api/tags")
        return Response({"models": [{"name": model, "digest": "fixture"}]})

    def post(session, url, data=None, **kwargs):
        if url.endswith("/api/show"):
            return Response({"capabilities": ["vision"]})
        assert url.endswith("/api/chat")
        request = json.loads(data)
        assert request["model"] == model
        encoded, = request["messages"][1]["images"]
        images.append(FrameCodec().decode(base64.b64decode(encoded)))
        return Response({"message": {"content": '{"probability":0.7}'}})

    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(requests.Session, "post", post)
    p, _ = build_policy(arguments(agent="rpt"), MAPPER, defaults={"map_size_m": 20.0}, ground_truth=source_with({2: "chair"}))
    assert len(images) == 1, "one multimodal preflight, no room-model probes"
    p.detector = SeesNothing()  # only perception is stubbed; no simulator annotations in these synthetic observations
    ep, _ = building(episodes=1).reset("demo-000")
    p.reset(ep, MAPPER.target_labels("chair"))
    for step in range(p.warmup_turns + 1):
        obs = replace(observation(ep, step, depth=2.0), pose=AgentPose(0, 0, 0, math.radians(30 * step)))
        command = p.plan(obs)
        if step < p.warmup_turns:
            assert command.info["kind"] == "warmup"
            p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.frontiers.calls == len(images) - 1 > 0 and p.solver.calls > 0
    assert command.waypoints and command.info["probability"] == 0.7


@pytest.mark.parametrize("agent", ["falcon-yolo", "stop"])
def test_an_agent_that_keeps_no_frame_log_refuses_the_variable_instead_of_ignoring_it(agent, monkeypatch, tmp_path):
    monkeypatch.setenv(FRAME_LOG_DIR_ENV, str(tmp_path / "frames"))
    with pytest.raises(ValueError, match="only rpt and rpt_room_llm keep a frame log; this run asked for --agent %s" % agent):
        build_policy(arguments(agent=agent), MAPPER, ground_truth=source_with({2: "chair"}))
    monkeypatch.delenv(FRAME_LOG_DIR_ENV)
    policy, method = build_policy(arguments(agent=agent), MAPPER, ground_truth=source_with({2: "chair"}))
    assert "frame_log" not in method and getattr(policy, "frame_log", None) is None
    assert not (tmp_path / "frames").exists()


# -- whose failure it is ----------------------------------------------------------------------------------------------
class Raising:
    def __init__(self, error):
        self.error = error

    def act(self, observation):
        raise self.error


def test_the_harness_stops_on_a_frame_log_failure_where_it_records_an_ordinary_agent_error():
    """Under ``on_agent_error="record"`` -- every benchmark entrypoint's mode -- a method's bug is a failed episode.

    A frame that could not be kept is not the method's bug, and must not become its failed episode: the log's
    error is of the family the runner re-raises in every mode.
    """
    spec = DiscreteActionSpec()
    action, described = decide(Raising(ValueError("the method's own bug")), None, spec, ON_AGENT_ERROR_RECORD)
    assert action == DiscreteAction.STOP and described == "ValueError: the method's own bug"
    with pytest.raises(FrameLogError, match="No space left on device"):
        decide(Raising(FrameLogError("the frame log cannot keep frame step_000007.jpg: OSError: No space left on device")),
               None, spec, ON_AGENT_ERROR_RECORD)


class Full(DiskFrameStore):
    """A disk that fills up at one step."""

    def put(self, record, image):
        if record.step == 12:
            raise OSError(28, "No space left on device")
        super().put(record, image)


def test_a_full_disk_stops_the_run_and_scores_nothing(tmp_path):
    env = building(episodes=1)
    with pytest.raises(FrameLogError, match="cannot keep frame step_000012.jpg of episode 'demo-000': OSError"):
        run_evaluation(env, searching(EpisodeFrameLog(Full(tmp_path / "frames"))), fake_label_mapper(),
                       output=tmp_path / "run", config={"purpose": "frame log"})
    assert (tmp_path / "run" / "episodes.jsonl").read_text() == "", "no row: the episode was not scored as a failure"
    header, records = read_frame_log(tmp_path / "frames" / episode_directory_name("demo-000"))
    assert list(records) == ["step_%06d.jpg" % step for step in range(12)], "what was kept before it is whole"
    assert [record.pose for record in records.values()] == [o.pose for o in env.handed_out["demo-000"][:12]]
