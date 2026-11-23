"""The policy feeds its frame log: every frame it decides on, once, under that frame's own pose -- or keeps none."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.codec import PNG, FrameCodec
from sparx_agency.core.planning.object_search.frame_log.disk_store import DiskFrameStore, read_frame_log
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import EpisodeFrameLog
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameLogError
from sparx_agency.core.planning.object_search.frame_log.memory_store import MemoryFrameStore
from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, FakeLLM, frame, make_policy
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def turning(step):
    """The pose of ``step`` of a look-around on the spot: a new heading every action, as the warm-up turns."""
    return AgentPose(0.0, 0.0, 0.0, math.radians(30.0) * step)


def shaded(ep, step, pose):
    """The rig's frame at ``pose``, with every pixel of the image set to a level of its step."""
    blank = frame(ep, step, depth=3.0, pose=pose)
    return ObjNavObservation(np.full_like(blank.rgb, 10 + 20 * step), blank.depth_m, pose, ep.camera,
                             blank.target_category, step)


def test_the_policy_logs_every_frame_it_decides_on_under_that_frames_own_pose():
    log = EpisodeFrameLog(MemoryFrameStore(), FrameCodec(PNG))
    policy, ep = make_policy(FakeDetector("bed"), FakeLLM(), frame_log=log)
    assert policy.frame_log is log and log.episode_id == ep.episode_id and len(log) == 0, "reset begins the episode's log"
    observations = [shaded(ep, step, turning(step)) for step in range(4)]
    for observation in observations:
        policy.plan(observation)
    assert list(log.records) == ["step_%06d.png" % step for step in range(4)]
    for observation, record in zip(observations, log.records.values()):
        assert record.step == observation.step and record.pose is observation.pose
        assert np.array_equal(log.rgb(record.name), observation.rgb), "the image is the frame that decision was taken on"


def test_a_frame_decided_twice_is_logged_once():
    """``plan`` answers a repeated step from its own memory; the log is not offered the frame again."""
    log = EpisodeFrameLog(MemoryFrameStore())
    policy, ep = make_policy(FakeDetector("bed"), frame_log=log)
    first = policy.plan(frame(ep, 0, depth=3.0))
    assert policy.plan(frame(ep, 0, depth=3.0)) is first and len(log) == 1
    policy.plan(frame(ep, 1, depth=3.0))
    assert list(log.records) == ["step_000000.jpg", "step_000001.jpg"]


def test_the_frame_is_logged_before_the_detector_is_shown_it():
    """Whatever reads the log during a decision finds the frame being decided on already in it."""
    log = EpisodeFrameLog(MemoryFrameStore())
    seen = []

    class Watching(FakeDetector):
        def detect(self, rgb):
            seen.append(len(log))
            return super().detect(rgb)

    policy, ep = make_policy(Watching("bed"), frame_log=log)
    policy.plan(frame(ep, 0, depth=3.0))
    policy.plan(frame(ep, 1, depth=3.0))
    assert seen == [1, 2]


def test_reset_starts_the_next_episodes_log_and_each_episode_keeps_its_own_folder(tmp_path):
    log = EpisodeFrameLog(DiskFrameStore(tmp_path))
    policy, ep = make_policy(FakeDetector("bed"), frame_log=log)
    for step in range(3):
        policy.plan(frame(ep, step, depth=3.0, pose=turning(step)))
    first = log.store.directory
    assert len(log) == 3 and len(list(first.glob("step_*.jpg"))) == 3
    policy.reset(ep, gibson_label_mapper().target_labels("chair"))        # the same episode, begun again
    assert len(log) == 0 and log.store.directory == first and not list(first.glob("step_*.jpg"))
    policy.warmup_actions, policy._warmup_pending = policy.settings.warmup_steps, False
    policy.plan(frame(ep, 0, depth=3.0))
    header, records = read_frame_log(first)
    assert list(records) == ["step_000000.jpg"] and records["step_000000.jpg"].pose == AgentPose(0, 0, 0, 0)
    assert (header["episode_id"], header["target_category"]) == (ep.episode_id, ep.target_category)


def test_without_an_explicit_log_the_policy_keeps_episode_frames_in_ram():
    plain, ep = make_policy(FakeDetector("bed"))
    logging, _ = make_policy(FakeDetector("bed"), frame_log=EpisodeFrameLog(MemoryFrameStore()))
    assert isinstance(plain.frame_log.store, MemoryFrameStore)
    assert plain.configuration() == logging.configuration()
    assert not plain.plan(frame(ep, 0, depth=3.0)).stop
    assert len(plain.frame_log) == 1
    plain.reset(ep, plain.target)
    assert len(plain.frame_log) == 0


class _Full(MemoryFrameStore):
    def put(self, record, image):
        raise OSError(28, "No space left on device")


def test_a_frame_that_cannot_be_kept_ends_the_run_not_the_episode():
    """An ``ObjNavInternalError``: the harness re-raises that family in every mode instead of scoring a failure."""
    policy, ep = make_policy(FakeDetector("bed"), frame_log=EpisodeFrameLog(_Full()))
    with pytest.raises(FrameLogError, match="cannot keep frame step_000000.jpg") as caught:
        policy.plan(frame(ep, 0, depth=3.0))
    assert isinstance(caught.value, ObjNavInternalError)
    assert policy.perception.counts["raw_frames"] == 0, "nothing was decided on a frame the log does not hold"
