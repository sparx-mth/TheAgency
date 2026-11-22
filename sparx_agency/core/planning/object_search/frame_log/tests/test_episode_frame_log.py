"""The episode log: every frame under its own step and exact pose, the same lookup whichever store keeps the images."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.codec import PNG, FrameCodec
from sparx_agency.core.planning.object_search.frame_log.disk_store import (
    HEADER_NAME, DiskFrameStore, episode_directory_name, read_frame_log)
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import FRAME_LOG_DIR_ENV, EpisodeFrameLog
from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    POSE_CONVENTION, SCHEMA, FrameLogError, FrameRecord)
from sparx_agency.core.planning.object_search.frame_log.memory_store import MemoryFrameStore
from sparx_agency.core.planning.object_search.frame_log.tests.rig import episode, frame, pattern, pose, red_level
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError


@pytest.fixture(params=["disk", "memory"])
def store(request, tmp_path):
    return DiskFrameStore(tmp_path / "frames") if request.param == "disk" else MemoryFrameStore()


def logged(store, steps, codec=None, ep=None):
    """A log with the frames of ``steps`` of one episode in it."""
    ep = ep or episode()
    log = EpisodeFrameLog(store, codec)
    log.begin(ep)
    for step in steps:
        log.add(frame(ep, step))
    return log


# -- what the log keeps -----------------------------------------------------------------------------------------------
def test_every_frame_is_kept_under_its_own_step_and_the_exact_pose_it_was_captured_from(store):
    ep = episode()
    log = EpisodeFrameLog(store, FrameCodec(PNG))
    log.begin(ep)
    observations = [frame(ep, step) for step in range(6)]
    returned = [log.add(observation) for observation in observations]
    assert log.episode_id == ep.episode_id and len(log) == 6
    assert list(log.records) == ["step_%06d.png" % step for step in range(6)], "the lookup is in capture order"
    for step, (observation, record) in enumerate(zip(observations, returned)):
        assert log.records[record.name] is record and record == FrameRecord("step_%06d.png" % step, step, pose(step))
        assert record.pose is observation.pose, "the pose is the observation's own, not a copy that could drift"
        assert np.array_equal(log.rgb(record.name), observation.rgb), "the image under a name is that step's frame"


def test_the_lookup_and_the_images_are_the_same_whichever_store_keeps_them(tmp_path):
    """Swapping disk for RAM changes where the bytes are and nothing a reader of the log can see."""
    on_disk = logged(DiskFrameStore(tmp_path), range(5))
    in_ram = logged(MemoryFrameStore(), range(5))
    assert dict(on_disk.records) == dict(in_ram.records) and list(on_disk.records) == list(in_ram.records)
    for name in on_disk.records:
        assert on_disk.store.get(name) == in_ram.store.get(name), "byte-identical encoded images"
        assert np.array_equal(on_disk.rgb(name), in_ram.rgb(name))
    assert on_disk.describe()["store"]["kind"] == "disk" and in_ram.describe()["store"]["kind"] == "memory"


def test_a_jpeg_log_returns_each_steps_frame_within_compression(store):
    log = logged(store, range(4))
    assert list(log.records) == ["step_%06d.jpg" % step for step in range(4)]
    for step, name in enumerate(log.records):
        back = log.rgb(name)
        assert back.shape == pattern(step).shape and back.dtype == np.uint8
        assert abs(float(back[..., 0].mean()) - red_level(step)) < 3, "red is this step's red: not a neighbour's, not blue"
        assert abs(float(back[..., 2].mean()) - (255 - red_level(step))) < 3


def test_the_pose_read_back_from_disk_is_the_pose_that_was_captured(tmp_path):
    log = logged(DiskFrameStore(tmp_path), range(7))
    header, records = read_frame_log(log.store.directory)
    assert records == dict(log.records) and list(records) == list(log.records)
    for step, record in enumerate(records.values()):
        captured = pose(step)
        assert (record.pose.x, record.pose.y, record.pose.z) == (captured.x, captured.y, captured.z)
        assert (record.pose.yaw, record.pose.camera_pitch) == (captured.yaw, captured.camera_pitch)
    for name in records:
        assert FrameCodec().decode((log.store.directory / name).read_bytes()).shape == pattern(0).shape


def test_a_logged_frame_does_not_change_when_the_environment_reuses_its_buffer(store):
    """Observations share their arrays; an adapter that renders into one buffer must not rewrite the log."""
    ep = episode()
    log = EpisodeFrameLog(store, FrameCodec(PNG))
    log.begin(ep)
    buffer = pattern(0)
    log.add(frame(ep, 0, rgb=buffer))
    buffer[...] = pattern(1)                                       # the next render, into the same array
    log.add(frame(ep, 1, rgb=buffer))
    assert np.array_equal(log.rgb("step_000000.png"), pattern(0)) and np.array_equal(log.rgb("step_000001.png"), pattern(1))
    assert np.array_equal(buffer, pattern(1)), "and the log wrote nothing into the frame it was shown"


# -- the order of things ----------------------------------------------------------------------------------------------
def test_a_step_is_logged_once_and_in_increasing_order(store):
    ep = episode()
    log = logged(store, (0, 1, 5), ep=ep)                          # a gap is a caller's choice; a repeat is not
    assert [record.step for record in log.records.values()] == [0, 1, 5]
    with pytest.raises(FrameLogError, match="step 5 was offered after step 5"):
        log.add(frame(ep, 5))
    with pytest.raises(FrameLogError, match="step 3 was offered after step 5"):
        log.add(frame(ep, 3))
    assert len(log) == 3, "a refused frame leaves the log as it was"
    assert log.add(frame(ep, 6)).name == "step_000006.jpg"


def test_nothing_is_logged_or_served_before_an_episode_begins(store):
    ep = episode()
    log = EpisodeFrameLog(store)
    assert log.episode_id is None and len(log) == 0 and dict(log.records) == {}
    with pytest.raises(FrameLogError, match=r"add\(\) before begin\(\)"):
        log.add(frame(ep, 0))
    with pytest.raises(FrameLogError, match="has no frame named 'step_000000.jpg'"):
        log.rgb("step_000000.jpg")


def test_the_log_takes_the_contracts_episode_and_observation_and_nothing_else(store):
    ep = episode()
    log = EpisodeFrameLog(store)
    with pytest.raises(FrameLogError, match="ObjNavEpisode"):
        log.begin(ep.episode_id)
    assert log.episode_id is None
    log.begin(ep)
    with pytest.raises(FrameLogError, match="ObjNavObservation"):
        log.add(pattern(0))
    for name in ("step_000009.jpg", "../step_000000.jpg", None, 3):
        with pytest.raises(FrameLogError, match="has no frame named"):
            log.rgb(name)


def test_beginning_the_next_episode_empties_the_lookup_and_keeps_the_episodes_apart(tmp_path):
    first, second = episode("scene-a/000003"), episode("scene-a/000004", target="bed")
    log = logged(DiskFrameStore(tmp_path), range(4), FrameCodec(PNG), ep=first)
    earlier = log.records
    log.begin(second)
    assert log.episode_id == "scene-a/000004" and len(log) == 0 and dict(log.records) == {}
    assert list(earlier) == ["step_%06d.png" % step for step in range(4)], "a view taken before shows the episode before"
    with pytest.raises(FrameLogError, match="has no frame named"):
        log.rgb("step_000000.png")
    log.add(frame(second, 0, rgb=pattern(9)))
    assert np.array_equal(log.rgb("step_000000.png"), pattern(9)), "step 0 of this episode, not of the last"
    folders = {path.name for path in tmp_path.iterdir()}
    assert folders == {episode_directory_name("scene-a/000003"), episode_directory_name("scene-a/000004")}
    assert len(read_frame_log(tmp_path / episode_directory_name("scene-a/000003"))[1]) == 4
    assert len(read_frame_log(tmp_path / episode_directory_name("scene-a/000004"))[1]) == 1


def test_the_lookup_cannot_be_written_through(store):
    log = logged(store, range(2))
    with pytest.raises(TypeError):
        log.records["step_000009.jpg"] = log.records["step_000000.jpg"]
    with pytest.raises(TypeError):
        del log.records["step_000000.jpg"]
    assert len(log) == 2


# -- how it fails -----------------------------------------------------------------------------------------------------
class FullDisk(MemoryFrameStore):
    """A store that refuses the frame of one step the way a full disk does."""

    def __init__(self, fails_at):
        super().__init__()
        self.fails_at = fails_at

    def put(self, record, image):
        if record.step == self.fails_at:
            raise OSError(28, "No space left on device")
        super().put(record, image)


def test_a_frame_that_cannot_be_stored_is_the_logs_own_error_and_leaves_no_record():
    """Raised as infrastructure: a benchmark run stops instead of scoring the episode as the method's failure."""
    ep = episode()
    log = EpisodeFrameLog(FullDisk(fails_at=2))
    log.begin(ep)
    log.add(frame(ep, 0))
    log.add(frame(ep, 1))
    with pytest.raises(FrameLogError, match="cannot keep frame step_000002.jpg of episode 'scene-a/000003': OSError") as caught:
        log.add(frame(ep, 2))
    assert isinstance(caught.value, ObjNavInternalError) and isinstance(caught.value.__cause__, OSError)
    assert list(log.records) == ["step_000000.jpg", "step_000001.jpg"], "no record without its image"


def test_a_store_that_cannot_begin_or_an_image_that_cannot_be_read_back_is_the_logs_own_error(tmp_path):
    blocked = tmp_path / "a-file-where-the-root-should-be"
    blocked.write_text("not a directory")
    log = EpisodeFrameLog(DiskFrameStore(blocked))
    with pytest.raises(FrameLogError, match="cannot begin episode 'scene-a/000003'") as caught:
        log.begin(episode())
    assert isinstance(caught.value.__cause__, OSError) and log.episode_id is None
    with pytest.raises(FrameLogError, match=r"add\(\) before begin\(\)"):
        log.add(frame(episode(), 0))
    log = logged(DiskFrameStore(tmp_path / "frames"), range(2))
    (log.store.directory / "step_000001.jpg").write_bytes(b"overwritten by something else")
    with pytest.raises(FrameLogError, match="not a decodable image"):
        log.rgb("step_000001.jpg")
    (log.store.directory / "step_000000.jpg").unlink()
    with pytest.raises(FrameLogError, match="no frame named 'step_000000.jpg'"):
        log.rgb("step_000000.jpg")


# -- what it says of itself -------------------------------------------------------------------------------------------
def test_the_header_names_the_episode_from_its_public_fields_and_never_from_its_metadata(tmp_path):
    ep = episode("scene-b/000017", target="tv_monitor", metadata={"somebody_left_this": "here"})
    log = logged(DiskFrameStore(tmp_path), range(1), ep=ep)
    header = json.loads((log.store.directory / HEADER_NAME).read_text())
    k = ep.camera.intrinsics
    assert header == {"schema": SCHEMA, "episode_id": "scene-b/000017", "scene_id": "scene-b", "benchmark": "synthetic",
                      "split": "val", "target_category": "tv_monitor", "image": {"format": "jpg", "jpeg_quality": 95},
                      "camera": {"width": 48, "height": 32, "fx": k.fx, "fy": k.fy, "cx": 23.5, "cy": 15.5,
                                 "height_m": 0.88, "min_depth_m": 0.5, "max_depth_m": 5.0},
                      "pose": POSE_CONVENTION}
    assert "somebody_left_this" not in (log.store.directory / HEADER_NAME).read_text()
    in_ram = logged(MemoryFrameStore(), range(1), FrameCodec(PNG), ep=ep)
    assert in_ram.store.header == dict(header, image={"format": "png"}), "the same header whichever store is handed it"


def test_a_camera_without_a_range_limit_still_writes_a_header_strict_json_can_hold(tmp_path):
    """``CameraSpec.max_depth_m`` may be infinite, and ``Infinity`` is not JSON: a header that cannot be read back."""
    ep = episode()
    limitless = dataclasses.replace(ep, camera=dataclasses.replace(ep.camera, max_depth_m=float("inf")))
    log = EpisodeFrameLog(DiskFrameStore(tmp_path))
    log.begin(limitless)
    text = (log.store.directory / HEADER_NAME).read_text()
    assert "Infinity" not in text and json.loads(text)["camera"]["max_depth_m"] is None
    assert read_frame_log(log.store.directory)[0]["camera"]["height_m"] == 0.88


def test_the_log_describes_itself_for_a_runs_record(tmp_path):
    log = EpisodeFrameLog(DiskFrameStore(tmp_path / "frames"))
    assert log.describe() == {"schema": "objnav_frame_log/1", "store": {"kind": "disk", "root": str(tmp_path / "frames")},
                              "image": {"format": "jpg", "jpeg_quality": 95}}
    assert EpisodeFrameLog(MemoryFrameStore(), FrameCodec(PNG)).describe() == {
        "schema": "objnav_frame_log/1", "store": {"kind": "memory"}, "image": {"format": "png"}}
    assert json.loads(json.dumps(log.describe())) == log.describe()


def test_the_environment_variable_names_a_disk_log_and_its_absence_none(tmp_path):
    assert FRAME_LOG_DIR_ENV == "OBJNAV_FRAME_LOG_DIR"
    assert EpisodeFrameLog.from_env({}) is None and EpisodeFrameLog.from_env({FRAME_LOG_DIR_ENV: "   "}) is None
    log = EpisodeFrameLog.from_env({FRAME_LOG_DIR_ENV: " %s " % (tmp_path / "frames")})
    assert isinstance(log.store, DiskFrameStore) and log.store.root == tmp_path / "frames"
    assert log.codec == FrameCodec() and not (tmp_path / "frames").exists(), "nothing is written until an episode begins"


def test_the_process_environment_is_read_when_none_is_given(tmp_path, monkeypatch):
    monkeypatch.delenv(FRAME_LOG_DIR_ENV, raising=False)
    assert EpisodeFrameLog.from_env() is None
    monkeypatch.setenv(FRAME_LOG_DIR_ENV, str(tmp_path))
    assert EpisodeFrameLog.from_env().store.root == tmp_path
