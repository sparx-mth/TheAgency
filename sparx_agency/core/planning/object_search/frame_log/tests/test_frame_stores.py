"""The stores: one contract kept on disk and in RAM, and a disk folder that explains itself and can be read back."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.disk_store import (
    HEADER_NAME, INDEX_NAME, DiskFrameStore, episode_directory_name, read_frame_log)
from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    POSE_FIELDS, SCHEMA, FrameLogError, FrameRecord, frame_name)
from sparx_agency.core.planning.object_search.frame_log.memory_store import MemoryFrameStore
from sparx_agency.core.planning.object_search.frame_log.tests.rig import pose
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.objnav.types.pose import AgentPose

EPISODE = "scene-a/000003"
HEADER = {"schema": SCHEMA, "episode_id": EPISODE, "note": "a store keeps the header it is handed"}


def record(step, extension=".jpg"):
    return FrameRecord(frame_name(step, extension), step, pose(step))


@pytest.fixture(params=["disk", "memory"])
def store(request, tmp_path):
    return DiskFrameStore(tmp_path / "frames") if request.param == "disk" else MemoryFrameStore()


# -- the record -------------------------------------------------------------------------------------------------------
def test_a_frame_is_named_after_its_step_so_names_sort_in_capture_order():
    assert frame_name(0, ".jpg") == "step_000000.jpg" and frame_name(137, ".png") == "step_000137.png"
    names = [frame_name(step, ".jpg") for step in (0, 9, 10, 99, 100, 499)]
    assert names == sorted(names)


def test_a_record_survives_its_json_row_with_the_pose_bit_for_bit():
    assert POSE_FIELDS == ("x", "y", "z", "yaw", "camera_pitch"), "position, heading and the camera's pitch"
    for step in range(8):
        original = record(step)
        row = json.loads(json.dumps(original.to_row(), allow_nan=False))
        assert set(row) == {"name", "step", "pose"} and tuple(row["pose"]) == POSE_FIELDS
        assert FrameRecord.from_row(row) == original
        assert all(type(value) is float for value in row["pose"].values())
    narrow = FrameRecord("step_000001.jpg", 1, AgentPose(np.float32(0.1), np.float32(-2.7), np.float32(1.5),
                                                         np.float32(3.1), np.float32(0.3)))
    back = FrameRecord.from_row(json.loads(json.dumps(narrow.to_row())))
    assert back == narrow and back.pose.x == float(np.float32(0.1)) != 0.1, "a float32 pose widens exactly; nothing is rounded"


def test_a_record_is_one_file_name_a_step_and_a_pose():
    for name in ("../step_000000.jpg", "sub/step_000000.jpg", "a\\b.jpg", "..", ".", "", None, 7):
        with pytest.raises(FrameLogError, match="one file name"):
            FrameRecord(name, 0, pose(0))
    for step in (-1, 1.0, True, None, "3"):
        with pytest.raises(FrameLogError, match="non-negative integer"):
            FrameRecord("step_000000.jpg", step, pose(0))
    with pytest.raises(FrameLogError, match="AgentPose"):
        FrameRecord("step_000000.jpg", 0, (0.0, 0.0, 0.0))
    assert FrameRecord("step_000003.jpg", np.int64(3), pose(3)).to_row()["step"] == 3


@pytest.mark.parametrize("row", [
    {"name": "step_000000.jpg", "step": 0},                                                     # no pose
    {"name": "step_000000.jpg", "step": 0, "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}},  # no camera pitch
    {"name": "step_000000.jpg", "step": 0,
     "pose": {"x": float("nan"), "y": 0.0, "z": 0.0, "yaw": 0.0, "camera_pitch": 0.0}},         # not a place
    {"name": "step_000000.jpg", "step": -2, "pose": record(0).to_row()["pose"]},
    {"name": "elsewhere/step_000000.jpg", "step": 0, "pose": record(0).to_row()["pose"]},
    {"step": 0, "pose": record(0).to_row()["pose"]},
    ["step_000000.jpg", 0],
    "step_000000.jpg",
])
def test_a_row_that_is_not_a_record_is_refused(row):
    with pytest.raises(FrameLogError):
        FrameRecord.from_row(row)


def test_the_logs_error_is_infrastructure_never_a_callers_input_error():
    """The harness records an agent's ordinary errors as failed episodes and re-raises this family."""
    assert issubclass(FrameLogError, ObjNavInternalError) and not issubclass(FrameLogError, ValueError)


# -- the contract both stores keep ------------------------------------------------------------------------------------
def test_a_store_returns_the_bytes_it_was_handed_under_each_name(store):
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"zero")
    store.put(record(1), b"one")
    store.put(record(2), bytearray(b"two"))
    assert store.get("step_000000.jpg") == b"zero" and store.get("step_000001.jpg") == b"one"
    assert store.get("step_000002.jpg") == b"two" and type(store.get("step_000002.jpg")) is bytes


def test_nothing_is_kept_or_served_before_an_episode_begins(store):
    with pytest.raises(FrameLogError, match="no episode has begun"):
        store.put(record(0), b"zero")
    with pytest.raises(FrameLogError, match="no episode has begun"):
        store.get("step_000000.jpg")


def test_a_name_the_episode_does_not_hold_is_refused(store):
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"zero")
    with pytest.raises(FrameLogError, match="no frame named 'step_000009.jpg'"):
        store.get("step_000009.jpg")
    for name in ("../step_000000.jpg", "sub/step_000000.jpg", "..", "", None, 7, ["step_000000.jpg"]):
        with pytest.raises(FrameLogError, match="one file name"):
            store.get(name)


def test_beginning_the_next_episode_stops_serving_the_frames_of_the_one_before(store):
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"the first episode")
    store.begin("scene-a/000004", HEADER)
    with pytest.raises(FrameLogError, match="no frame named"):
        store.get("step_000000.jpg")
    store.put(record(0), b"the second episode")
    assert store.get("step_000000.jpg") == b"the second episode"


def test_a_store_describes_itself_in_plain_json(store):
    description = store.describe()
    assert description["kind"] in ("disk", "memory") and json.loads(json.dumps(description)) == description


def test_the_memory_store_keeps_the_header_and_nothing_on_disk(tmp_path):
    store = MemoryFrameStore()
    assert store.header is None and store.describe() == {"kind": "memory"}
    header = dict(HEADER)
    store.begin(EPISODE, header)
    header["note"] = "changed by the caller afterwards"
    assert store.header == HEADER, "a copy: the caller's dict is its own"
    store.put(record(0), b"zero")
    assert not list(tmp_path.iterdir())


# -- the folder a disk store leaves -----------------------------------------------------------------------------------
def test_the_disk_store_lays_an_episode_out_as_images_beside_their_index(tmp_path):
    store = DiskFrameStore(tmp_path / "frames")
    assert store.directory is None and not (tmp_path / "frames").exists(), "nothing is made before an episode begins"
    assert store.describe() == {"kind": "disk", "root": str(tmp_path / "frames")}
    store.begin(EPISODE, HEADER)
    directory = tmp_path / "frames" / episode_directory_name(EPISODE)
    assert store.directory == directory
    assert json.loads((directory / HEADER_NAME).read_text()) == HEADER and (directory / INDEX_NAME).read_text() == ""
    store.put(record(0), b"zero")
    store.put(record(1), b"one")
    assert sorted(p.name for p in directory.iterdir()) == ["episode.json", "frames.jsonl", "step_000000.jpg",
                                                           "step_000001.jpg"]
    assert (directory / "step_000001.jpg").read_bytes() == b"one"
    rows = [json.loads(line) for line in (directory / INDEX_NAME).read_text().splitlines()]
    assert rows == [record(0).to_row(), record(1).to_row()]
    assert rows[1] == {"name": "step_000001.jpg", "step": 1,
                       "pose": {field: getattr(pose(1), field) for field in POSE_FIELDS}}


def test_the_folder_name_is_readable_cannot_collide_and_ends_in_the_recorders_key():
    episode_id = "00877-4ok3usBNeis/000003"
    key = hashlib.sha256(episode_id.encode()).hexdigest()[:12]      # what the runtime's recorder names recordings/<key>/
    assert episode_directory_name(episode_id) == "00877-4ok3usBNeis_000003-" + key
    assert episode_directory_name("a/b") != episode_directory_name("a_b"), "the slug loses the difference; the hash keeps it"
    for odd in ("../../etc/passwd", "..", "/", "a b\tc", "\u00e9pisode/1", "x" * 500, ".hidden"):
        folder = episode_directory_name(odd)
        assert "/" not in folder and "\\" not in folder and not folder.startswith(".") and len(folder) <= 80 + 13
        assert folder.endswith("-" + hashlib.sha256(odd.encode("utf-8")).hexdigest()[:12])
    for blank in ("", "   ", None, 3):
        with pytest.raises(FrameLogError, match="non-blank string"):
            episode_directory_name(blank)


def test_an_episode_logged_again_replaces_its_frames_and_nothing_else(tmp_path):
    """A run resumed after a crash re-runs the episode it died in: the longer first attempt must not linger."""
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    for step in range(5):
        store.put(record(step), b"first attempt %d" % step)
    directory = store.directory
    (directory / "notes.txt").write_text("mine")
    (directory / "analysis").mkdir()
    (directory / "analysis" / "step_000001.jpg").write_bytes(b"my own copy")
    (directory / "step_000007.jpg.part").write_bytes(b"a write the process died in")
    store.begin(EPISODE, dict(HEADER, note="the second attempt"))
    store.put(record(0), b"second attempt 0")
    assert sorted(p.name for p in directory.iterdir()) == ["analysis", "episode.json", "frames.jsonl", "notes.txt",
                                                           "step_000000.jpg"]
    assert (directory / "notes.txt").read_text() == "mine"
    assert (directory / "analysis" / "step_000001.jpg").read_bytes() == b"my own copy"
    header, records = read_frame_log(directory)
    assert header["note"] == "the second attempt" and list(records) == ["step_000000.jpg"]
    assert store.get("step_000000.jpg") == b"second attempt 0"


def test_each_episode_has_its_own_folder_and_a_finished_one_stays_as_it_was(tmp_path):
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"episode three")
    store.begin("scene-a/000004", HEADER)
    store.put(record(0), b"episode four")
    assert len(list(tmp_path.iterdir())) == 2
    finished = tmp_path / episode_directory_name(EPISODE)
    assert (finished / "step_000000.jpg").read_bytes() == b"episode three"
    assert list(read_frame_log(finished)[1]) == ["step_000000.jpg"]
    store.begin(EPISODE, HEADER)                                    # logged again, this time as PNG
    store.put(record(0, ".png"), b"episode three again")
    assert sorted(p.name for p in finished.glob("step_*")) == ["step_000000.png"]


def test_no_half_written_file_is_left_behind(tmp_path):
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    for step in range(4):
        store.put(record(step), b"x" * 4096)
    assert not list(store.directory.glob("*.part")) and len(list(store.directory.glob("step_*.jpg"))) == 4


def test_a_frame_cannot_take_the_name_of_the_header_or_the_index(tmp_path):
    """Its image would be written over them, and the folder would stop explaining itself."""
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"zero")
    for name in (HEADER_NAME, INDEX_NAME, "step_000001.jpg.part"):
        with pytest.raises(FrameLogError, match="one of the store's own file names"):
            store.put(FrameRecord(name, 1, pose(1)), b"not a frame")
    assert read_frame_log(store.directory) == (HEADER, {"step_000000.jpg": record(0)})


def test_the_root_is_made_with_the_first_episode_and_a_home_relative_root_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = DiskFrameStore("~/runs/frames")
    assert store.root == tmp_path / "runs" / "frames" and not store.root.exists()
    store.begin(EPISODE, HEADER)
    assert store.directory.parent == tmp_path / "runs" / "frames" and store.directory.is_dir()


# -- reading a folder back --------------------------------------------------------------------------------------------
def test_the_index_is_read_back_as_the_lookup_in_capture_order(tmp_path):
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    for step in (0, 1, 2, 10):
        store.put(record(step), b"an image")
    header, records = read_frame_log(store.directory)
    assert header == HEADER
    assert list(records) == ["step_000000.jpg", "step_000001.jpg", "step_000002.jpg", "step_000010.jpg"]
    assert all(records[frame_name(step, ".jpg")] == record(step) for step in (0, 1, 2, 10))
    assert records["step_000010.jpg"].pose == pose(10) and records["step_000010.jpg"].step == 10
    assert read_frame_log(str(store.directory))[1] == records, "a path given as text is read the same"


def test_the_reader_refuses_a_folder_it_cannot_vouch_for(tmp_path):
    with pytest.raises(FrameLogError, match="not an episode's frame folder"):
        read_frame_log(tmp_path / "nowhere")
    store = DiskFrameStore(tmp_path)
    store.begin(EPISODE, HEADER)
    store.put(record(0), b"zero")
    store.put(record(1), b"one")
    index, whole = store.directory / INDEX_NAME, (store.directory / INDEX_NAME).read_text()
    index.write_text(whole[:-25])                                   # the last line cut short, as a kill leaves it
    with pytest.raises(FrameLogError, match=r"frames\.jsonl line 2 of 2 is not a frame record"):
        read_frame_log(store.directory)
    index.write_text(whole + whole.splitlines()[0] + "\n")
    with pytest.raises(FrameLogError, match=r"line 3 of 3 names 'step_000000\.jpg', which an earlier line already did"):
        read_frame_log(store.directory)
    index.write_text(whole)
    assert len(read_frame_log(store.directory)[1]) == 2
    (store.directory / HEADER_NAME).write_text(json.dumps(dict(HEADER, schema="objnav_frame_log/0")))
    with pytest.raises(FrameLogError, match="schema 'objnav_frame_log/0'"):
        read_frame_log(store.directory)
    (store.directory / HEADER_NAME).write_text("[1, 2]")
    with pytest.raises(FrameLogError, match="schema None"):
        read_frame_log(store.directory)
