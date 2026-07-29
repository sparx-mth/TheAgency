"""A simulated recording must carry full ground truth and still load as the old schema.

Two properties are load-bearing and easy to break:

* Columns 0-3 of ``poses.npy`` stay ``[t, x, y, yaw]``, because
  :class:`FlightRecording` slices them positionally and every label generator
  goes through that.
* A depth frame reads back in **metres** whatever container it was stored in.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
    load_recording,
)
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import (
    DEPTH_FORMAT_NPY, DEPTH_FORMAT_PNG, POSE_COLUMNS, FlightWriter, SimFrame,
    clamp_depth, export_flight, parse_resolution, path_length_m, resolution_of,
)

INTRINSICS = Intrinsics(width=16, height=12, fx=8.0, fy=8.0, cx=8.0, cy=6.0)


def _frame(index: int, depth_value: float = 3.0) -> SimFrame:
    return SimFrame(
        depth=np.full((12, 16), depth_value, np.float32),
        pose=(float(index), 2.0 * index, 0.25 * index),
        rgb=np.full((12, 16, 3), index * 10, np.uint8),
        stamp_s=0.1 * index,
        z=1.5,
        quaternion=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=(1.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.1),
    )


def _export(tmp_path, count: int = 4, **kwargs) -> dict:
    return export_flight(
        [_frame(i) for i in range(count)], tmp_path, INTRINSICS,
        rate_hz=10.0, camera_height_m=1.5, pitch_deg=0.0, **kwargs,
    )


def test_pose_columns_start_with_the_legacy_schema():
    assert POSE_COLUMNS[:4] == ("t", "x", "y", "yaw")


def test_full_pose_is_written_and_the_first_four_columns_are_unchanged(tmp_path):
    _export(tmp_path)
    poses = np.load(tmp_path / "poses.npy")

    assert poses.shape == (4, len(POSE_COLUMNS))
    np.testing.assert_allclose(poses[:, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(poses[:, 1], [0.0, 1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(poses[:, 2], [0.0, 2.0, 4.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(poses[:, 3], [0.0, 0.25, 0.5, 0.75], atol=1e-6)
    np.testing.assert_allclose(poses[:, 4], 1.5)                          # z
    np.testing.assert_allclose(poses[:, 5:9], np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)))
    np.testing.assert_allclose(poses[:, 9:12], np.tile([1.0, 0.0, 0.0], (4, 1)))
    np.testing.assert_allclose(poses[:, 12:15], np.tile([0.0, 0.0, 0.1], (4, 1)), atol=1e-7)


def test_a_frame_without_extra_state_still_exports(tmp_path):
    """The original two-field SimFrame construction must keep working."""
    frames = [SimFrame(depth=np.ones((12, 16), np.float32), pose=(0.0, 0.0, 0.0))]
    stats = export_flight(frames, tmp_path, INTRINSICS, rate_hz=10.0,
                          camera_height_m=1.0, pitch_deg=0.0)
    assert stats["frames"] == 1
    assert not stats["has_rgb"]


def test_the_stamp_is_the_simulation_clock_not_a_frame_index(tmp_path):
    frames = [SimFrame(depth=np.ones((12, 16), np.float32), pose=(0.0, 0.0, 0.0),
                       stamp_s=t) for t in (0.0, 0.13, 0.31)]
    export_flight(frames, tmp_path, INTRINSICS, rate_hz=10.0,
                  camera_height_m=1.0, pitch_deg=0.0)
    np.testing.assert_allclose(np.load(tmp_path / "poses.npy")[:, 0],
                               [0.0, 0.13, 0.31], atol=1e-6)


def test_frames_without_a_stamp_fall_back_to_the_nominal_rate(tmp_path):
    frames = [SimFrame(depth=np.ones((12, 16), np.float32), pose=(0.0, 0.0, 0.0))
              for _ in range(3)]
    export_flight(frames, tmp_path, INTRINSICS, rate_hz=4.0,
                  camera_height_m=1.0, pitch_deg=0.0)
    np.testing.assert_allclose(np.load(tmp_path / "poses.npy")[:, 0],
                               [0.0, 0.25, 0.5], atol=1e-6)


@pytest.mark.parametrize("depth_format", [DEPTH_FORMAT_PNG, DEPTH_FORMAT_NPY])
def test_depth_round_trips_in_metres_whatever_it_was_stored_as(tmp_path, depth_format):
    _export(tmp_path, depth_format=depth_format)
    recording = load_recording(tmp_path)

    depth = recording.depth(0)
    assert depth.shape == (12, 16)
    np.testing.assert_allclose(depth, 3.0, atol=1e-3)


def test_png_depth_is_much_smaller_than_float32(tmp_path):
    """The reason PNG is the default: a campaign is storage-bound."""
    big = [SimFrame(depth=np.full((240, 320), 5.0, np.float32), pose=(0.0, 0.0, 0.0))]
    export_flight(big, tmp_path / "npy", INTRINSICS, rate_hz=10.0, camera_height_m=1.0,
                  pitch_deg=0.0, depth_format=DEPTH_FORMAT_NPY)
    export_flight(big, tmp_path / "png", INTRINSICS, rate_hz=10.0, camera_height_m=1.0,
                  pitch_deg=0.0, depth_format=DEPTH_FORMAT_PNG)

    npy_bytes = (tmp_path / "npy" / "depth" / "000000.npy").stat().st_size
    png_bytes = (tmp_path / "png" / "depth" / "000000.png").stat().st_size
    assert png_bytes < npy_bytes / 4


def test_an_unknown_depth_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown depth_format"):
        _export(tmp_path, depth_format="tiff")


def test_the_recording_loads_through_the_shared_reader(tmp_path):
    _export(tmp_path)
    recording = load_recording(tmp_path)

    assert recording.num_frames == 4
    assert recording.has_full_pose
    assert recording.intrinsics == INTRINSICS
    assert recording.camera_height_m == pytest.approx(1.5)
    assert recording.rgb(0) is not None
    np.testing.assert_allclose(recording.pose(1), [1.0, 2.0, 0.25], atol=1e-6)
    assert recording.pose_full(1).shape == (len(POSE_COLUMNS),)


def test_label_generation_geometry_still_runs_against_it(tmp_path):
    """future_path_body is what the ESDF label generator calls."""
    _export(tmp_path)
    recording = load_recording(tmp_path)

    path = recording.future_path_body(0, horizon=3)
    assert path.shape[1] == 2
    assert path[0].tolist() == [0.0, 0.0]
    assert path[1][0] > 0.0, "the drone flew forward, so the body path must lead forward"


def test_a_legacy_four_column_recording_still_loads(tmp_path):
    (tmp_path / "depth").mkdir(parents=True)
    np.save(tmp_path / "depth" / "000000.npy", np.ones((12, 16), np.float32))
    np.save(tmp_path / "poses.npy", np.zeros((1, 4), np.float32))
    (tmp_path / "intrinsics.json").write_text(json.dumps({
        "width": 16, "height": 12, "fx": 8.0, "fy": 8.0, "cx": 8.0, "cy": 6.0}))
    (tmp_path / "meta.json").write_text(json.dumps({"rate_hz": 10.0, "frames": 1}))

    recording = load_recording(tmp_path)
    assert recording.num_frames == 1
    assert not recording.has_full_pose
    np.testing.assert_allclose(recording.depth(0), 1.0)


def test_meta_records_the_provenance_it_was_given(tmp_path):
    stats = _export(tmp_path, extra_meta={"scene": "office", "outcome": "landed"})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["scene"] == "office"
    assert meta["outcome"] == "landed"
    assert stats["duration_s"] == pytest.approx(0.3)
    assert stats["path_length_m"] > 0.0


def test_extra_meta_cannot_overwrite_the_schema(tmp_path):
    stats = _export(tmp_path, extra_meta={"frames": 9999, "depth_format": "lies"})
    assert stats["frames"] == 4
    assert stats["depth_format"] == DEPTH_FORMAT_PNG


def test_exporting_nothing_raises_rather_than_leaving_an_empty_directory(tmp_path):
    out_dir = tmp_path / "empty"
    with pytest.raises(ValueError):
        export_flight([], out_dir, INTRINSICS, rate_hz=10.0,
                      camera_height_m=1.0, pitch_deg=0.0)
    assert not out_dir.exists()


def test_a_flight_that_captured_nothing_is_discarded_not_raised_over(tmp_path):
    """A campaign has to survive an episode that never got airborne."""
    out_dir = tmp_path / "never_flew"
    writer = FlightWriter(out_dir, INTRINSICS, rate_hz=10.0, camera_height_m=1.0,
                          pitch_deg=0.0)

    stats = writer.close({"outcome": "arm_timeout"})

    assert stats["frames"] == 0
    assert stats["discarded"] is True
    assert stats["outcome"] == "arm_timeout"
    assert not out_dir.exists(), "an empty directory would look like a recording"


def test_writer_streams_frames_to_disk_as_they_arrive(tmp_path):
    writer = FlightWriter(tmp_path, INTRINSICS, rate_hz=10.0, camera_height_m=1.0,
                          pitch_deg=0.0)
    writer.append(_frame(0))
    assert (tmp_path / "depth" / "000000.png").exists(), "not buffered until close()"
    assert writer.frames == 1
    writer.append(_frame(1))
    assert writer.close()["frames"] == 2


def test_non_finite_depth_is_saturated_not_propagated():
    depth = np.array([[np.inf, np.nan, -np.inf, 5.0]], np.float32)
    clamped = clamp_depth(depth, max_depth_m=20.0)
    assert np.isfinite(clamped).all()
    assert clamped.tolist() == [[20.0, 20.0, 0.0, 5.0]]


def test_path_length_uses_all_three_axes():
    poses = np.zeros((2, 15), np.float32)
    poses[1, 1] = 3.0   # x
    poses[1, 4] = 4.0   # z
    assert path_length_m(poses) == pytest.approx(5.0)
    assert path_length_m(poses[:1]) == 0.0


def test_intrinsics_rescale_by_the_axis_ratios():
    scaled = resolution_of(INTRINSICS, 32, 6)
    assert (scaled.width, scaled.height) == (32, 6)
    assert scaled.fx == pytest.approx(16.0)
    assert scaled.cx == pytest.approx(16.0)
    assert scaled.fy == pytest.approx(4.0)
    assert scaled.cy == pytest.approx(3.0)


def test_intrinsics_rescale_rejects_a_degenerate_size():
    with pytest.raises(ValueError):
        resolution_of(INTRINSICS, 0, 10)


@pytest.mark.parametrize("text,expected", [("640x480", (640, 480)), ("504X392", (504, 392))])
def test_resolution_parsing(text, expected):
    assert parse_resolution(text) == expected


@pytest.mark.parametrize("text", ["640", "640x", "axb", "-1x10", "640x480x3"])
def test_bad_resolution_strings_are_rejected(text):
    with pytest.raises(ValueError):
        parse_resolution(text)
