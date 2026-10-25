"""Replay uses an adapter callback and cannot alter scored artifacts."""
import math

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.replay import render_replay, replay_poses


def rows():
    return [dict(x=0, y=0, z=0, yaw=math.pi - 0.1, camera_pitch=0.0),
            dict(x=1, y=0, z=0.2, yaw=-math.pi + 0.1, camera_pitch=0.2)]


def test_replay_includes_final_pose_and_interpolates_pitch_and_short_yaw_arc():
    poses = list(replay_poses(rows(), 2))
    assert len(poses) == 3 and poses[-1].x == 1 and poses[-1].camera_pitch == 0.2
    assert poses[1].x == 0.5 and poses[1].camera_pitch == 0.1
    assert abs(abs(poses[1].yaw) - math.pi) < 1e-9
    with pytest.raises(ValueError):
        list(replay_poses(rows(), 0))


def test_replay_preserves_artifacts_and_refuses_overwrite(tmp_path):
    (tmp_path / "metrics.json").write_text('{"test": true}')
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text("x,y,z,yaw,camera_pitch\n0,0,0,0,0\n1,0,0,0.2,0.2\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    frames = []

    class Sink:
        def __init__(self, path, fps):
            self.path = path
            assert fps == 12

        def write(self, frame):
            assert frame.shape == (900, 1600, 3)
            frames.append(1)

        def close(self):
            self.path.write_bytes(b"synthetic-writer")

    output = render_replay(tmp_path, lambda pose: np.zeros((16, 32, 3), np.uint8),
                           subframes=2, writer_factory=Sink)
    assert len(frames) == 3 and output.name == "smooth_rgb.mp4"
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    with pytest.raises(FileExistsError):
        render_replay(tmp_path, lambda pose: None, writer_factory=Sink)
    with pytest.raises(ValueError, match="completed"):
        render_replay(tmp_path / "unfinished", lambda pose: None)

