"""Tests for the recording schema + offline label generation (numpy, no torch)."""
import numpy as np

from sparx_agency.tasks.planning.vlas.common.finetune.datasets.esdf_label_gen import generate_labels
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
    load_recording,
    synthesize_recording,
)


def test_synthesize_and_load(tmp_path):
    rec = synthesize_recording(tmp_path / "rec", num_frames=8)
    assert rec.num_frames == 8
    reloaded = load_recording(tmp_path / "rec")
    assert reloaded.num_frames == 8
    assert reloaded.intrinsics.width == 160


def test_future_path_and_goal_body(tmp_path):
    rec = synthesize_recording(tmp_path / "rec", num_frames=10)
    fut = rec.future_path_body(0, horizon=5)
    assert fut.shape[1] == 2
    # straight +x flight -> future is forward, ~zero lateral
    assert fut[-1, 0] > 0.5
    assert abs(fut[:, 1]).max() < 1e-3
    gf, gl = rec.goal_body(0, lookahead=4)
    assert gf > 0.5 and abs(gl) < 1e-3


def test_generate_labels_writes_npz(tmp_path):
    rec = synthesize_recording(tmp_path / "rec", num_frames=8)
    n = generate_labels(rec, tmp_path / "labels", goal_lookahead=5)
    assert n == 8
    z = np.load(tmp_path / "labels" / "000000.npz")
    assert z["navdp"].shape == (24, 3)
    assert z["flownav"].shape == (8, 2)
    assert z["sdf"].ndim == 2
    assert float(z["navdp"].max()) <= 1.0 and float(z["navdp"].min()) >= -1.0
