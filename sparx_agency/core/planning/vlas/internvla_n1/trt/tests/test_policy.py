"""The policy's orchestration, against stub engines.

TensorRT is not needed to test what this class is actually responsible for: the
order of the three stages, the shape contract at each boundary, uploading the
condition once and reusing it across all ten steps, and refusing to serve an
incomplete selection.
"""
import json

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.internvla_n1.trt import policy as policy_mod


class _StubRunner:
    """Records its feeds and returns a fixed-shape output."""

    def __init__(self, out_shape, value=0.0):
        self.out_shape = out_shape
        self.value = value
        self.calls = []

    def infer(self, feeds):
        self.calls.append(dict((k, np.array(v, copy=True)) for k, v in feeds.items()))
        return [np.full(self.out_shape, self.value, dtype=np.float32)]


def _policy(tmp_path, monkeypatch, samples=32, steps=10):
    """Build the policy with stub runners in place of real engines."""
    (tmp_path / "selected.json").write_text(json.dumps({
        "precision": "fp16",
        "engines": {policy_mod.VISION_KEY: "v.engine",
                    policy_mod.CONDITION_KEY: "c.engine",
                    policy_mod.DENOISE_KEY: "d.engine"}}))
    stubs = {
        "v.engine": _StubRunner((1, 512, 384), 0.5),
        "c.engine": _StubRunner((1, 36, 768), 0.25),
        "d.engine": _StubRunner((samples, 32, 3), 0.1),
    }
    monkeypatch.setattr(policy_mod, "TRTEngineRunner",
                        lambda path, device_id=0: stubs[path.name])
    net = policy_mod.InternVLAN1System1TRT(tmp_path, samples=samples, steps=steps,
                                           seed=0)
    return net, stubs


def test_missing_selection_is_an_actionable_error(tmp_path):
    with pytest.raises(policy_mod.InternVLAN1System1Error, match="no selected.json"):
        policy_mod.InternVLAN1System1TRT(tmp_path)


def test_a_selection_missing_one_engine_is_refused(tmp_path, monkeypatch):
    """A pipeline that cannot be completed must not half-load."""
    (tmp_path / "selected.json").write_text(json.dumps({
        "precision": "fp16",
        "engines": {policy_mod.VISION_KEY: "v.engine"}}))
    monkeypatch.setattr(policy_mod, "TRTEngineRunner",
                        lambda path, device_id=0: _StubRunner((1, 512, 384)))
    with pytest.raises(policy_mod.InternVLAN1System1Error, match="names no engine"):
        policy_mod.InternVLAN1System1TRT(tmp_path)


def test_an_empty_selection_is_refused(tmp_path):
    (tmp_path / "selected.json").write_text(json.dumps({"engines": {}}))
    with pytest.raises(policy_mod.InternVLAN1System1Error, match="lists no engines"):
        policy_mod.InternVLAN1System1TRT(tmp_path)


def test_denoise_runs_once_per_step_with_the_scheduled_timesteps(tmp_path, monkeypatch):
    net, stubs = _policy(tmp_path, monkeypatch, steps=10)
    net.denoise(np.zeros((1, 36, 768), dtype=np.float32))
    calls = stubs["d.engine"].calls
    assert len(calls) == 10
    assert [float(c[policy_mod.IN_TIMESTEP][0]) for c in calls] == [
        1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0]


def test_the_condition_is_identical_on_every_step(tmp_path, monkeypatch):
    """It is uploaded once and left resident; recomputing it would be a bug."""
    net, stubs = _policy(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    net.denoise(rng.standard_normal((1, 36, 768)).astype(np.float32))
    conditions = [c[policy_mod.IN_CONDITION] for c in stubs["d.engine"].calls]
    for other in conditions[1:]:
        assert np.array_equal(conditions[0], other)
    assert conditions[0].shape == (32, 36, 768)


def test_step_count_changes_without_a_rebuild(tmp_path, monkeypatch):
    """The denoise engine is one step, so the loop length is a runtime knob."""
    net, stubs = _policy(tmp_path, monkeypatch, steps=4)
    net.denoise(np.zeros((1, 36, 768), dtype=np.float32))
    assert len(stubs["d.engine"].calls) == 4


def test_injected_noise_is_used_verbatim(tmp_path, monkeypatch):
    """Sharing the initial sample is what makes an A/B attributable."""
    net, stubs = _policy(tmp_path, monkeypatch)
    noise = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    net.denoise(np.zeros((1, 36, 768), dtype=np.float32), noise=noise)
    assert np.array_equal(stubs["d.engine"].calls[0][policy_mod.IN_LATENTS], noise)


def test_predict_actions_runs_the_three_stages_in_order(tmp_path, monkeypatch):
    net, stubs = _policy(tmp_path, monkeypatch)
    images = np.zeros((1, 2, 224, 224, 3), dtype=np.float32)
    latents = np.zeros((1, 4, 3584), dtype=np.float32)
    actions, trajectories = net.predict_actions(images, latents)
    assert len(stubs["v.engine"].calls) == 1
    assert len(stubs["c.engine"].calls) == 1
    assert len(stubs["d.engine"].calls) == 10
    assert trajectories.shape == (32, 32, 3)
    assert all(a in (1, 2, 3) for a in actions)


def test_a_wrong_input_shape_names_the_tensor_and_both_shapes(tmp_path, monkeypatch):
    net, _ = _policy(tmp_path, monkeypatch)
    with pytest.raises(policy_mod.InternVLAN1System1Error, match="images has shape"):
        net.encode_vision(np.zeros((1, 3, 224, 224, 3), dtype=np.float32))


def test_traj_latents_must_be_the_vlm_width(tmp_path, monkeypatch):
    net, _ = _policy(tmp_path, monkeypatch)
    with pytest.raises(policy_mod.InternVLAN1System1Error, match="traj_latents"):
        net.encode_condition(np.zeros((1, 512, 384), dtype=np.float32),
                             np.zeros((1, 4, 1024), dtype=np.float32))
