"""Sidecar roundtrip + compatibility-validation tests. No GPU, no TensorRT."""
import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.model_registry.errors import IncompatibleArtifactError
from sparx_agency.tasks.common.model_registry.key import ModelKey
from sparx_agency.tasks.common.model_registry.sidecar import (
    build_sidecar, read_sidecar, validate_sidecar, write_sidecar,
)


def _profile(**overrides):
    base = dict(arch="x86_64", is_jetson=False, gpu_name="RTX 4090",
               compute_capability=(8, 9), target_tag="rtx4090_sm89")
    base.update(overrides)
    return HardwareProfile(**base)


def test_write_then_read_roundtrip(tmp_path):
    engine = tmp_path / "x.engine"
    engine.write_bytes(b"fake engine bytes")
    key = ModelKey(model_id="da3_metric_large", role="depth_only", precision="fp16",
                  height=546, width=364, target_tag="rtx4090_sm89")
    data = build_sidecar(key, _profile(), origin="built", trt_version="10.3.0")
    write_sidecar(engine, data)

    loaded = read_sidecar(engine)
    assert loaded["model_id"] == "da3_metric_large"
    assert loaded["sm"] == 89
    assert loaded["arch"] == "x86_64"


def test_read_sidecar_missing_returns_none(tmp_path):
    assert read_sidecar(tmp_path / "nope.engine") is None


def test_validate_sidecar_none_is_a_noop():
    validate_sidecar(None, profile=_profile())  # must not raise


def test_validate_sidecar_arch_mismatch_raises():
    sidecar = {"arch": "aarch64", "sm": 87}
    with pytest.raises(IncompatibleArtifactError):
        validate_sidecar(sidecar, profile=_profile())


def test_validate_sidecar_sm_mismatch_raises():
    sidecar = {"arch": "x86_64", "sm": 120}
    with pytest.raises(IncompatibleArtifactError):
        validate_sidecar(sidecar, profile=_profile())


def test_validate_sidecar_matching_profile_passes():
    sidecar = {"arch": "x86_64", "sm": 89}
    validate_sidecar(sidecar, profile=_profile())  # must not raise
