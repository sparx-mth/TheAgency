"""Tests for the hardware probe + build-policy derivation (no torch/TRT needed).

Verifies :func:`detect` never raises and yields a stable tag, and that
:func:`build_policy` makes the DLA / precision / opt-level decisions the builder
depends on -- including that DLA is only *requested* where the board has one.
"""
import dataclasses

from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import build_policy
from sparx_agency.tasks.mapping.yolo_world_trt.hardware import HardwareProfile, detect


def test_detect_never_raises_and_tags():
    p = detect()
    assert isinstance(p.target_tag, str) and p.target_tag
    assert p.recommended_workspace_bytes > 0
    dataclasses.asdict(p)               # fully serializable


def _orin_15w():
    return HardwareProfile(
        arch="aarch64", is_jetson=True, gpu_name="Orin",
        compute_capability=(8, 7), dla_cores=2, allow_dla=True,
        power_budget_w=15, total_mem_bytes=64 << 30,
        recommended_workspace_bytes=1 << 30, target_tag="orin_sm87")


def _x86():
    return HardwareProfile(
        arch="x86_64", is_jetson=False, gpu_name="RTX 5070",
        compute_capability=(12, 0), dla_cores=0, allow_dla=False,
        total_mem_bytes=8 << 30, recommended_workspace_bytes=4 << 30,
        target_tag="rtx5070_sm120")


def test_orin_enables_dla_fp16_default():
    pol = build_policy("s", _orin_15w(), config={})
    assert pol.use_dla is True
    assert pol.precision == "fp16" and pol.use_fp16 is True
    assert pol.gpu_fallback is True
    assert pol.imgsz == (288, 512)      # code default when config empty


def test_x86_never_uses_dla_even_if_requested():
    pol = build_policy("s", _x86(), config={}, dla=True)
    assert pol.use_dla is False         # board has no DLA -> request ignored


def test_no_dla_override_forces_gpu_on_orin():
    pol = build_policy("l", _orin_15w(), config={}, dla=False)
    assert pol.use_dla is False


def test_int8_precision_flag():
    pol = build_policy("x", _orin_15w(), config={}, precision="int8")
    assert pol.use_int8 is True and pol.precision == "int8"
    assert pol.use_fp16 is True          # FP16 stays on as the floor
