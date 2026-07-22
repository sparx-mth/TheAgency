"""Tests for the hardware probe + per-role build-policy (no torch/TRT needed).

Verifies :func:`detect` never raises, and that :func:`build_policy` makes the
split's core decisions: the backbone prefers DLA (only where the board has one)
and is static, while the head is always GPU + dynamic (DLA cannot run its runtime
prompt-count shapes).
"""
import dataclasses

import pytest

from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import build_policy
from sparx_agency.tasks.mapping.yolo_world_trt.hardware import HardwareProfile, detect


def test_detect_never_raises_and_tags():
    p = detect()
    assert isinstance(p.target_tag, str) and p.target_tag
    assert p.recommended_workspace_bytes > 0
    dataclasses.asdict(p)


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


def test_backbone_uses_dla_on_orin_and_is_static():
    pol = build_policy("backbone", "s", _orin_15w(), config={})
    assert pol.use_dla is True and pol.gpu_fallback is True
    assert pol.is_dynamic is False
    assert pol.use_fp16 is True


def test_head_is_always_gpu_and_dynamic():
    pol = build_policy("head", "s", _orin_15w(), config={}, dla=True)
    assert pol.use_dla is False          # DLA can't do dynamic shapes
    assert pol.is_dynamic is True


def test_backbone_never_uses_dla_on_x86():
    pol = build_policy("backbone", "l", _x86(), config={}, dla=True)
    assert pol.use_dla is False


def test_no_dla_override_forces_gpu_backbone_on_orin():
    pol = build_policy("backbone", "l", _orin_15w(), config={}, dla=False)
    assert pol.use_dla is False


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        build_policy("neck", "s", _orin_15w(), config={})
