"""Hardware detection must run gracefully on x86 and resolve Jetson fields.

Pure-stdlib, runnable anywhere. The x86 path is exercised directly; the Jetson
path is exercised by feeding the field-fillers synthetic sysfs values, so the
test passes on a dev box with no Tegra hardware.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.navdp.hardware import detect as D


def test_detect_never_raises_and_sets_arch():
    p = D.detect()
    assert p.arch
    assert isinstance(p.is_jetson, bool)
    assert p.recommended_workspace_bytes > 0
    # allow_dla reports hardware capability, not a per-model recommendation --
    # NavDP's own build path never reads it (see hardware/detect.py docstring).
    assert isinstance(p.allow_dla, bool)
    assert p.target_tag


def test_power_budget_parsing():
    assert D._power_budget_w("MODE_15W") == 15
    assert D._power_budget_w("10W") == 10
    assert D._power_budget_w("MAXN") is None
    assert D._power_budget_w(None) is None


def test_workspace_scales_and_caps():
    orin = D.HardwareProfile(arch="aarch64", is_jetson=True,
                             power_budget_w=15, total_mem_bytes=32 << 30)
    ws = D._workspace_bytes(orin)
    assert ws <= (1 << 30)                # 15 W Jetson capped tight
    x86 = D.HardwareProfile(arch="x86_64", is_jetson=False, total_mem_bytes=16 << 30)
    assert D._workspace_bytes(x86) <= (4 << 30)


def test_orin_target_tag():
    orin = D.HardwareProfile(arch="aarch64", is_jetson=True,
                             compute_capability=(8, 7))
    assert D._target_tag(orin) == "orin_sm87"
    assert orin.sm == 87
