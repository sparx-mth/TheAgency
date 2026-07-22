"""Hardware-detection unit tests (no GPU needed: pure parsing/derivation)."""
from sparx_agency.tasks.planning.flownav.hardware.detect import (
    HardwareProfile, _power_budget_w, _target_tag, _workspace_bytes,
)


def test_sm_property():
    p = HardwareProfile(arch="x86_64", is_jetson=False, compute_capability=(12, 0))
    assert p.sm == 120
    assert HardwareProfile(arch="x86_64", is_jetson=False).sm is None


def test_power_budget_parsing():
    assert _power_budget_w("MODE_15W") == 15
    assert _power_budget_w("MAXN") is None
    assert _power_budget_w(None) is None


def test_workspace_jetson_15w_is_capped():
    p = HardwareProfile(arch="aarch64", is_jetson=True, power_budget_w=15,
                        total_mem_bytes=64 << 30)
    assert _workspace_bytes(p) == 1 << 30        # 1 GiB cap at <=15 W


def test_workspace_x86_half_mem_capped_at_4g():
    p = HardwareProfile(arch="x86_64", is_jetson=False, total_mem_bytes=16 << 30)
    assert _workspace_bytes(p) == 4 << 30        # min(4 GiB, mem/2)


def test_target_tag_orin_and_x86():
    orin = HardwareProfile(arch="aarch64", is_jetson=True, compute_capability=(8, 7))
    assert _target_tag(orin) == "orin_sm87"
    x86 = HardwareProfile(arch="x86_64", is_jetson=False,
                          gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
                          compute_capability=(12, 0))
    assert _target_tag(x86) == "nvidiageforcertx_sm120"
