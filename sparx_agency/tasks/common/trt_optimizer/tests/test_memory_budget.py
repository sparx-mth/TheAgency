"""Tests for the device-memory budget.

The two targets are built by hand rather than detected, which is the point of
:mod:`..memory_budget` accepting a profile: the Orin case has to be exercised on
an x86 workstation. The numbers for the x86 target are the real measured ones
from the RTX 5070 Laptop (8151 MiB), so a regression in the free-memory or
context reservations shows up as a changed verdict here, not on the aircraft.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer.memory_budget import (
    GIB, MIB, MEASURED_TRT11_DEFAULT_POOL_BYTES, Budget, builder_pool_limits,
    bytes_per_param, cuda_context_bytes, estimate, recommendations)
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, Component, Plan


def x86_8gb():
    """The dev workstation: RTX 5070 Laptop, 8151 MiB, display attached."""
    return HardwareProfile(
        arch="x86_64", is_jetson=False,
        gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        compute_capability=(12, 0), total_mem_bytes=8151 * MIB,
        recommended_workspace_bytes=4 * GIB, target_tag="nvidiageforcert_sm120")


def orin_16gb_15w():
    """A 16 GB AGX Orin pinned to 15 W: unified memory, tight everything."""
    return HardwareProfile(
        arch="aarch64", is_jetson=True, gpu_name="NVIDIA Jetson AGX Orin",
        jetson_model="NVIDIA Jetson AGX Orin", nvpmodel_id=2,
        nvpmodel_name="MODE_15W", power_budget_w=15, compute_capability=(8, 7),
        total_mem_bytes=16 * GIB, dla_cores=2, allow_dla=True,
        recommended_workspace_bytes=1 * GIB, target_tag="orin_sm87")


def plan_with(*params):
    """A Plan whose components carry the given parameter counts."""
    comps = [Component(name="c%d" % i, params=n, cadence=Cadence.PER_FRAME)
             for i, n in enumerate(params)]
    return Plan(model="test_vla", target_tag="test", components=comps)


# --------------------------------------------------------------------- widths

@pytest.mark.parametrize("precision,width", [
    ("fp32", 4), ("float32", 4), ("FP32", 4), ("tf32", 4),
    ("fp16", 2), ("float16", 2), ("half", 2), ("bf16", 2), ("bfloat16", 2),
    ("int8", 1), ("uint8", 1), ("fp8", 1),
    ("int4", 0.5), ("nvfp4", 0.5), ("nv-fp4", 0.5), ("NV_FP4", 0.5),
])
def test_bytes_per_param_every_precision(precision, width):
    assert bytes_per_param(precision) == width


@pytest.mark.parametrize("bad", ["fp24", "", "int3", "  ", "float3.5", None])
def test_bytes_per_param_raises_on_unknown(bad):
    with pytest.raises(ValueError):
        bytes_per_param(bad)


def test_bytes_per_param_message_names_the_offender():
    with pytest.raises(ValueError, match="fp24"):
        bytes_per_param("fp24")


# --------------------------------------------------------------------- Budget

def test_budget_required_sums_every_claim_but_not_the_device():
    b = Budget(total_bytes=100, free_bytes=90, weight_bytes=10,
               activation_bytes=4, workspace_bytes=3,
               runtime_overhead_bytes=2, streaming_scratch_bytes=1)
    assert b.required_bytes == 20
    assert b.headroom_bytes == 70
    assert b.fits


def test_budget_defaults_are_zero_and_an_empty_budget_fits():
    b = Budget()
    assert b.required_bytes == 0
    assert b.headroom_bytes == 0
    assert b.fits


def test_budget_headroom_goes_negative_and_fits_flips():
    b = Budget(total_bytes=8 * GIB, free_bytes=4 * GIB, weight_bytes=6 * GIB)
    assert b.headroom_bytes == -2 * GIB
    assert not b.fits


def test_budget_exactly_full_still_fits():
    b = Budget(free_bytes=1024, weight_bytes=1024)
    assert b.headroom_bytes == 0
    assert b.fits


# ------------------------------------------------------------ context reserve

def test_cuda_context_headless_is_cheaper_than_display_attached():
    hw = x86_8gb()
    assert cuda_context_bytes(hw, headless=True) < cuda_context_bytes(hw)
    assert cuda_context_bytes(hw) == 600 * MIB
    assert cuda_context_bytes(hw, headless=True) == 300 * MIB


def test_cuda_context_default_assumes_a_display_is_attached():
    """The conservative direction: over-reserve on paper, never in flight."""
    hw = x86_8gb()
    assert cuda_context_bytes(hw, headless=None) == cuda_context_bytes(hw)


def test_cuda_context_is_larger_on_unified_memory():
    jetson = cuda_context_bytes(orin_16gb_15w())
    assert jetson > cuda_context_bytes(x86_8gb())
    # A display flag cannot talk a Jetson down: the SoC pays either way.
    assert cuda_context_bytes(orin_16gb_15w(), headless=True) == jetson


# ------------------------------------------------------------- builder limits

def test_builder_pool_limits_caps_both_pools_well_under_the_trt11_default():
    limits = builder_pool_limits(x86_8gb())
    assert set(limits) == {"WORKSPACE", "TACTIC_DRAM"}
    assert limits["WORKSPACE"] < limits["TACTIC_DRAM"]
    for value in limits.values():
        assert 0 < value < MEASURED_TRT11_DEFAULT_POOL_BYTES
        assert value < x86_8gb().total_mem_bytes


def test_builder_pool_limits_hits_the_documented_8gb_numbers():
    limits = builder_pool_limits(x86_8gb())
    assert 1.9 * GIB < limits["WORKSPACE"] <= 2 * GIB
    assert 3.9 * GIB < limits["TACTIC_DRAM"] <= 4 * GIB


def test_builder_pool_limits_are_tighter_on_jetson():
    x86 = builder_pool_limits(x86_8gb())
    orin = builder_pool_limits(orin_16gb_15w())
    assert orin["WORKSPACE"] < x86["WORKSPACE"]
    assert orin["TACTIC_DRAM"] < x86["TACTIC_DRAM"]
    # Nowhere near the 75%-of-unified-memory default that would starve ROS.
    assert orin["TACTIC_DRAM"] < 0.75 * orin_16gb_15w().total_mem_bytes / 4


def test_builder_pool_limits_15w_is_tighter_than_full_power():
    full = orin_16gb_15w()
    full.nvpmodel_name = "MAXN"
    full.power_budget_w = 60
    assert not full.is_15w
    assert (builder_pool_limits(orin_16gb_15w())["TACTIC_DRAM"]
            < builder_pool_limits(full)["TACTIC_DRAM"])


def test_builder_pool_limits_raises_on_an_unknown_device_size():
    hw = x86_8gb()
    hw.total_mem_bytes = 0
    with pytest.raises(ValueError, match="total_mem_bytes"):
        builder_pool_limits(hw)


# ------------------------------------------------------------------- estimate

def test_estimate_small_plan_fits_the_8gb_card():
    budget = estimate(plan_with(300_000_000), "fp16", x86_8gb())
    assert budget.fits
    assert budget.weight_bytes == 600_000_000
    assert budget.activation_bytes == 210_000_000
    assert budget.total_bytes == 8151 * MIB
    assert budget.free_bytes < budget.total_bytes
    assert budget.streaming_scratch_bytes == 0


def test_estimate_multi_gb_vla_does_not_fit_the_8gb_card():
    budget = estimate(plan_with(7_000_000_000), "fp16", x86_8gb())
    assert not budget.fits
    assert budget.headroom_bytes < 0


def test_estimate_counts_cold_components_too():
    """A once-per-episode text encoder is resident all episode."""
    hot = Component(name="hot", params=1_000_000_000)
    cold = Component(name="cold", params=1_000_000_000,
                     cadence=Cadence.ONCE_PER_EPISODE)
    plan = Plan(model="m", components=[hot, cold])
    assert (estimate(plan, "fp16", x86_8gb()).weight_bytes
            == 2 * estimate(plan_with(1_000_000_000), "fp16",
                            x86_8gb()).weight_bytes)


def test_residency_mode_flips_the_verdict():
    plan = plan_with(1_000_000_000, 1_000_000_000)
    concurrent = estimate(plan, "fp16", x86_8gb())
    sequential = estimate(plan, "fp16", x86_8gb(), resident="sequential")
    assert concurrent.weight_bytes == 4_000_000_000
    assert sequential.weight_bytes == 2_000_000_000
    assert not concurrent.fits
    assert sequential.fits


def test_precision_flips_the_verdict():
    plan = plan_with(2_000_000_000)
    assert not estimate(plan, "fp32", x86_8gb()).fits
    assert estimate(plan, "int8", x86_8gb()).fits


def test_estimate_jetson_free_fraction_is_stingier_than_a_dgpu():
    orin = orin_16gb_15w()
    budget = estimate(plan_with(1_000_000_000), "fp16", orin)
    assert budget.free_bytes < 0.6 * orin.total_mem_bytes
    assert budget.runtime_overhead_bytes == cuda_context_bytes(orin)
    assert budget.workspace_bytes == builder_pool_limits(orin)["WORKSPACE"]


def test_estimate_activation_factor_is_linear_and_zero_is_allowed():
    plan = plan_with(1_000_000_000)
    assert estimate(plan, "fp16", x86_8gb(),
                    activation_factor=0.0).activation_bytes == 0
    doubled = estimate(plan, "fp16", x86_8gb(), activation_factor=0.7)
    single = estimate(plan, "fp16", x86_8gb(), activation_factor=0.35)
    assert doubled.activation_bytes == 2 * single.activation_bytes


def test_estimate_empty_plan_still_charges_context_and_workspace():
    budget = estimate(Plan(model="empty"), "fp16", x86_8gb())
    assert budget.weight_bytes == 0
    assert budget.required_bytes > 0
    assert budget.fits


def test_estimate_raises_on_a_bad_residency_mode():
    with pytest.raises(ValueError, match="resident"):
        estimate(plan_with(1000), "fp16", x86_8gb(), resident="overlapped")


def test_estimate_raises_on_a_negative_activation_factor():
    with pytest.raises(ValueError, match="activation_factor"):
        estimate(plan_with(1000), "fp16", x86_8gb(), activation_factor=-0.1)


def test_estimate_raises_on_an_unknown_precision():
    with pytest.raises(ValueError, match="fp24"):
        estimate(plan_with(1000), "fp24", x86_8gb())


def test_estimate_raises_on_an_unknown_device_size():
    hw = x86_8gb()
    hw.total_mem_bytes = 0
    with pytest.raises(ValueError, match="total_mem_bytes"):
        estimate(plan_with(1000), "fp16", hw)


# ------------------------------------------------------------ recommendations

def test_recommendations_are_empty_when_it_fits():
    budget = estimate(plan_with(200_000_000), "fp16", x86_8gb())
    assert budget.fits
    assert recommendations(budget, x86_8gb()) == []


def test_recommendations_lead_with_precision_and_end_with_offload():
    budget = estimate(plan_with(7_000_000_000), "fp16", x86_8gb())
    recs = recommendations(budget, x86_8gb(), precision="fp16")
    assert len(recs) == 5
    assert recs[0].startswith("Drop the weight precision one step (fp16 -> int8)")
    assert "sidecar" in recs[-1]
    assert any("STRIP_PLAN" in r and "REFIT" in r for r in recs)
    assert any("sequential" in r for r in recs)


def test_precision_remedy_is_suppressed_at_the_floor():
    budget = estimate(plan_with(30_000_000_000), "nvfp4", x86_8gb())
    assert not budget.fits
    recs = recommendations(budget, x86_8gb(), precision="nvfp4")
    assert not any(r.startswith("Drop the weight precision") for r in recs)


def test_precision_remedy_is_generic_without_a_precision_argument():
    budget = estimate(plan_with(7_000_000_000), "fp16", x86_8gb())
    recs = recommendations(budget, x86_8gb())
    assert "halves the parameter width" in recs[0]


def test_weight_streaming_only_past_half_of_free_memory():
    hw = x86_8gb()
    free = estimate(plan_with(0), "fp16", hw).free_bytes

    just_under = Budget(total_bytes=hw.total_mem_bytes, free_bytes=free,
                        weight_bytes=int(0.49 * free),
                        runtime_overhead_bytes=free)  # forced not to fit
    just_over = Budget(total_bytes=hw.total_mem_bytes, free_bytes=free,
                       weight_bytes=int(0.51 * free),
                       runtime_overhead_bytes=free)
    assert not just_under.fits and not just_over.fits
    assert not any("WEIGHT_STREAMING" in r
                   for r in recommendations(just_under, hw))
    assert any("WEIGHT_STREAMING" in r for r in recommendations(just_over, hw))


def test_weight_streaming_quotes_nvidias_budget_formula():
    hw = x86_8gb()
    budget = estimate(plan_with(7_000_000_000), "fp16", hw)
    rec = [r for r in recommendations(budget, hw) if "WEIGHT_STREAMING" in r][0]
    assert "min(free // 2, streamable_weights_size // 2)" in rec


def test_weight_streaming_is_refused_on_unified_memory():
    hw = orin_16gb_15w()
    budget = estimate(plan_with(6_000_000_000), "fp16", hw)
    assert not budget.fits
    recs = recommendations(budget, hw)
    streaming = [r for r in recs if "WEIGHT_STREAMING" in r]
    assert len(streaming) == 1
    assert streaming[0].startswith("Do NOT")
    assert "same physical memory" in streaming[0]


def test_recommendations_skip_weight_items_when_there_are_no_weights():
    """An overhead-only overflow cannot be fixed by touching the weights."""
    budget = Budget(total_bytes=GIB, free_bytes=GIB, weight_bytes=0,
                    workspace_bytes=2 * GIB)
    recs = recommendations(budget, x86_8gb())
    assert not any("STRIP_PLAN" in r for r in recs)
    assert not any(r.startswith("Drop the weight precision") for r in recs)
    assert recs  # residency and offload still apply


def test_recommendations_report_the_actual_deficit():
    budget = Budget(free_bytes=GIB, weight_bytes=3 * GIB)
    recs = recommendations(budget, x86_8gb())
    assert any("2.00 GiB" in r for r in recs)
