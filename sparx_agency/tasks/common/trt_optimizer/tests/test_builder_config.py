"""Tests for the TensorRT builder-knob policy.

No GPU and no ``tensorrt``: :mod:`..engine.builder_config` reaches the real
module only through ``trt.BuilderFlag``, ``trt.MemoryPoolType``,
``trt.ProfilingVerbosity`` and ``trt.HardwareCompatibilityLevel``, all by name,
so a namespace with a controllable member list is a complete stand-in. It is
also a *better* one than the real thing for the case that matters most -- a
flag this TensorRT does not have -- which the 11.1 install on this machine
cannot produce on demand.

``_TRT11_FLAGS`` is the measured ``BuilderFlag`` member list of TensorRT
11.1.0.106 (no FP16, no INT8: precision is baked into the ONNX instead).
``_TRT8_FLAGS`` is an older generation that still carries FP16/INT8 and has
neither STRIP_PLAN, MONITOR_MEMORY nor WEIGHT_STREAMING -- the generation on
which a silently-ignored ``set_flag`` used to go unnoticed.
"""
from __future__ import annotations

import types

import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer import memory_budget
from sparx_agency.tasks.common.trt_optimizer.engine import builder_config as B
from sparx_agency.tasks.common.trt_optimizer.engine.builder_config import (
    _floor_pow2,
    safe_optimization_level,
    BuildOptions, LARGE_BUILD_BYTES, cap_pool, configure, load_timing_cache,
    set_flag)
from sparx_agency.tasks.common.trt_optimizer.target import Target

MIB = 1 << 20
GIB = 1 << 30

#: Measured on tensorrt 11.1.0.106 (``dir(trt.BuilderFlag)``).
_TRT11_FLAGS = (
    "DEBUG", "DIRECT_IO", "DISABLE_COMPILATION_CACHE", "DISABLE_TIMING_CACHE",
    "DISTRIBUTIVE_INDEPENDENCE", "EDITABLE_TIMING_CACHE",
    "ERROR_ON_TIMING_CACHE_MISS", "EXCLUDE_LEAN_RUNTIME", "GPU_FALLBACK",
    "MONITOR_MEMORY", "REFIT", "REFIT_IDENTICAL", "REFIT_INDIVIDUAL",
    "SAFETY_SCOPE", "SPARSE_WEIGHTS", "STRICT_NANS", "STRIP_PLAN", "TF32",
    "VERSION_COMPATIBLE", "WEIGHT_STREAMING",
)

#: A TensorRT 8-era flag set: weak typing, and none of the newer knobs.
_TRT8_FLAGS = ("DEBUG", "FP16", "GPU_FALLBACK", "INT8", "REFIT",
               "SPARSE_WEIGHTS", "STRICT_TYPES", "TF32")

#: Both pools default to this on the 8 GB card -- ~100% of the device.
_TRT11_DEFAULT_POOL = memory_budget.MEASURED_TRT11_DEFAULT_POOL_BYTES


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

def _enum(kind, names):
    """A stand-in for a pybind enum: each member is a unique readable marker."""
    return types.SimpleNamespace(**dict(
        (name, "%s.%s" % (kind, name)) for name in names))


def fake_trt(flags=_TRT11_FLAGS, pools=("WORKSPACE", "TACTIC_DRAM"),
             verbosities=("DETAILED", "LAYER_NAMES_ONLY", "NONE"),
             compat=("NONE", "AMPERE_PLUS", "SAME_COMPUTE_CAPABILITY")):
    """A ``tensorrt`` module stand-in exposing exactly what the module touches.

    An empty ``verbosities``/``compat`` omits the enum entirely, standing in for
    a TensorRT generation that does not have it at all.
    """
    ns = types.SimpleNamespace(BuilderFlag=_enum("BuilderFlag", flags),
                               MemoryPoolType=_enum("MemoryPoolType", pools))
    if verbosities:
        ns.ProfilingVerbosity = _enum("ProfilingVerbosity", verbosities)
    if compat:
        ns.HardwareCompatibilityLevel = _enum("HardwareCompatibilityLevel",
                                              compat)
    return ns


class _FakeCache(object):
    """What ``create_timing_cache`` hands back."""

    def __init__(self, data):
        self.data = bytes(data)

    def serialize(self):
        return self.data


class _FakeConfig(object):
    """An ``IBuilderConfig`` that records every call and every assignment.

    ``unsettable`` names raise ``AttributeError`` on assignment, standing in for
    a TensorRT whose builder config has no such property.
    """

    def __init__(self, unsettable=(), timing_cache_accepted=True):
        object.__setattr__(self, "unsettable", frozenset(unsettable))
        object.__setattr__(self, "timing_cache_accepted", timing_cache_accepted)
        object.__setattr__(self, "assigned", {})
        object.__setattr__(self, "flags_set", [])
        object.__setattr__(self, "flags_cleared", [])
        object.__setattr__(self, "pool_limits", [])
        object.__setattr__(self, "created_caches", [])
        object.__setattr__(self, "attached_caches", [])

    def __setattr__(self, name, value):
        if name in self.unsettable:
            raise AttributeError(name)
        self.assigned[name] = value
        object.__setattr__(self, name, value)

    def set_flag(self, flag):
        self.flags_set.append(flag)

    def clear_flag(self, flag):
        self.flags_cleared.append(flag)

    def set_memory_pool_limit(self, pool, limit):
        self.pool_limits.append((pool, limit))

    def create_timing_cache(self, data):
        cache = _FakeCache(data)
        self.created_caches.append(cache)
        return cache

    def set_timing_cache(self, cache, ignore_mismatch):
        self.attached_caches.append((cache, ignore_mismatch))
        return self.timing_cache_accepted

    @property
    def pools(self):
        """Pool caps as ``{'WORKSPACE': bytes, ...}``."""
        return dict((name.split(".")[-1], limit)
                    for name, limit in self.pool_limits)


class _ReadbackConfig(_FakeConfig):
    """A config that answers ``get_memory_pool_limit``, like TensorRT 11.1.

    Pools named in ``power_of_two_only`` silently ignore any size that is not a
    power of two -- the measured TACTIC_DRAM behaviour ``cap_pool`` exists to
    survive. Pools in ``rejects`` ignore every size, which is the same failure
    with no way out.
    """

    def __init__(self, default_bytes=_TRT11_DEFAULT_POOL,
                 power_of_two_only=(), rejects=(), **kwargs):
        _FakeConfig.__init__(self, **kwargs)
        object.__setattr__(self, "current", {})
        object.__setattr__(self, "default_bytes", int(default_bytes))
        object.__setattr__(self, "power_of_two_only",
                           frozenset(power_of_two_only))
        object.__setattr__(self, "rejects", frozenset(rejects))

    def set_memory_pool_limit(self, pool, limit):
        _FakeConfig.set_memory_pool_limit(self, pool, limit)
        name = pool.split(".")[-1]
        if name in self.rejects:
            return
        if name in self.power_of_two_only and (int(limit) & (int(limit) - 1)):
            return                      # logged as an API Usage Error, not raised
        self.current[pool] = int(limit)

    def get_memory_pool_limit(self, pool):
        return self.current.get(pool, self.default_bytes)


def rtx5070():
    """The dev workstation: RTX 5070 Laptop, 8151 MiB."""
    return HardwareProfile(
        arch="x86_64", is_jetson=False,
        gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        compute_capability=(12, 0), total_mem_bytes=8151 * MIB,
        recommended_workspace_bytes=4 * GIB, target_tag="nvidiageforcert_sm120")


def orin_15w():
    """A 16 GB AGX Orin at 15 W: unified memory, the tightest pool caps."""
    return HardwareProfile(
        arch="aarch64", is_jetson=True, gpu_name="NVIDIA Jetson AGX Orin",
        jetson_model="NVIDIA Jetson AGX Orin", nvpmodel_name="MODE_15W",
        power_budget_w=15, compute_capability=(8, 7), total_mem_bytes=16 * GIB,
        dla_cores=2, allow_dla=True, target_tag="orin_sm87")


def make_target(hardware=None, trt_version="11.1.0.106"):
    return Target(hardware=hardware or rtx5070(), trt_version=trt_version,
                  strongly_typed=True)


def run_configure(options=None, trt=None, config=None, weight_bytes=0,
                  target=None):
    """Call ``configure`` and return ``(config, notes)``."""
    config = config if config is not None else _FakeConfig()
    notes = []
    configure(config, trt or fake_trt(), target or make_target(),
              options or BuildOptions(), weight_bytes, notes)
    return config, notes


# --------------------------------------------------------------------------
# set_flag -- the honest report of what TensorRT 11 removed
# --------------------------------------------------------------------------

def test_set_flag_returns_false_for_a_flag_this_tensorrt_lacks():
    """FP16 is genuinely gone on 11; the caller must be told, not fooled."""
    config = _FakeConfig()
    assert set_flag(config, fake_trt(), "FP16") is False
    assert config.flags_set == []
    assert config.flags_cleared == []


def test_set_flag_returns_true_and_sets_a_flag_that_exists():
    config = _FakeConfig()
    assert set_flag(config, fake_trt(), "MONITOR_MEMORY") is True
    assert config.flags_set == ["BuilderFlag.MONITOR_MEMORY"]


def test_set_flag_clears_instead_of_setting_when_on_is_false():
    config = _FakeConfig()
    assert set_flag(config, fake_trt(), "TF32", on=False) is True
    assert config.flags_cleared == ["BuilderFlag.TF32"]
    assert config.flags_set == []


def test_set_flag_finds_the_same_flag_on_an_older_generation():
    config = _FakeConfig()
    assert set_flag(config, fake_trt(flags=_TRT8_FLAGS), "FP16") is True
    assert config.flags_set == ["BuilderFlag.FP16"]


def test_set_flag_does_not_mistake_the_zero_valued_member_for_absence():
    """The first enum member is falsy; the probe must be ``is None``."""
    trt = types.SimpleNamespace(BuilderFlag=types.SimpleNamespace(DEBUG=0))
    config = _FakeConfig()
    assert set_flag(config, trt, "DEBUG") is True
    assert config.flags_set == [0]


# --------------------------------------------------------------------------
# the central claim: BOTH memory pools are capped
# --------------------------------------------------------------------------

def test_configure_caps_both_workspace_and_tactic_dram():
    """Capping WORKSPACE alone does not bound the peak -- TACTIC_DRAM spikes."""
    target = make_target()
    config, _ = run_configure(target=target)
    expected = dict(memory_budget.builder_pool_limits(target.hardware))
    # TACTIC_DRAM is rounded down to a power of two BEFORE it is requested:
    # TensorRT accepts nothing else for that pool and only logs the rejection.
    expected["TACTIC_DRAM"] = _floor_pow2(expected["TACTIC_DRAM"])
    assert len(config.pool_limits) == 2
    assert config.pools == expected
    assert set(config.pools) == {"WORKSPACE", "TACTIC_DRAM"}


def test_both_pool_caps_are_below_the_measured_default_and_fit_together():
    """Both pools default to ~100% of an 8 GB card; that is what is fixed."""
    target = make_target()
    config, _ = run_configure(target=target)
    for limit in config.pools.values():
        assert limit < memory_budget.MEASURED_TRT11_DEFAULT_POOL_BYTES
    assert sum(config.pools.values()) < target.hardware.total_mem_bytes


def test_pool_caps_are_integers_the_binding_will_accept():
    config, _ = run_configure()
    for _, limit in config.pool_limits:
        assert isinstance(limit, int)


def test_jetson_pool_caps_are_tighter_than_the_workstation_ones():
    """Unified memory: every byte capped here is a byte ROS and the mapper get."""
    laptop, _ = run_configure(target=make_target())
    jetson, _ = run_configure(target=make_target(hardware=orin_15w()))
    assert jetson.pools["WORKSPACE"] < laptop.pools["WORKSPACE"]
    assert jetson.pools["TACTIC_DRAM"] < laptop.pools["TACTIC_DRAM"]


def test_configure_notes_a_memory_pool_this_tensorrt_does_not_have():
    trt = fake_trt(pools=("WORKSPACE",))
    config, notes = run_configure(trt=trt)
    assert list(config.pools) == ["WORKSPACE"]
    assert any("MemoryPoolType.TACTIC_DRAM absent" in note for note in notes)
    assert any("11.1.0.106" in note for note in notes)


def test_configure_raises_when_the_profile_has_no_memory_to_size_pools_from():
    """A zero cap would fail the build far from here, so it fails here."""
    blind = HardwareProfile(arch="x86_64", is_jetson=False, total_mem_bytes=0)
    with pytest.raises(ValueError):
        run_configure(target=make_target(hardware=blind))


# --------------------------------------------------------------------------
# cap_pool -- a rejected pool size is logged by TensorRT, never raised
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (0, 0), (-1, 0), (1, 1), (2, 2), (3, 2), (2 * GIB, 2 * GIB),
    (4273471488, 2 * GIB),                       # the 3.98 GiB TACTIC_DRAM cap
])
def test_floor_pow2(value, expected):
    assert B._floor_pow2(value) == expected


def test_cap_pool_notes_that_it_could_not_read_the_cap_back():
    """An older binding has no getter; the note must not claim it was verified."""
    config = _FakeConfig()
    notes = []
    applied = cap_pool(config, "MemoryPoolType.WORKSPACE", "WORKSPACE",
                       2 * GIB, notes)
    assert applied == 2 * GIB
    assert "not read back" in notes[0]


def test_cap_pool_reports_the_size_the_pool_accepted():
    config = _ReadbackConfig()
    notes = []
    applied = cap_pool(config, "MemoryPoolType.WORKSPACE", "WORKSPACE",
                       2 * GIB, notes)
    assert applied == 2 * GIB
    assert notes == ["WORKSPACE capped to 2.00 GiB"]


def test_tactic_dram_is_requested_as_a_power_of_two_first_time():
    """TensorRT accepts only a power-of-two TACTIC_DRAM and does not raise on
    anything else -- it logs an API-usage error and leaves the pool at ~100% of
    the device. So the cap is rounded before it is asked for: one call, no error
    in the build log, and the pool actually bounded."""
    config = _ReadbackConfig(power_of_two_only=("TACTIC_DRAM",))
    target = make_target()
    _, notes = run_configure(config=config, target=target)
    requested = memory_budget.builder_pool_limits(target.hardware)["TACTIC_DRAM"]

    assert config.current["MemoryPoolType.TACTIC_DRAM"] == 2 * GIB
    assert [limit for name, limit in config.pool_limits
            if name.endswith("TACTIC_DRAM")] == [2 * GIB]
    assert any("rounded" in note and "power of two" in note for note in notes)
    assert requested != 2 * GIB       # the raw budget really was not a power of two



def test_a_pool_that_refuses_every_size_is_a_hard_failure():
    """Silently uncapped, this pool takes the build host down with it."""
    config = _ReadbackConfig(rejects=("TACTIC_DRAM",))
    with pytest.raises(RuntimeError) as excinfo:
        run_configure(config=config)
    message = str(excinfo.value)
    assert "TACTIC_DRAM" in message
    assert "power of two" in message


def test_an_accepted_cap_is_not_retried():
    config = _ReadbackConfig()
    notes = []
    cap_pool(config, "MemoryPoolType.WORKSPACE", "WORKSPACE", 2 * GIB, notes)
    assert len(config.pool_limits) == 1


# --------------------------------------------------------------------------
# the explicitly-assigned knobs
# --------------------------------------------------------------------------

def test_builder_optimization_level_and_max_aux_streams_are_assigned():
    # fp32, so the Blackwell non-FP32 clamp does not apply and the requested
    # level reaches the config untouched.
    config, _ = run_configure(BuildOptions(precision="fp32",
                                           optimization_level=5,
                                           max_aux_streams=0))
    assert config.assigned["builder_optimization_level"] == 5
    assert config.assigned["max_aux_streams"] == 0


def test_optimization_level_is_coerced_to_an_int():
    config, _ = run_configure(BuildOptions(optimization_level="3"))
    # Clamped from 3 to 1: this fixture is the Blackwell + TRT 11 target
    # where a non-FP32 build above level 1 segfaults. See the clamp tests.
    assert config.assigned["builder_optimization_level"] == 1
    assert isinstance(config.assigned["builder_optimization_level"], int)


def test_default_max_aux_streams_is_zero_for_a_single_model_loop():
    config, _ = run_configure()
    assert config.assigned["max_aux_streams"] == 0


def test_configure_notes_when_max_aux_streams_is_not_settable():
    config = _FakeConfig(unsettable=("max_aux_streams",))
    config, notes = run_configure(config=config)
    assert "max_aux_streams" not in config.assigned
    assert any("max_aux_streams not settable" in note for note in notes)
    # Clamped from 3 to 1: this fixture is the Blackwell + TRT 11 target
    # where a non-FP32 build above level 1 segfaults. See the clamp tests.
    assert config.assigned["builder_optimization_level"] == 1


# --------------------------------------------------------------------------
# TF32
# --------------------------------------------------------------------------

def test_tf32_is_left_alone_by_default():
    config, notes = run_configure(BuildOptions(tf32=True))
    assert config.flags_cleared == []
    assert not any("TF32" in note for note in notes)


def test_tf32_is_cleared_and_noted_when_disabled():
    config, notes = run_configure(BuildOptions(tf32=False))
    assert config.flags_cleared == ["BuilderFlag.TF32"]
    tf32_notes = [note for note in notes if "TF32 cleared" in note]
    assert len(tf32_notes) == 1
    assert "timing cache" in tf32_notes[0]


def test_tf32_clearing_is_silent_on_a_tensorrt_without_the_flag():
    trt = fake_trt(flags=("DEBUG", "REFIT"))
    config, notes = run_configure(BuildOptions(tf32=False), trt=trt)
    assert config.flags_cleared == []
    assert not any("TF32" in note for note in notes)


# --------------------------------------------------------------------------
# MONITOR_MEMORY
# --------------------------------------------------------------------------

def test_monitor_memory_is_auto_enabled_past_the_large_build_threshold():
    config, notes = run_configure(weight_bytes=LARGE_BUILD_BYTES + 1)
    assert "BuilderFlag.MONITOR_MEMORY" in config.flags_set
    assert any("MONITOR_MEMORY on" in note for note in notes)


def test_monitor_memory_is_auto_enabled_exactly_at_the_threshold():
    config, _ = run_configure(weight_bytes=LARGE_BUILD_BYTES)
    assert "BuilderFlag.MONITOR_MEMORY" in config.flags_set


def test_monitor_memory_is_not_auto_enabled_below_the_threshold():
    config, notes = run_configure(weight_bytes=LARGE_BUILD_BYTES - 1)
    assert "BuilderFlag.MONITOR_MEMORY" not in config.flags_set
    assert not any("MONITOR_MEMORY" in note for note in notes)


def test_monitor_memory_can_be_forced_on_for_a_small_build():
    config, _ = run_configure(BuildOptions(monitor_memory=True), weight_bytes=0)
    assert "BuilderFlag.MONITOR_MEMORY" in config.flags_set


def test_monitor_memory_can_be_forced_off_for_a_large_build():
    config, _ = run_configure(BuildOptions(monitor_memory=False),
                              weight_bytes=8 * GIB)
    assert "BuilderFlag.MONITOR_MEMORY" not in config.flags_set


def test_monitor_memory_is_skipped_without_the_flag_and_claims_nothing():
    trt = fake_trt(flags=_TRT8_FLAGS)
    config, notes = run_configure(trt=trt, weight_bytes=8 * GIB)
    assert config.flags_set == []
    assert not any("MONITOR_MEMORY" in note for note in notes)


# --------------------------------------------------------------------------
# STRIP_PLAN / REFIT
# --------------------------------------------------------------------------

def test_strip_plan_without_refit_raises():
    """A stripped non-refittable engine can never have its weights put back."""
    with pytest.raises(ValueError) as excinfo:
        run_configure(BuildOptions(strip_plan=True, refit=False))
    assert "refit" in str(excinfo.value)


def test_strip_plan_with_refit_sets_both_flags():
    config, notes = run_configure(BuildOptions(strip_plan=True, refit=True))
    assert config.flags_set == ["BuilderFlag.REFIT", "BuilderFlag.STRIP_PLAN"]
    assert any("STRIP_PLAN + REFIT" in note for note in notes)
    assert any("OnnxParserRefitter" in note for note in notes)


def test_refit_alone_sets_only_refit():
    config, notes = run_configure(BuildOptions(refit=True))
    assert config.flags_set == ["BuilderFlag.REFIT"]
    assert not any("STRIP_PLAN" in note for note in notes)


def test_strip_plan_raises_when_this_tensorrt_has_no_such_flag():
    """Silently ignoring it would ship a full-weight engine claiming a stripped
    plan in its sidecar -- exactly the lie set_flag exists to prevent."""
    trt = fake_trt(flags=_TRT8_FLAGS)          # REFIT yes, STRIP_PLAN no
    with pytest.raises(RuntimeError) as excinfo:
        run_configure(BuildOptions(strip_plan=True, refit=True), trt=trt)
    assert "STRIP_PLAN" in str(excinfo.value)


def test_refit_raises_when_this_tensorrt_has_no_such_flag():
    trt = fake_trt(flags=("DEBUG", "TF32"))
    with pytest.raises(RuntimeError) as excinfo:
        run_configure(BuildOptions(refit=True), trt=trt)
    assert "REFIT" in str(excinfo.value)


# --------------------------------------------------------------------------
# the remaining opt-in flags
# --------------------------------------------------------------------------

def test_weight_streaming_is_off_unless_asked_for():
    config, _ = run_configure()
    assert "BuilderFlag.WEIGHT_STREAMING" not in config.flags_set


def test_weight_streaming_sets_the_flag_and_states_the_runtime_budget():
    config, notes = run_configure(BuildOptions(weight_streaming=True))
    assert "BuilderFlag.WEIGHT_STREAMING" in config.flags_set
    assert any("streamable_weights_size" in note for note in notes)


def test_sparse_weights_sets_the_flag_only_when_asked():
    on, notes = run_configure(BuildOptions(sparse_weights=True))
    off, _ = run_configure()
    assert "BuilderFlag.SPARSE_WEIGHTS" in on.flags_set
    assert "BuilderFlag.SPARSE_WEIGHTS" not in off.flags_set
    assert any("SPARSE_WEIGHTS on" in note for note in notes)


# --------------------------------------------------------------------------
# profiling verbosity and hardware compatibility
# --------------------------------------------------------------------------

def test_detailed_profiling_verbosity_is_set_by_default():
    """Layer names only makes a six-month-old regression undiagnosable."""
    config, _ = run_configure()
    assert config.assigned["profiling_verbosity"] == "ProfilingVerbosity.DETAILED"


def test_profiling_verbosity_is_left_alone_when_detailed_profiling_is_off():
    config, _ = run_configure(BuildOptions(detailed_profiling=False))
    assert "profiling_verbosity" not in config.assigned


def test_profiling_verbosity_is_skipped_when_detailed_is_unavailable():
    trt = fake_trt(verbosities=("LAYER_NAMES_ONLY", "NONE"))
    config, _ = run_configure(trt=trt)
    assert "profiling_verbosity" not in config.assigned


def test_hardware_compatibility_level_is_pinned_to_none():
    """A device-pinned robot never trades tactics for portability."""
    config, _ = run_configure()
    assert config.assigned["hardware_compatibility_level"] == \
        "HardwareCompatibilityLevel.NONE"


def test_hardware_compatibility_is_skipped_where_the_enum_does_not_exist():
    config, _ = run_configure(trt=fake_trt(compat=()))
    assert "hardware_compatibility_level" not in config.assigned


# --------------------------------------------------------------------------
# the notes are the build's own record
# --------------------------------------------------------------------------

def test_notes_are_non_empty_and_name_both_pool_caps():
    _, notes = run_configure()
    assert notes
    assert any("WORKSPACE capped to" in note for note in notes)
    assert any("TACTIC_DRAM capped to" in note for note in notes)


def test_pool_cap_notes_report_gibibytes():
    target = make_target()
    _, notes = run_configure(target=target)
    expected = memory_budget.builder_pool_limits(target.hardware)["WORKSPACE"]
    head = "WORKSPACE capped to %.2f GiB" % (expected / float(GIB))
    assert any(note.startswith(head) for note in notes)


def test_configure_appends_to_the_callers_notes_rather_than_replacing_them():
    config = _FakeConfig()
    notes = ["precision baked into the ONNX (fp16)"]
    configure(config, fake_trt(), make_target(), BuildOptions(), 0, notes)
    assert notes[0] == "precision baked into the ONNX (fp16)"
    assert len(notes) > 1


# --------------------------------------------------------------------------
# the persisted timing cache
# --------------------------------------------------------------------------

def test_timing_cache_is_created_when_the_file_is_absent(tmp_path):
    config = _FakeConfig()
    notes = []
    cache = load_timing_cache(config, fake_trt(),
                              tmp_path / "timing_orin_sm87.cache", notes)
    assert cache.data == b""
    assert config.attached_caches == [(cache, False)]
    assert notes == ["timing cache created (0 bytes in)"]


def test_timing_cache_is_reused_when_the_file_is_present(tmp_path):
    path = tmp_path / "timing_orin_sm87.cache"
    path.write_bytes(b"\x01\x02\x03\x04")
    config = _FakeConfig()
    notes = []
    cache = load_timing_cache(config, fake_trt(), path, notes)
    assert cache.data == b"\x01\x02\x03\x04"
    assert notes == ["timing cache reused (4 bytes in)"]


def test_timing_cache_is_attached_with_strict_verification(tmp_path):
    """ignore_mismatch=False: a cache from another device must not be trusted."""
    config = _FakeConfig()
    load_timing_cache(config, fake_trt(), tmp_path / "t.cache", [])
    assert config.attached_caches[0][1] is False


def test_timing_cache_accepts_a_string_path(tmp_path):
    path = tmp_path / "t.cache"
    path.write_bytes(b"abc")
    config = _FakeConfig()
    notes = []
    cache = load_timing_cache(config, fake_trt(), str(path), notes)
    assert cache.data == b"abc"
    assert "reused" in notes[0]


def test_a_rejected_timing_cache_is_not_reported_as_reused(tmp_path):
    """TensorRT returns False for a cache built on another device/version. The
    build may proceed on a fresh cache, but the note must not claim the stale
    one was used -- that note is written verbatim into the engine sidecar."""
    path = tmp_path / "t.cache"
    path.write_bytes(b"stale-cache-from-another-gpu")
    config = _FakeConfig(timing_cache_accepted=False)
    notes = []
    cache = load_timing_cache(config, fake_trt(), path, notes)
    assert cache.data == b""
    assert not any("reused" in note for note in notes)
    assert any("REJECTED" in note for note in notes)


# --------------------------------------------------------------------------
# The Blackwell optimization-level clamp
# --------------------------------------------------------------------------

def test_level_is_clamped_on_blackwell_trt11_for_a_non_fp32_build():
    """Measured: TensorRT 11.1 segfaults building FP16 above level 1 on sm_120."""
    level, note = safe_optimization_level(make_target(), "fp16", 3)
    assert level == 1
    assert "segfault" in note
    assert "3 -> 1" in note


def test_level_is_not_clamped_for_an_fp32_build():
    level, note = safe_optimization_level(make_target(), "fp32", 3)
    assert (level, note) == (3, "")


def test_level_is_not_clamped_below_the_threshold():
    level, note = safe_optimization_level(make_target(), "fp16", 1)
    assert (level, note) == (1, "")


def test_level_is_not_clamped_on_a_tensorrt_10_target():
    target = make_target(trt_version="10.3.0.26")
    level, note = safe_optimization_level(target, "fp16", 3)
    assert (level, note) == (3, "")


def test_level_is_not_clamped_on_orin():
    target = make_target(hardware=orin_15w())
    level, note = safe_optimization_level(target, "fp16", 3)
    assert (level, note) == (3, "")


def test_clamp_note_reaches_the_build_record():
    _, notes = run_configure(BuildOptions(precision="fp16",
                                          optimization_level=3))
    assert any("builder_optimization_level lowered" in note for note in notes)
