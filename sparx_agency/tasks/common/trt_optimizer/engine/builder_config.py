"""Every TensorRT builder knob this pipeline sets, and why it sets it.

Separated from the build orchestration because *what to set* is a policy that
changes with the target and the model, while *how to build* does not. Keeping
them apart means the policy can be reviewed on its own -- which matters, since
several TensorRT defaults are actively wrong here:

* **Both memory pools default to ~100% of the device.** Measured on an 8 GB
  card: WORKSPACE *and* TACTIC_DRAM each defaulted to 8,080,064,512 bytes.
  Capping WORKSPACE alone -- which is all the repo's older builders do -- does
  not bound the peak, because TACTIC_DRAM is what spikes during tactic
  evaluation. On a Jetson, TACTIC_DRAM defaults to 75% of *unified* memory,
  which is memory the mapper and ROS also need.
* **TF32 is on by default** and is the only flag set in a fresh config. Fine for
  a network; wrong for an FP32 sub-graph whose numerics must be reproducible.
* **``max_aux_streams`` defaults to a heuristic** that costs runtime activation
  memory which cannot be reused. A single-model real-time loop has no
  parallelism to exploit, so 0 is right.
* **Profiling verbosity defaults to layer names only**, so a regression six
  months from now cannot be diagnosed without a rebuild.

Flags are set through :func:`set_flag`, which reports whether the flag existed
at all. That matters because TensorRT 11 removed a great many of them, and a
silently ignored ``set_flag`` is how a build ends up FP32 while claiming FP16.
Pool caps go through :func:`cap_pool` for the same reason: a rejected pool size
is logged, not raised, so the cap is read back rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sparx_agency.tasks.common.trt_optimizer import memory_budget

#: Highest builder_optimization_level that is safe for a non-FP32 build on
#: Blackwell with TensorRT 11.x. See :func:`safe_optimization_level`.
_BLACKWELL_SAFE_LEVEL = 1

#: Pools TensorRT requires to be an exact power of two.
POWER_OF_TWO_POOLS = ("TACTIC_DRAM", "TACTIC_SHARED_MEMORY",
                      "DLA_MANAGED_SRAM", "DLA_LOCAL_DRAM",
                      "DLA_GLOBAL_DRAM")

#: Weight bytes above which memory monitoring is worth its overhead.
LARGE_BUILD_BYTES = 1 << 30


@dataclass
class BuildOptions:
    """Explicit TensorRT builder configuration for one engine.

    Args:
        precision: target precision name (``fp32``/``fp16``/``int8``/...).
        optimization_level: builder search effort. This is *offline build
            effort*, not a runtime power knob -- ``nvpmodel`` clamps runtime
            power regardless of which tactics the engine picked. Use 1 while
            iterating and 3-5 for the engine you actually fly.
        max_aux_streams: 0 for a single-model real-time loop.
        tf32: leave TF32 enabled (the default) or clear it for reproducible
            FP32 numerics. Clearing it invalidates the timing cache entry.
        monitor_memory: set ``BuilderFlag.MONITOR_MEMORY`` so the build log says
            where the peak was. Auto-enabled for large builds.
        weight_streaming: stream weights from host memory. Only worth it when
            weights exceed roughly half the free VRAM.
        strip_plan: build a weightless plan. Must be paired with ``refit`` --
            STRIP_PLAN alone leaves the engine non-refittable, so the weights
            can never be put back.
        refit: build a refittable engine.
        sparse_weights: enable structured-sparsity tactics. Only pays on weights
            that were actually pruned to the 2:4 pattern.
        detailed_profiling: record DETAILED layer information and dump the
            engine inspector JSON beside the plan.
        timing_cache: path to a persisted timing cache.
    """

    precision: str = "fp16"
    optimization_level: int = 3
    max_aux_streams: int = 0
    tf32: bool = True
    monitor_memory: Optional[bool] = None
    weight_streaming: bool = False
    strip_plan: bool = False
    refit: bool = False
    sparse_weights: bool = False
    detailed_profiling: bool = True
    timing_cache: Optional[str] = None


def safe_optimization_level(target, precision, requested):
    """Clamp the builder search effort away from a combination that crashes.

    Measured on this machine (RTX 5070 Laptop, sm_120, TensorRT 11.1.0.106):
    ``build_serialized_network`` **segfaults** intermittently on an FP16
    transformer graph at ``builder_optimization_level >= 2`` -- level 3 failed
    roughly one run in three, level 2 occasionally, levels 0 and 1 never, and
    the same graph at FP32 built reliably at the default level. Reproduced with
    a bare ``IBuilderConfig`` carrying none of this module's knobs, so it is a
    TensorRT bug rather than a configuration one.

    A segfault cannot be caught, so there is no retry to fall back on: the
    process simply dies mid-build. The only defence is not to ask for it.

    Args:
        target: the :class:`..target.Target` doing the building.
        precision: the requested precision.
        requested: the optimization level the caller asked for.

    Returns:
        ``(level, note)`` -- the level to use and an explanation, or
        ``(requested, "")`` when the combination is not the known-bad one.
    """
    version = getattr(target, "trt_major_minor", None)
    sm = getattr(getattr(target, "hardware", None), "sm", None)
    known_bad = (sm is not None and sm >= 120
                 and version is not None and version[0] >= 11
                 and precision != "fp32" and requested >= _BLACKWELL_SAFE_LEVEL + 1)
    if not known_bad:
        return requested, ""
    return _BLACKWELL_SAFE_LEVEL, (
        "builder_optimization_level lowered %d -> %d: TensorRT %s on sm_%s "
        "segfaults intermittently while building a non-FP32 graph above level "
        "%d. A segfault cannot be caught, so the level is avoided rather than "
        "retried. Raise it deliberately once TensorRT fixes this."
        % (requested, _BLACKWELL_SAFE_LEVEL, target.trt_version, sm,
           _BLACKWELL_SAFE_LEVEL))


def set_flag(config, trt, name, on=True):
    """Set a BuilderFlag by name if this TensorRT has it; report whether it did."""
    flag = getattr(trt.BuilderFlag, name, None)
    if flag is None:
        return False
    if on:
        config.set_flag(flag)
    else:
        config.clear_flag(flag)
    return True


def _floor_pow2(value):
    """Largest power of two not exceeding ``value``; 0 for a non-positive one."""
    v = int(value)
    if v <= 0:
        return 0
    return 1 << (v.bit_length() - 1)


def cap_pool(config, pool, pool_name, limit, notes):
    """Cap one memory pool and record the size TensorRT actually accepted.

    ``set_memory_pool_limit`` does not raise on a rejected size: it logs an
    ``API Usage Error`` through the builder's logger and leaves the pool
    where it was. Measured on TensorRT 11.1, ``TACTIC_DRAM`` accepts only a
    **power-of-two** size, so the 3.98 GiB cap
    :func:`..memory_budget.builder_pool_limits` derives from an 8 GiB card is
    dropped on the floor and the pool stays at its ~100%-of-device default --
    which is precisely the failure that function exists to prevent. The value
    is therefore read back, retried at the next power of two down, and the
    note records what was *applied* rather than what was asked for.

    Args:
        config: the ``IBuilderConfig`` being configured.
        pool: the ``MemoryPoolType`` member to cap.
        pool_name: its name, for the note.
        limit: requested cap in bytes.
        notes: build-note list, appended to in place.

    Returns:
        int: the cap TensorRT is holding, or the requested value when this
        TensorRT exposes no ``get_memory_pool_limit`` to read it back with.

    Raises:
        RuntimeError: if the pool is still above the requested cap after the
            power-of-two retry. A silently uncapped pool takes the build host
            down with it, so it fails here instead.
    """
    requested = int(limit)
    if pool_name in POWER_OF_TWO_POOLS:
        # Round BEFORE asking. TensorRT validates these pools against a power
        # of two and only LOGS the rejection, so asking for a non-power-of-two
        # emits an alarming API-usage error on every single build while
        # silently leaving the pool at its ~100%-of-device default.
        rounded = _floor_pow2(requested)
        if rounded and rounded != requested:
            notes.append("%s request rounded %.2f -> %.2f GiB (TensorRT accepts "
                         "only a power of two for this pool)"
                         % (pool_name, requested / (1 << 30),
                            rounded / (1 << 30)))
            requested = rounded
    config.set_memory_pool_limit(pool, requested)
    readback = getattr(config, "get_memory_pool_limit", None)
    if readback is None:
        notes.append("%s capped to %.2f GiB (not read back: this TensorRT has "
                     "no get_memory_pool_limit)"
                     % (pool_name, requested / (1 << 30)))
        return requested
    applied = int(readback(pool))
    if applied == requested:
        notes.append("%s capped to %.2f GiB" % (pool_name, applied / (1 << 30)))
        return applied
    fallback = _floor_pow2(requested)
    if fallback and fallback != requested:
        config.set_memory_pool_limit(pool, fallback)
        applied = int(readback(pool))
    if applied > requested:
        raise RuntimeError(
            "%s would not take a %d-byte cap and is still at %d bytes; "
            "TensorRT logged the rejection instead of raising, so the pool is "
            "left at its ~100%%-of-device default and this build can exhaust "
            "the host. Pass a size this TensorRT accepts (TACTIC_DRAM wants a "
            "power of two)." % (pool_name, requested, applied))
    notes.append("%s capped to %.2f GiB (asked %.2f GiB; TensorRT rounded the "
                 "request down to a power of two)"
                 % (pool_name, applied / (1 << 30), requested / (1 << 30)))
    return applied


def configure(config, trt, target, options, weight_bytes, notes):
    """Apply every builder knob explicitly, recording what was and was not set."""
    pools = memory_budget.builder_pool_limits(target.hardware)
    for pool_name, limit in pools.items():
        pool = getattr(trt.MemoryPoolType, pool_name, None)
        if pool is None:
            notes.append("MemoryPoolType.%s absent on TensorRT %s"
                         % (pool_name, target.trt_version))
            continue
        cap_pool(config, pool, pool_name, limit, notes)

    level, level_note = safe_optimization_level(target, options.precision,
                                               int(options.optimization_level))
    config.builder_optimization_level = level
    if level_note:
        notes.append(level_note)
    try:
        config.max_aux_streams = int(options.max_aux_streams)
    except AttributeError:
        notes.append("max_aux_streams not settable on this TensorRT")

    if not options.tf32:
        if set_flag(config, trt, "TF32", on=False):
            notes.append("TF32 cleared for reproducible FP32 numerics "
                         "(this invalidates the timing cache entry)")

    monitor = options.monitor_memory
    if monitor is None:
        monitor = weight_bytes >= LARGE_BUILD_BYTES
    if monitor and set_flag(config, trt, "MONITOR_MEMORY"):
        notes.append("MONITOR_MEMORY on: the build log reports its peak")

    if options.weight_streaming and set_flag(config, trt, "WEIGHT_STREAMING"):
        notes.append("WEIGHT_STREAMING on: set the budget explicitly at runtime, "
                     "min(free//2, streamable_weights_size//2)")
    if options.sparse_weights and set_flag(config, trt, "SPARSE_WEIGHTS"):
        notes.append("SPARSE_WEIGHTS on")

    if options.strip_plan and not options.refit:
        raise ValueError(
            "strip_plan without refit produces an engine reporting "
            "refittable=False, so its weights can never be restored. Set both.")
    if options.refit and not set_flag(config, trt, "REFIT"):
        raise RuntimeError(
            "BuilderFlag.REFIT is absent on TensorRT %s, so a refittable "
            "engine cannot be built here; the build would produce a normal "
            "engine instead. Drop refit (and strip_plan with it) or build on a "
            "TensorRT that has the flag." % target.trt_version)
    if options.strip_plan:
        if not set_flag(config, trt, "STRIP_PLAN"):
            raise RuntimeError(
                "BuilderFlag.STRIP_PLAN is absent on TensorRT %s. Ignoring it "
                "would ship a full-weight engine whose sidecar claims a "
                "weightless plan, so the build stops here." % target.trt_version)
        notes.append("STRIP_PLAN + REFIT: weightless plan, refill at load with "
                     "trt.OnnxParserRefitter")

    if options.detailed_profiling:
        verbosity = getattr(trt.ProfilingVerbosity, "DETAILED", None)
        if verbosity is not None:
            config.profiling_verbosity = verbosity

    # Device-pinned robot: never trade tactics for portability. AMPERE_PLUS caps
    # shared memory to 48 KiB and disables cuDNN/cuBLAS tactic sources.
    none_level = getattr(getattr(trt, "HardwareCompatibilityLevel", None),
                         "NONE", None)
    if none_level is not None:
        config.hardware_compatibility_level = none_level


def load_timing_cache(config, trt, path, notes):
    """Load (or start) a persisted timing cache and attach it.

    ``set_timing_cache`` *answers* whether the cache was taken: with
    ``ignore_mismatch=False`` TensorRT refuses one recorded under different
    CUDA device properties -- a driver bump, a different GPU, a JetPack point
    release -- and returns False instead of raising. Ignoring that answer
    leaves the build on TensorRT's own empty cache while the note claims the
    persisted one was reused, and that note is copied verbatim into the engine
    sidecar. So the answer is read, an empty cache is attached in its place,
    and the note says which of the two happened.

    Args:
        config: the ``IBuilderConfig`` being configured.
        trt: the ``tensorrt`` module (unused today; kept for symmetry with the
            other helpers here, all of which take the generation they target).
        path: the persisted cache file. It need not exist.
        notes: build-note list, appended to in place.

    Returns:
        The attached timing cache, whose ``serialize()`` the caller writes back
        to ``path`` once the build is done.
    """
    existing = Path(path).read_bytes() if Path(path).exists() else b""
    cache = config.create_timing_cache(existing)
    if config.set_timing_cache(cache, ignore_mismatch=False) is False:
        cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)
        notes.append(
            "persisted timing cache %s was REJECTED by TensorRT (recorded "
            "under different CUDA device properties); started an empty one, so "
            "this build gets no cache hits -- delete the file to silence this"
            % path)
        return cache
    notes.append("timing cache %s (%d bytes in)"
                 % ("reused" if existing else "created", len(existing)))
    return cache


