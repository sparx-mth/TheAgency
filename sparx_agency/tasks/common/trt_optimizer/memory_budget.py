"""Will the model *fit*, and what to do when it will not.

Latency is the optimizer's first constraint; residency is the second, and it is
the one that turns a good plan into a `cudaErrorMemoryAllocation` five seconds
after takeoff. The driving case is an 8 GB laptop card (8151 MiB total, 7613 MiB
free at idle with a display attached) asked to host a multi-GB dual-system VLA,
and after it a Jetson Orin whose GPU memory is *unified with the CPU* -- every
byte an engine holds is a byte ROS, the mapper and the page cache do not get.

Three decisions in here are not obvious:

  * **Residency is an argument, not an assumption.** ``resident='concurrent'``
    sums every component's weights, because that is what a dual-system model
    actually does: System-2 stays loaded while System-1 runs. ``'sequential'``
    counts only the largest engine and is a *promise* the caller makes -- it is
    legal only if the pipeline genuinely tears an engine down before building
    the next. Getting this wrong is the single most common way a budget lies.
  * **Activations are a heuristic and nothing more.** ``activation_factor``
    multiplies the weight bytes; there is no shape-walk behind it. The ground
    truth is a real build with ``ProfilingVerbosity``/``MONITOR_MEMORY`` and
    ``IExecutionContext`` device memory read back. Treat a fit that depends on
    the activation term to two significant figures as a non-fit.
  * **Capping WORKSPACE alone does nothing.** Measured on TensorRT 11.1: *both*
    the WORKSPACE and the TACTIC_DRAM memory pools default to ~100% of the
    device (8,080,064,512 bytes on this 8 GB card). TACTIC_DRAM is what spikes
    while tactics are evaluated, so a build that only caps WORKSPACE still OOMs
    the machine it is building on. :func:`builder_pool_limits` caps both.

Pure standard library plus dataclasses; no torch, no TensorRT, no CUDA.
Nothing here probes the device: the ``HardwareProfile`` is passed in, so a
plan can be budgeted for the Orin from the workstation and the tests can
target a machine they are not running on.
Python-3.8-compatible syntax, importable on a Jetson's system interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer.spec import Plan

MIB = 1 << 20
GIB = 1 << 30

#: Measured default of BOTH ``MemoryPoolType.WORKSPACE`` and
#: ``MemoryPoolType.TACTIC_DRAM`` on TensorRT 11.1 on an 8 GB RTX 5070 Laptop:
#: ~100% of the device. Kept as a number because the documentation says
#: "device memory" and people read that as "some sensible fraction".
MEASURED_TRT11_DEFAULT_POOL_BYTES = 8080064512

#: Fraction of a discrete GPU's total memory actually free at idle. Measured
#: 7613/8151 MiB on the RTX 5070 Laptop with a desktop session attached.
_DGPU_FREE_FRACTION = 0.93

#: Fraction of a Jetson's unified memory that may be spent on engines. The rest
#: is the OS, the ROS graph and the page cache, which are not optional.
_JETSON_FREE_FRACTION = 0.55

#: CUDA context + kernel images + cuDNN/cuBLAS handles, reserved before a single
#: weight lands. A display-attached dGPU pays for the framebuffer too.
_CTX_DGPU_DISPLAY_BYTES = 600 * MIB
_CTX_DGPU_HEADLESS_BYTES = 300 * MIB
#: Unified memory: the context, the kernel images and the host-side staging all
#: come out of the same DRAM, so reserve well above the dGPU figure.
_CTX_JETSON_BYTES = 768 * MIB

_BYTES_PER_PARAM = {
    "fp32": 4, "float32": 4, "float": 4, "tf32": 4,
    "fp16": 2, "float16": 2, "half": 2, "bf16": 2, "bfloat16": 2,
    "int8": 1, "uint8": 1, "fp8": 1, "float8": 1,
    "int4": 0.5, "nvfp4": 0.5, "fp4": 0.5,
}

#: One step down the precision ladder. ``None`` marks the floor.
_NEXT_PRECISION = {
    "fp32": "fp16", "float32": "fp16", "float": "fp16", "tf32": "fp16",
    "fp16": "int8", "float16": "int8", "half": "int8",
    "bf16": "int8", "bfloat16": "int8",
    "int8": "int4", "uint8": "int4", "fp8": "nvfp4", "float8": "nvfp4",
    "int4": None, "nvfp4": None, "fp4": None,
}

_RESIDENCY_MODES = ("concurrent", "sequential")


@dataclass
class Budget:
    """One device-memory budget: what is there, and what the plan will take.

    Every field is a byte count. ``total_bytes``/``free_bytes`` describe the
    device; the remaining five are the claim the plan makes on it and are summed
    by :attr:`required_bytes`.

    Args:
        total_bytes: physical device memory (unified system RAM on a Jetson).
        free_bytes: what is actually available to this process at idle.
        weight_bytes: resident parameter storage at the chosen precision.
        activation_bytes: intermediate tensors, from :func:`estimate`'s coarse
            heuristic -- see that function's warning.
        workspace_bytes: the TensorRT workspace/scratch reservation.
        runtime_overhead_bytes: CUDA context and kernel images.
        streaming_scratch_bytes: ``WEIGHT_STREAMING`` staging buffer; zero
            unless a build actually enabled it.
    """

    total_bytes: int = 0
    free_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    workspace_bytes: int = 0
    runtime_overhead_bytes: int = 0
    streaming_scratch_bytes: int = 0

    @property
    def required_bytes(self) -> int:
        """Total device memory the plan claims, in bytes."""
        return (self.weight_bytes + self.activation_bytes + self.workspace_bytes
                + self.runtime_overhead_bytes + self.streaming_scratch_bytes)

    @property
    def headroom_bytes(self) -> int:
        """``free_bytes - required_bytes``. Negative means it does not fit."""
        return self.free_bytes - self.required_bytes

    @property
    def fits(self) -> bool:
        """True when the plan fits in free memory with nothing left over."""
        return self.headroom_bytes >= 0


def bytes_per_param(precision: str) -> float:
    """Storage width of one parameter at ``precision``, in bytes.

    Args:
        precision: precision name, case- and separator-insensitive
            (``FP16``, ``float16``, ``bfloat16``, ``int8``, ``nvfp4``, ...).

    Returns:
        int or float: 4 for fp32, 2 for fp16/bf16, 1 for int8/fp8, 0.5 for
        int4/nvfp4.

    Raises:
        ValueError: on an unknown precision. Guessing a width here would
            silently misreport whether a model fits, so it raises instead.
    """
    key = str(precision).strip().lower().replace("-", "").replace("_", "")
    if key not in _BYTES_PER_PARAM:
        raise ValueError(
            "unknown precision %r; known: %s"
            % (precision, ", ".join(sorted(_BYTES_PER_PARAM))))
    return _BYTES_PER_PARAM[key]


def cuda_context_bytes(hardware: HardwareProfile,
                       headless: Optional[bool] = None) -> int:
    """Fixed runtime overhead to reserve before any engine is loaded.

    A CUDA context, the kernel images TensorRT pulls in and the cuDNN/cuBLAS
    handles cost hundreds of MiB that no weight/activation arithmetic accounts
    for. On a Jetson the GPU shares system DRAM with the CPU, so the same
    allocations plus host-side staging come out of the one pool -- reserve more.

    Args:
        hardware: target :class:`HardwareProfile`.
        headless: True for a device with no display server, which saves the
            framebuffer share. Left as None this assumes a display IS attached,
            deliberately: the host cannot probe the *target*, and over-reserving
            costs headroom on paper where under-reserving costs an allocation
            failure in flight.

    Returns:
        int: bytes to reserve.
    """
    if hardware.is_jetson:
        return _CTX_JETSON_BYTES
    if headless:
        return _CTX_DGPU_HEADLESS_BYTES
    return _CTX_DGPU_DISPLAY_BYTES


def builder_pool_limits(hardware: HardwareProfile) -> Dict[str, int]:
    """Memory pool caps to set on the ``IBuilderConfig`` for this target.

    On TensorRT 11.1 both pools default to essentially the whole device (see
    :data:`MEASURED_TRT11_DEFAULT_POOL_BYTES`), and capping WORKSPACE alone is
    the classic non-fix: TACTIC_DRAM is what balloons while the builder times
    candidate tactics, so an uncapped TACTIC_DRAM will still take the machine
    down. On a Jetson it defaults to 75% of *unified* memory, which starves the
    rest of the stack, so the caps here are much tighter.

    Args:
        hardware: target :class:`HardwareProfile`.

    Returns:
        Dict[str, int]: ``{'WORKSPACE': bytes, 'TACTIC_DRAM': bytes}``, to be
        applied as ``config.set_memory_pool_limit(MemoryPoolType.<key>, value)``.

    Raises:
        ValueError: if the profile carries no memory size. Sizing a pool from a
            zero total would emit a zero cap and fail the build far from here.
    """
    total = int(hardware.total_mem_bytes)
    if total <= 0:
        raise ValueError(
            "HardwareProfile has total_mem_bytes=%d; cannot size builder pools. "
            "detect() leaves it at 0 when nvidia-smi / /proc/meminfo is "
            "unreadable -- supply the target's memory explicitly." % total)
    if hardware.is_jetson:
        if hardware.is_15w:
            return {"WORKSPACE": min(512 * MIB, total // 16),
                    "TACTIC_DRAM": min(1 * GIB, total // 8)}
        return {"WORKSPACE": min(1 * GIB, total // 8),
                "TACTIC_DRAM": min(2 * GIB, total // 4)}
    return {"WORKSPACE": min(2 * GIB, total // 4),
            "TACTIC_DRAM": min(4 * GIB, total // 2)}


def estimate(plan: Plan, precision: str, hardware: HardwareProfile,
             resident: str = "concurrent",
             activation_factor: float = 0.35) -> Budget:
    """Estimate the runtime footprint of a :class:`Plan` at one precision.

    Weights come from the plan's components -- *all* of them, including
    ``Cadence.COLD`` ones, because a text encoder that runs once per episode is
    resident for the whole episode and costs exactly as much as a hot one.
    Workspace is charged at the builder's WORKSPACE cap: the context's real
    scratch is usually far less, but the cap is the only number that exists
    before the engine does, and over-charging is the safe direction.

    Args:
        plan: the :class:`Plan` to budget.
        precision: precision the weights will be built at (see
            :func:`bytes_per_param`).
        hardware: target :class:`HardwareProfile`.
        resident: ``'concurrent'`` (default) sums every component -- the
            honest assumption for a dual-system model. ``'sequential'`` counts
            only the largest and is legal ONLY if the pipeline destroys one
            engine before creating the next; if the stages overlap for even one
            frame this understates the peak and the process dies at that frame.
        activation_factor: coarse multiplier on the weight bytes standing in for
            intermediate tensors. A heuristic with no shape-walk behind it; the
            ground truth is a real build with ``MONITOR_MEMORY`` and the
            context's device memory read back. A fit that only holds because of
            this term is not a fit.

    Returns:
        Budget: the estimated footprint against this device.

    Raises:
        ValueError: on an unknown ``resident`` mode, a negative
            ``activation_factor``, an unknown precision, or a profile with no
            memory size.
    """
    if resident not in _RESIDENCY_MODES:
        raise ValueError("resident must be one of %s, got %r"
                         % (_RESIDENCY_MODES, resident))
    if activation_factor < 0:
        raise ValueError("activation_factor must be >= 0, got %r"
                         % (activation_factor,))
    total = int(hardware.total_mem_bytes)
    if total <= 0:
        raise ValueError(
            "HardwareProfile has total_mem_bytes=%d; cannot budget against an "
            "unknown device size." % total)

    per_param = bytes_per_param(precision)
    per_component = [c.weight_bytes(per_param) for c in plan.components]
    if not per_component:
        weights = 0
    elif resident == "sequential":
        weights = max(per_component)
    else:
        weights = sum(per_component)

    fraction = _JETSON_FREE_FRACTION if hardware.is_jetson else _DGPU_FREE_FRACTION
    return Budget(
        total_bytes=total,
        free_bytes=int(total * fraction),
        weight_bytes=int(weights),
        activation_bytes=int(weights * activation_factor),
        workspace_bytes=builder_pool_limits(hardware)["WORKSPACE"],
        runtime_overhead_bytes=cuda_context_bytes(hardware),
        streaming_scratch_bytes=0,
    )


def recommendations(budget: Budget, hardware: HardwareProfile,
                    precision: Optional[str] = None) -> List[str]:
    """Ordered remedies for a budget that does not fit, most effective first.

    Args:
        budget: the :class:`Budget` produced by :func:`estimate`.
        hardware: target :class:`HardwareProfile`; the advice differs on a
            Jetson, where weight streaming moves bytes within the same physical
            DRAM and therefore frees nothing.
        precision: the precision the budget was estimated at. Supplying it names
            the exact next step down and suppresses the advice at the floor;
            without it the item falls back to the generic "one step halves it".

    Returns:
        List[str]: concrete remedies, or ``[]`` when the budget already fits.
    """
    if budget.fits:
        return []
    items = (_rec_precision(budget, precision),
             _rec_streaming(budget, hardware),
             _rec_strip_plan(budget),
             _rec_sequential(hardware),
             _rec_offload(budget))
    return [text for text in items if text]


def _rec_precision(budget: Budget, precision: Optional[str]) -> Optional[str]:
    """Remedy: build the weights one step narrower."""
    if budget.weight_bytes <= 0:
        return None
    if precision is None:
        saved = budget.weight_bytes // 2
        step = "one step (each step halves the parameter width)"
    else:
        key = str(precision).strip().lower().replace("-", "").replace("_", "")
        nxt = _NEXT_PRECISION.get(key, "sentinel")
        if nxt == "sentinel":
            raise ValueError("unknown precision %r" % (precision,))
        if nxt is None:
            return None
        ratio = bytes_per_param(nxt) / float(bytes_per_param(key))
        saved = int(budget.weight_bytes * (1.0 - ratio))
        step = "one step (%s -> %s)" % (key, nxt)
    return ("Drop the weight precision %s: frees ~%s of resident weights, and "
            "the activations shrink with them. Deficit to close: %s."
            % (step, _fmt(saved), _fmt(-budget.headroom_bytes)))


def _rec_streaming(budget: Budget, hardware: HardwareProfile) -> Optional[str]:
    """Remedy: stream weights from host memory -- dGPU only, past the threshold."""
    if budget.weight_bytes <= 0.5 * budget.free_bytes:
        return None
    if hardware.is_jetson:
        return ("Do NOT reach for BuilderFlag.WEIGHT_STREAMING on %s: the GPU "
                "shares system DRAM with the CPU, so streamed weights sit in "
                "the same physical memory and free nothing. Precision, "
                "residency and offload are the only levers here."
                % (hardware.jetson_model or hardware.gpu_name))
    stream_budget = min(budget.free_bytes // 2, budget.weight_bytes // 2)
    return ("Enable BuilderFlag.WEIGHT_STREAMING: weights are %.0f%% of free "
            "memory, past the point where they can all stay resident. Build "
            "strongly typed with the flag, then set the runtime budget to "
            "NVIDIA's formula min(free // 2, streamable_weights_size // 2) = "
            "min(%s, %s) = %s. It trades PCIe bandwidth for residency, so "
            "re-measure latency afterwards."
            % (100.0 * budget.weight_bytes / max(budget.free_bytes, 1),
               _fmt(budget.free_bytes // 2), _fmt(budget.weight_bytes // 2),
               _fmt(stream_budget)))


def _rec_strip_plan(budget: Budget) -> Optional[str]:
    """Remedy: ship a weightless plan and refit at load."""
    if budget.weight_bytes <= 0:
        return None
    return ("Build with BuilderFlag.REFIT *and* BuilderFlag.STRIP_PLAN "
            "together: STRIP_PLAN alone leaves ICudaEngine.refittable False and "
            "the weights are simply gone. Together they ship a weightless plan "
            "(~%s off the file and off the host-side copy at load) refilled "
            "through IRefitter. This buys deployment size and host RAM at load, "
            "NOT device headroom -- refitted weights are resident again."
            % _fmt(budget.weight_bytes))


def _rec_sequential(hardware: HardwareProfile) -> str:
    """Remedy: hold one engine at a time."""
    where = ("unified memory, so anything freed goes back to the whole system"
             if hardware.is_jetson else "device memory")
    return ("Tear each engine down before building the next and re-estimate "
            "with resident='sequential': only the largest one stays in %s. "
            "Legal only if the stages genuinely do not overlap -- if System-2 "
            "must stay warm across a System-1 frame, it is not available."
            % where)


def _rec_offload(budget: Budget) -> str:
    """Remedy: move the biggest component off the device entirely."""
    return ("Move the largest component off-device into a host-side sidecar "
            "over the existing bridge: at least %s has to leave the device. "
            "This repo already runs GPU inference in that shape (the ROS2 "
            "sidecar serving the Noetic container), so it is a deployment "
            "change rather than a new mechanism." % _fmt(-budget.headroom_bytes))


def _fmt(n: float) -> str:
    """Format a byte count for a human reading the report."""
    value = float(n)
    if abs(value) >= GIB:
        return "%.2f GiB" % (value / GIB)
    if abs(value) >= MIB:
        return "%.0f MiB" % (value / MIB)
    return "%d B" % int(value)
