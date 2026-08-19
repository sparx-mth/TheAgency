"""What the *build toolchain* on this machine can actually do, not just what GPU it is.

``sparx_agency.tasks.common.hardware.detect`` answers "what silicon is this?".
That is necessary but not sufficient: two machines with the same GPU produce
incompatible engines and support different precisions depending on which
TensorRT they import. The decisions this package makes -- which precision to
target, whether DLA is even reachable, whether a calibrator exists -- are
decided by the *toolchain generation* far more than by the silicon.

The single fact that matters most here: **TensorRT 11 removed weak typing.**
On TensorRT <= 10 you build a weakly-typed network and set
``BuilderFlag.FP16``, and the builder mixes precision per layer with FP32
accumulation. On TensorRT >= 11 that flag does not exist; every network is
strongly typed and the engine's precision is exactly whatever the ONNX carries.
The same source that "works" on both silently produces an FP32 engine on 11 if
nobody converted the graph. :class:`Target` makes that difference explicit and
checkable instead of implicit.

Everything here is best-effort probing of external tools and never raises on a
missing one -- a field is left at a conservative default, matching the contract
of the hardware detector it builds on. The one exception is
:func:`require_buildable`, which exists precisely to fail loud.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from sparx_agency.tasks.common.hardware.detect import HardwareProfile, detect

#: TensorRT releases up to and including this one still support DLA. Owned by
#: :mod:`..trt_optimizer.engine.dla`, which is where the rest of the DLA
#: decision lives; re-exported here so probing does not import that module.
from sparx_agency.tasks.common.trt_optimizer.engine.dla import (  # noqa: E402
    LAST_TRT_WITH_DLA as LAST_DLA_TRT,
)


@dataclass
class Target:
    """The hardware plus the toolchain that will build and run engines on it.

    Args:
        hardware: the detected :class:`HardwareProfile`.
        trt_version: ``tensorrt.__version__`` string, or None if not importable.
        cuda_driver_version: driver-reported CUDA version, or None.
        torch_version: ``torch.__version__``, or None.
        free_mem_bytes: currently free device memory, or 0 if unknown.
        strongly_typed: True when this TensorRT has removed weak typing, so
            precision must be baked into the ONNX rather than set with a flag.
        dla_usable: True only when the runtime *and* the version support DLA.
        clocks_lockable: True when GPU clocks can be pinned for reproducible
            timing. False on this laptop without root, which makes absolute
            latency numbers comparable only within a single interleaved run.
        toolchain: names of optional build tools that were found.
        missing: names of optional build tools that were NOT found, each with
            what it would have enabled.
    """

    hardware: HardwareProfile
    trt_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    torch_version: Optional[str] = None
    free_mem_bytes: int = 0
    strongly_typed: bool = False
    dla_usable: bool = False
    clocks_lockable: bool = False
    toolchain: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def target_tag(self):
        """The per-device slug naming the engine output directory."""
        return self.hardware.target_tag

    @property
    def trt_major_minor(self):
        """TensorRT (major, minor) as ints, or None if unknown."""
        if not self.trt_version:
            return None
        m = re.match(r"(\d+)\.(\d+)", str(self.trt_version))
        return (int(m.group(1)), int(m.group(2))) if m else None

    @property
    def engine_identity(self):
        """The tuple an engine is locked to; a mismatch means rebuild.

        Serialized engines deserialize only under the exact TensorRT build and
        GPU compute capability that produced them -- including the patch level,
        which JetPack point releases bump.
        """
        return (self.hardware.target_tag, str(self.trt_version),
                self.hardware.sm)

    def supported_precisions(self):
        """Precisions this target can actually build today, cheapest first.

        Reflects toolchain reality, not the silicon's datasheet: a format whose
        only producer is ``nvidia-modelopt`` is excluded when that package is
        absent, because a build that silently falls back to FP32 is worse than a
        refusal.
        """
        out = ["fp32"]
        if "onnxconverter_common" in self.toolchain:
            out.append("fp16")
        if "modelopt" in self.toolchain:
            out.extend(["bf16", "int8", "fp8"])
            if self.hardware.sm is not None and self.hardware.sm >= 100:
                out.append("nvfp4")
        elif not self.strongly_typed:
            # TensorRT <= 10 still ships the INT8 entropy calibrator, so INT8 is
            # reachable without modelopt on the Orin's stack.
            out.append("int8")
        return out


def _run(cmd):
    """Run a command, returning stdout or '' on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001  (tool missing / not permitted)
        return ""


def _free_mem_bytes():
    """Free device memory in bytes from nvidia-smi, or 0."""
    txt = _run(["nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"]).strip()
    if not txt:
        return 0
    try:
        return int(float(txt.splitlines()[0])) * (1 << 20)
    except ValueError:
        return 0


def _clocks_lockable():
    """True when GPU application clocks can be pinned (needs root on most hosts)."""
    txt = _run(["nvidia-smi", "--query-gpu=clocks.applications.graphics",
                "--format=csv,noheader"]).strip()
    return bool(txt) and "N/A" not in txt and "Not Supported" not in txt


def _probe_tool(name, importer):
    """Return (found, note) for one optional build dependency."""
    try:
        importer()
        return True, name
    except Exception as exc:  # noqa: BLE001  (probing, absence is the answer)
        return False, "%s (%s)" % (name, exc.__class__.__name__)


def _probe_toolchain():
    """Probe every optional build tool; return (found, missing_with_reasons)."""
    probes = (
        ("onnx", lambda: __import__("onnx"), "ONNX export and graph inspection"),
        ("onnxruntime", lambda: __import__("onnxruntime"),
         "the FP32 CPU parity gate"),
        ("onnxconverter_common", lambda: __import__("onnxconverter_common"),
         "FP16 graph conversion, which is the ONLY way to get FP16 on TensorRT 11"),
        ("onnxslim", lambda: __import__("onnxslim"), "graph simplification"),
        ("modelopt", lambda: __import__("modelopt"),
         "BF16/INT8/FP8/NVFP4 quantized ONNX (Q/DQ)"),
        ("pycuda", lambda: __import__("pycuda.driver"),
         "device buffers for the engine runner and INT8 calibration"),
    )
    found, missing = [], []
    for name, importer, buys in probes:
        ok, _ = _probe_tool(name, importer)
        if ok:
            found.append(name)
        else:
            missing.append("%s -- would enable %s" % (name, buys))
    return found, missing


def _trt_facts():
    """(version, strongly_typed, dla_cores) from the importable TensorRT, if any."""
    try:
        import tensorrt as trt
    except Exception:  # noqa: BLE001
        return None, False, 0
    # BuilderFlag.FP16 genuinely disappears in TensorRT 11, so hasattr is the
    # correct probe here (unlike DLA, whose symbols outlive its support).
    strongly_typed = not hasattr(trt.BuilderFlag, "FP16")
    cores = 0
    try:
        cores = int(trt.Runtime(trt.Logger(trt.Logger.ERROR)).num_DLA_cores)
    except Exception:  # noqa: BLE001
        cores = 0
    return str(trt.__version__), strongly_typed, cores


def _torch_version():
    """``torch.__version__`` or None."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:  # noqa: BLE001
        return None


def _cuda_driver_version():
    """Driver-reported CUDA version string from nvidia-smi, or None."""
    txt = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return txt.strip().splitlines()[0] if txt.strip() else None


def resolve(hardware=None):
    """Probe this machine and return its :class:`Target`.

    Args:
        hardware: an already-detected profile; detected here when omitted.

    Returns:
        A fully populated :class:`Target`. Never raises -- every probe degrades
        to a conservative default so this can run on a machine with no GPU at
        all (which is exactly the case when planning a build for another one).
    """
    hw = hardware if hardware is not None else detect()
    trt_version, strongly_typed, dla_cores = _trt_facts()
    found, missing = _probe_toolchain()
    version = None
    if trt_version:
        m = re.match(r"(\d+)\.(\d+)", trt_version)
        version = (int(m.group(1)), int(m.group(2))) if m else None
    dla_usable = bool(dla_cores > 0 and version is not None
                      and version <= LAST_DLA_TRT)
    return Target(
        hardware=hw,
        trt_version=trt_version,
        cuda_driver_version=_cuda_driver_version(),
        torch_version=_torch_version(),
        free_mem_bytes=_free_mem_bytes() or hw.total_mem_bytes,
        strongly_typed=strongly_typed,
        dla_usable=dla_usable,
        clocks_lockable=_clocks_lockable(),
        toolchain=found,
        missing=missing,
    )


def require_buildable(target, precision):
    """Raise unless ``target`` can genuinely build engines at ``precision``.

    This is the fail-loud counterpart to all the best-effort probing above. A
    build that "succeeds" while silently producing FP32 because the FP16
    converter was missing is the exact failure this prevents.

    Raises:
        RuntimeError: TensorRT is not importable, or ``precision`` is not in
            :meth:`Target.supported_precisions`, with the missing tool named.
    """
    if not target.trt_version:
        raise RuntimeError(
            "TensorRT is not importable from this interpreter. Engines must be "
            "built with the SAME python tensorrt the runtime imports; on this "
            "machine that is the 'navdp' conda env, not .venv.")
    supported = target.supported_precisions()
    if precision not in supported:
        raise RuntimeError(
            "Cannot build %s on this toolchain (supported: %s). Missing: %s"
            % (precision, ", ".join(supported), "; ".join(target.missing) or "none"))


def describe(target):
    """A short multi-line human summary for the top of a report."""
    hw = target.hardware
    lines = [
        "GPU:        %s (sm_%s, %.1f GiB total, %.1f GiB free)"
        % (hw.gpu_name, hw.sm, hw.total_mem_bytes / (1 << 30),
           target.free_mem_bytes / (1 << 30)),
        "Target tag: %s" % target.target_tag,
        "TensorRT:   %s (%s typing)"
        % (target.trt_version or "not importable",
           "strong" if target.strongly_typed else "weak"),
        "Torch:      %s" % (target.torch_version or "not importable"),
        "DLA:        %s" % ("usable, %d cores" % hw.dla_cores if target.dla_usable
                            else "not usable on this runtime"),
        "Precisions: %s" % ", ".join(target.supported_precisions()),
    ]
    if not target.clocks_lockable:
        lines.append("Clocks:     NOT lockable -- interleave A/B timing runs")
    return "\n".join(lines)
