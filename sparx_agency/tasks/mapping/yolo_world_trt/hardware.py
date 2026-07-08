"""Detect the GPU / Jetson SoC and power state that parameterize the build.

A TensorRT engine is locked to the exact compute capability and TensorRT build
that produced it, and the right builder knobs differ sharply between an x86 dGPU
and a Jetson AGX Orin pinned to 15 W. This module produces a
:class:`HardwareProfile` that :mod:`build_policy` and :mod:`build_engine` consume,
and it runs *gracefully on both* the dev laptop (no Jetson sysfs, no DLA) and the
Orin.

Unlike the NavDP hardware probe -- whose ViT graphs are a poor DLA fit, so it
hard-codes ``allow_dla=False`` -- this one **enables DLA on Orin**: a prompt-baked
YOLO-World export is a pure convolutional detector, exactly the CNN workload the
two NVDLA cores are built for. Offloading the backbone/neck to DLA frees the GPU,
which is the throughput *and* power bottleneck at a 15 W cap.

Everything is best-effort and never raises on a missing tool/node: a field is
left at its conservative default. Pure standard library; importable anywhere.
"""
from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class HardwareProfile:
    """Resolved hardware description used to configure a TensorRT build."""

    arch: str
    is_jetson: bool
    gpu_name: str = "unknown"
    jetson_model: Optional[str] = None
    nvpmodel_id: Optional[int] = None
    nvpmodel_name: Optional[str] = None
    power_budget_w: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    total_mem_bytes: int = 0
    dla_cores: int = 0
    allow_dla: bool = False
    recommended_workspace_bytes: int = 1 << 30  # 1 GiB conservative default
    target_tag: str = "unknown"

    @property
    def sm(self) -> Optional[int]:
        """Compute capability as the integer SM number (e.g. 87), or None."""
        if self.compute_capability is None:
            return None
        return self.compute_capability[0] * 10 + self.compute_capability[1]

    @property
    def is_15w(self) -> bool:
        """True on a Jetson whose active nvpmodel budget is 15 W or below."""
        return self.is_jetson and (self.power_budget_w or 99) <= 15


def _run(cmd):
    """Run a command, returning stdout (str) or '' on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 (tool missing / not permitted)
        return ""


def _read(path):
    """Read a sysfs/devicetree file, returning '' on failure (nul-stripped)."""
    try:
        return Path(path).read_text(errors="ignore").replace("\x00", "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _is_jetson(arch):
    """True on a Tegra SoC (aarch64 + L4T release / Jetson device-tree model)."""
    if arch != "aarch64":
        return False
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = _read("/proc/device-tree/model").lower()
    return "jetson" in model or "orin" in model or "tegra" in model


def _nvpmodel():
    """Parse ``nvpmodel -q`` -> (mode_id, mode_name); (None, None) if absent."""
    txt = _run(["nvpmodel", "-q"])
    name_m = re.search(r"NV Power Mode.*?:?\s*(\w+)", txt)
    num_m = re.search(r"^\s*(\d+)\s*$", txt, re.MULTILINE)
    name = name_m.group(1) if name_m else None
    num = int(num_m.group(1)) if num_m else None
    return num, name


def _power_budget_w(mode_name):
    """Extract a wattage from an nvpmodel mode name like ``MODE_15W`` -> 15."""
    if not mode_name:
        return None
    m = re.search(r"(\d+)\s*W", mode_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _x86_compute_capability():
    """Read compute capability from nvidia-smi (-> (major, minor)) or None."""
    txt = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    m = re.search(r"(\d+)\.(\d+)", txt)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _x86_gpu(profile):
    """Fill GPU name / memory / compute capability for an x86 dGPU (no DLA)."""
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
    mem = _run(["nvidia-smi", "--query-gpu=memory.total",
                "--format=csv,noheader,nounits"]).strip()
    if name:
        profile.gpu_name = name.splitlines()[0]
    if mem:
        try:
            profile.total_mem_bytes = int(float(mem.splitlines()[0])) * (1 << 20)
        except ValueError:
            pass
    profile.compute_capability = _x86_compute_capability()
    profile.dla_cores = 0
    profile.allow_dla = False


def _jetson_fill(profile):
    """Fill Jetson fields (model, power mode, DLA cores, memory, SM)."""
    profile.jetson_model = _read("/proc/device-tree/model") or "Jetson"
    profile.gpu_name = profile.jetson_model
    profile.nvpmodel_id, profile.nvpmodel_name = _nvpmodel()
    profile.power_budget_w = _power_budget_w(profile.nvpmodel_name)
    model_l = profile.jetson_model.lower()
    if "orin" in model_l:
        profile.compute_capability = (8, 7)
        profile.dla_cores = 2                    # NVDLA v2, two cores on AGX/NX Orin
    elif "xavier" in model_l:
        profile.compute_capability = (7, 2)
        profile.dla_cores = 2
    # A prompt-baked YOLO-World export is a CNN -> DLA is worth it whenever present.
    profile.allow_dla = profile.dla_cores > 0
    meminfo = _read("/proc/meminfo")
    m = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
    if m:
        profile.total_mem_bytes = int(m.group(1)) * 1024


def _workspace_bytes(profile):
    """Conservative TensorRT workspace pool size for this device."""
    if profile.total_mem_bytes <= 0:
        return 1 << 30
    if profile.is_jetson:
        # Memory is shared with the CPU; stay well under the budget, esp. at 15 W.
        cap = (2 << 30) if not profile.is_15w else (1 << 30)
        return min(cap, profile.total_mem_bytes // 4)
    return min(4 << 30, profile.total_mem_bytes // 2)


def _target_tag(profile):
    """A stable per-target slug used to name the engine output directory."""
    if profile.is_jetson and profile.compute_capability == (8, 7):
        return "orin_sm87"
    sm = profile.sm
    slug = re.sub(r"[^a-z0-9]+", "", profile.gpu_name.lower()) or profile.arch
    return "%s_sm%s" % (slug[:16], sm if sm is not None else "x")


def detect() -> HardwareProfile:
    """Detect the current machine's hardware profile (never raises)."""
    arch = platform.machine()
    profile = HardwareProfile(arch=arch, is_jetson=_is_jetson(arch))
    if profile.is_jetson:
        _jetson_fill(profile)
    else:
        _x86_gpu(profile)
    profile.recommended_workspace_bytes = _workspace_bytes(profile)
    profile.target_tag = _target_tag(profile)
    return profile


def main():
    import dataclasses
    import json
    print(json.dumps(dataclasses.asdict(detect()), indent=2, default=str))


if __name__ == "__main__":
    main()
