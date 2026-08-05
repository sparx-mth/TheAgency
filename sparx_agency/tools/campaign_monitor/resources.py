"""What the machine is actually doing right now.

Sampled from ``/proc`` and ``nvidia-smi`` rather than ``psutil``, which is not
installed in every interpreter this runs under. Every reading is best-effort:
a machine with no NVIDIA driver reports ``None`` for the GPU rather than
raising, because a dashboard that dies when one number is unavailable is worse
than one that prints a dash.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class GpuSample:
    """One GPU's instantaneous load."""

    name: str
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float

    @property
    def memory_pct(self) -> float:
        """VRAM in use, as a percentage of the card's total."""
        return 100.0 * self.memory_used_mb / self.memory_total_mb if self.memory_total_mb else 0.0


@dataclass(frozen=True)
class Resources:
    """A snapshot of the whole machine, as the dashboard wants to show it."""

    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    gpu: Optional[GpuSample]

    @property
    def ram_pct(self) -> float:
        """Resident memory in use, as a percentage of the machine's total."""
        return 100.0 * self.ram_used_gb / self.ram_total_gb if self.ram_total_gb else 0.0


def _read_cpu_times() -> Tuple[float, float]:
    """``(busy, total)`` jiffies from ``/proc/stat``'s aggregate line."""
    fields = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    values = [float(field) for field in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    total = sum(values)
    return total - idle, total


class CpuMeter:
    """Percentage busy since the previous call.

    ``/proc/stat`` is cumulative, so a single read says only what the machine
    has averaged since boot. Keeping the previous sample turns it into the
    instantaneous number a dashboard wants.
    """

    def __init__(self) -> None:
        self._previous = _read_cpu_times()

    def sample(self) -> float:
        """Busy percentage over the interval since the last :meth:`sample`."""
        busy, total = _read_cpu_times()
        previous_busy, previous_total = self._previous
        self._previous = (busy, total)
        elapsed = total - previous_total
        return 100.0 * (busy - previous_busy) / elapsed if elapsed > 0 else 0.0


def memory_gb() -> Tuple[float, float]:
    """``(used, total)`` gibibytes, counting cache as free like ``free -g`` does."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = float(rest.split()[0]) / (1024.0 * 1024.0)
    total = values.get("MemTotal", 0.0)
    return total - values.get("MemAvailable", 0.0), total


def gpu_sample(index: int = 0) -> Optional[GpuSample]:
    """The named GPU's load, or ``None`` where ``nvidia-smi`` is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    query = "name,memory.used,memory.total,utilization.gpu"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={index}", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    parts = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(parts) != 4:
        return None
    return GpuSample(parts[0], float(parts[1]), float(parts[2]), float(parts[3]))


def sample(disk_path: Path, cpu_meter: CpuMeter, gpu_index: int = 0) -> Resources:
    """One snapshot of everything, with the disk measured at ``disk_path``."""
    usage = shutil.disk_usage(disk_path)
    used_ram, total_ram = memory_gb()
    return Resources(
        cpu_pct=cpu_meter.sample(),
        ram_used_gb=used_ram,
        ram_total_gb=total_ram,
        disk_used_gb=usage.used / 1e9,
        disk_free_gb=usage.free / 1e9,
        gpu=gpu_sample(gpu_index),
    )


def directory_bytes(root: Path) -> int:
    """Total bytes under ``root``.

    Shells out to ``du`` because a campaign directory holds hundreds of
    thousands of small files and Python's ``os.walk`` over that is slow enough
    to make the dashboard stutter.
    """
    if not root.exists():
        return 0
    try:
        completed = subprocess.run(["du", "-sb", str(root)], capture_output=True,
                                   text=True, timeout=120, check=True)
    except (subprocess.SubprocessError, OSError):
        return 0
    return int(completed.stdout.split()[0])


class CachedDirectorySize:
    """``du`` over a growing campaign, re-run no more often than ``interval_s``.

    Scanning is seconds of work on a large tree; the dashboard redraws every
    second or two. Without the cache the display spends all its time in ``du``.
    """

    def __init__(self, root: Path, interval_s: float = 30.0) -> None:
        self.root = root
        self.interval_s = interval_s
        self._bytes = 0
        self._sampled_at = 0.0

    def get(self) -> int:
        """The most recent size, refreshing it when the cache has expired."""
        now = time.time()
        if now - self._sampled_at >= self.interval_s:
            self._bytes = directory_bytes(self.root)
            self._sampled_at = now
        return self._bytes
