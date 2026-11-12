"""Sample the machine while a campaign runs, one CSV row per interval, and summarise the log afterwards.

A campaign's "utilization" is a claim about the whole run, not about the moment
someone looked at ``nvidia-smi``: the LLM saturates the card while twenty
workers wait on it and idles while they render, and the tail of a one-process-
per-scene campaign runs on a handful of workers. So the machine is sampled
every ``--interval`` seconds for the life of the run -- GPU utilisation and
VRAM, CPU, RAM, disk (``campaign_monitor.resources``) and the VRAM of every
process on the card (``nvidia-smi --query-compute-apps``) -- into a CSV the
report reads back.

::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.utilization_log \\
        --out runs/X_support/utilization.csv --interval 15          # until killed
    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.utilization_log \\
        --summarise runs/X_support/utilization.csv                    # means, percentiles, process VRAM

Standard library plus ``campaign_monitor.resources`` only; nothing here reads
a run directory or touches a worker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from sparx_agency.tools.campaign_monitor.resources import CpuMeter, sample

FIELDS = ("utc", "gpu_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "cpu_pct", "ram_used_gb", "ram_total_gb",
          "disk_free_gb", "gpu_processes", "ollama_mib", "renderer_mib_max", "renderer_mib_sum", "load1")


def gpu_processes(index: int = 0) -> List[Dict[str, object]]:
    """``[{pid, name, mib}]`` of every process holding memory on the card, or ``[]`` where unavailable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--id=%d" % index, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    except (subprocess.SubprocessError, OSError):
        return []
    rows = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        rows.append({"pid": int(parts[0]), "name": parts[1], "mib": int(parts[2])})
    return rows


def one_row(meter: CpuMeter, disk_path: Path, gpu_index: int) -> Dict[str, object]:
    """One sample, shaped for :data:`FIELDS`; the renderers are every GPU process that is not Ollama."""
    snapshot = sample(disk_path, meter, gpu_index)
    processes = gpu_processes(gpu_index)
    ollama = [p["mib"] for p in processes if "ollama" in p["name"].lower()]
    renderers = [p["mib"] for p in processes if "ollama" not in p["name"].lower()]
    gpu = snapshot.gpu
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_util_pct": None if gpu is None else gpu.utilization_pct,
        "gpu_mem_used_mib": None if gpu is None else gpu.memory_used_mb,
        "gpu_mem_total_mib": None if gpu is None else gpu.memory_total_mb,
        "cpu_pct": round(snapshot.cpu_pct, 1),
        "ram_used_gb": round(snapshot.ram_used_gb, 2),
        "ram_total_gb": round(snapshot.ram_total_gb, 2),
        "disk_free_gb": round(snapshot.disk_free_gb, 1),
        "gpu_processes": len(processes),
        "ollama_mib": sum(ollama),
        "renderer_mib_max": max(renderers) if renderers else 0,
        "renderer_mib_sum": sum(renderers),
        "load1": round(os.getloadavg()[0], 2),
    }


def record(out: Path, interval_s: float, gpu_index: int, disk_path: Path) -> None:
    """Append one row per ``interval_s`` to ``out`` until SIGTERM/SIGINT; the header is written once."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = {"now": False}

    def _stop(signum, frame):
        stop["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    meter = CpuMeter()
    meter.sample()                                   # prime the jiffies baseline
    new = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        while not stop["now"]:
            writer.writerow(one_row(meter, disk_path, gpu_index))
            stream.flush()
            deadline = time.time() + interval_s
            while not stop["now"] and time.time() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.time())))


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarise(path: Path) -> Dict[str, object]:
    """Means and percentiles over the log, and the duration it spans; the numbers the run report quotes."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("no samples in %s" % path)

    def floats(key):
        return [float(r[key]) for r in rows if r.get(key) not in (None, "", "None")]

    def stats(key):
        values = floats(key)
        if not values:
            return None
        return {"mean": round(sum(values) / len(values), 1), "median": _percentile(values, 0.5),
                "p10": _percentile(values, 0.1), "p90": _percentile(values, 0.9), "max": max(values)}

    first, last = rows[0]["utc"], rows[-1]["utc"]
    started = time.mktime(time.strptime(first, "%Y-%m-%dT%H:%M:%SZ"))
    ended = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))
    gpu_util = floats("gpu_util_pct")
    return {"samples": len(rows), "first_utc": first, "last_utc": last, "span_h": round((ended - started) / 3600.0, 2),
            "gpu_util_pct": stats("gpu_util_pct"),
            "gpu_util_share_above_80": (round(sum(1 for v in gpu_util if v >= 80.0) / len(gpu_util), 3) if gpu_util else None),
            "gpu_mem_used_mib": stats("gpu_mem_used_mib"), "cpu_pct": stats("cpu_pct"), "ram_used_gb": stats("ram_used_gb"),
            "load1": stats("load1"), "ollama_mib": stats("ollama_mib"), "renderer_mib_max": stats("renderer_mib_max"),
            "renderer_mib_sum": stats("renderer_mib_sum"), "gpu_processes": stats("gpu_processes"),
            "disk_free_gb_last": float(rows[-1]["disk_free_gb"])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="CSV to append samples to (recording mode)")
    parser.add_argument("--interval", type=float, default=15.0, help="Seconds between samples")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd(), help="Where free disk is measured")
    parser.add_argument("--summarise", type=Path, metavar="CSV", help="Print a JSON summary of a recorded log and exit")
    args = parser.parse_args(argv)
    if args.summarise is not None:
        print(json.dumps(summarise(args.summarise), indent=2))
        return 0
    if args.out is None:
        parser.error("--out (record) or --summarise CSV is required")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    record(args.out, args.interval, args.gpu_index, args.disk_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
