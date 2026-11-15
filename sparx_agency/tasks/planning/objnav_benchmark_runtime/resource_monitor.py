"""Sample the machine while a campaign runs, and summarise the samples afterwards.

A campaign's report wants to say how full the machine was -- the card's compute
and memory, the cores, the RAM -- and the harness had nothing that recorded it:
the 2026-11-05 MP3D run's "peaks: RAM 46 %, CPU 52 %, VRAM 87 %, GPU 53 %" were
read off a terminal. This writes one CSV row every ``--interval`` seconds::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.resource_monitor record out.csv --interval 15
    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.resource_monitor summarize out.csv [--since-utc ISO]

Each row: the UTC time, GPU utilisation (%), VRAM used and total (MiB), CPU
utilisation over the interval (%, all cores), RAM used and total (MiB), the
one-minute load, and how many ``mp3d.run`` / ``hm3d.run`` / ``gibson.run``
workers are alive. ``summarize`` prints the mean, the median and the peak of each
over the samples (optionally those after a time), the sampled span, and the
share of samples the GPU spent above 80 % busy -- the number "about 90 %
utilisation" is judged by.

Standard library only; nvidia-smi for the card (a row's GPU fields are empty
when it fails, the rest is still sampled).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

FIELDS = ("utc", "unix", "gpu_util_pct", "vram_used_mib", "vram_total_mib", "cpu_pct", "ram_used_mib",
          "ram_total_mib", "load1", "workers")
#: Command-line marks of a benchmark worker process, counted in the ``workers`` column.
WORKER_MARKS = ("objnav_benchmark_runtime.mp3d.run", "objnav_benchmark_runtime.hm3d.run",
                "objnav_benchmark_runtime.gibson.run")


def gpu_sample(device=0):
    """``(utilisation %, used MiB, total MiB)`` of the card, or ``None`` when nvidia-smi fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits", "-i", str(device)],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        util, used, total = [float(v.strip()) for v in out.strip().splitlines()[0].split(",")]
        return util, used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cpu_times():
    """``(busy, total)`` jiffies over every core, from ``/proc/stat``."""
    with open("/proc/stat") as stream:
        fields = [float(v) for v in stream.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
    return sum(fields) - idle, sum(fields)


def ram_sample():
    """``(used MiB, total MiB)`` from ``/proc/meminfo`` (used = total - available)."""
    values = {}
    with open("/proc/meminfo") as stream:
        for line in stream:
            key, rest = line.split(":", 1)
            values[key] = float(rest.split()[0])
    total = values["MemTotal"] / 1024.0
    return total - values.get("MemAvailable", values["MemFree"]) / 1024.0, total


def worker_count():
    """Benchmark worker processes alive now, by their command lines."""
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as stream:
                argv = stream.read().decode(errors="replace")
        except OSError:
            continue
        if any(mark in argv for mark in WORKER_MARKS):
            count += 1
    return count


def record(path, interval_s, device=0, samples=None, sleep=time.sleep):
    """Append one row every ``interval_s`` seconds until interrupted (or ``samples`` rows, for tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    busy0, total0 = cpu_times()
    written = 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
            stream.flush()
        while samples is None or written < samples:
            sleep(interval_s)
            busy1, total1 = cpu_times()
            cpu = 100.0 * (busy1 - busy0) / max(total1 - total0, 1e-9)
            busy0, total0 = busy1, total1
            gpu = gpu_sample(device)
            ram_used, ram_total = ram_sample()
            now = time.time()
            writer.writerow({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "unix": round(now, 1),
                             "gpu_util_pct": "" if gpu is None else round(gpu[0], 1),
                             "vram_used_mib": "" if gpu is None else round(gpu[1]),
                             "vram_total_mib": "" if gpu is None else round(gpu[2]),
                             "cpu_pct": round(cpu, 1), "ram_used_mib": round(ram_used), "ram_total_mib": round(ram_total),
                             "load1": round(os.getloadavg()[0], 2), "workers": worker_count()})
            stream.flush()
            written += 1
    return written


def summarize(path, since_unix=None, until_unix=None):
    """Mean, median, peak of every sampled quantity over the rows in the window, and the span they cover."""
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = float(row["unix"])
            if (since_unix is None or stamp >= since_unix) and (until_unix is None or stamp <= until_unix):
                rows.append(row)
    if not rows:
        return {"samples": 0}
    out = {"samples": len(rows), "first_utc": rows[0]["utc"], "last_utc": rows[-1]["utc"],
           "span_h": round((float(rows[-1]["unix"]) - float(rows[0]["unix"])) / 3600.0, 2)}
    for key in ("gpu_util_pct", "vram_used_mib", "cpu_pct", "ram_used_mib", "load1", "workers"):
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        if values:
            out[key] = {"mean": round(statistics.fmean(values), 1), "median": round(statistics.median(values), 1),
                        "peak": round(max(values), 1)}
    totals = [float(row["vram_total_mib"]) for row in rows if row.get("vram_total_mib")]
    if totals and "vram_used_mib" in out:
        out["vram_pct"] = {k: round(100.0 * v / totals[0], 1) for k, v in out["vram_used_mib"].items()}
    rams = [float(row["ram_total_mib"]) for row in rows if row.get("ram_total_mib")]
    if rams and "ram_used_mib" in out:
        out["ram_pct"] = {k: round(100.0 * v / rams[0], 1) for k, v in out["ram_used_mib"].items()}
    gpu = [float(row["gpu_util_pct"]) for row in rows if row.get("gpu_util_pct")]
    if gpu:
        out["gpu_busy_share_above_80"] = round(sum(v >= 80.0 for v in gpu) / len(gpu), 3)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="append a row every --interval seconds until interrupted")
    rec.add_argument("csv", type=Path)
    rec.add_argument("--interval", type=float, default=15.0)
    rec.add_argument("--gpu-device", type=int, default=0)
    summ = sub.add_parser("summarize", help="mean / median / peak of each column over the window")
    summ.add_argument("csv", type=Path)
    summ.add_argument("--since-utc", help="ISO-8601 UTC, e.g. 2026-11-11T09:00:00Z")
    summ.add_argument("--until-utc")
    args = parser.parse_args(argv)
    if args.command == "record":
        try:
            record(args.csv, args.interval, args.gpu_device)
        except KeyboardInterrupt:
            pass
        return 0
    import calendar
    import json

    def to_unix(text):
        return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")) if text else None

    print(json.dumps(summarize(args.csv, to_unix(args.since_utc), to_unix(args.until_utc)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
