"""Keep a sharded campaign under a utilisation ceiling by moving its live jobs file.

``shard_pool.run_pool`` reads ``--jobs-file`` before every start, and its VRAM
gate only judges the *next* start. Nothing watches the machine once the shards
run: a campaign whose scenes grow in RAM, or whose LLM saturates the CPU, sits
above the ceiling until the shards end on their own. This governor samples the
machine (``campaign_monitor.resources``) and moves the cap:

* **down by one** when the mean of any watched quantity -- CPU, RAM, VRAM, GPU
  compute, in percent -- over the last ``window`` samples exceeds ``limit_pct``;
* **up by one**, never past ``ceiling``, when every quantity has stayed under
  ``raise_below_pct`` for ``raise_after_s``;
* never more often than once per ``cooldown_s``, and never below ``floor``.

Lowering the cap stops new shards only; running work is never touched (a
paused renderer keeps its memory, ``shard_pool``'s first rule). A file holding
0 (an operator's drain) is left alone.

::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.jobs_governor \\
        --jobs-file runs/X/support/stage1_jobs --ceiling 18 --log runs/X/support/stage1_governor.csv

Standard library plus ``campaign_monitor.resources``.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from sparx_agency.tools.campaign_monitor.resources import CpuMeter, sample

WATCHED = ("cpu_pct", "ram_pct", "vram_pct", "gpu_pct")


def decide(current: int, ceiling: int, floor: int, means: Dict[str, float], limit_pct: float,
           calm_for_s: float, raise_after_s: float) -> int:
    """The next cap: one lower when a windowed mean is over the limit, one higher after a calm spell, else unchanged.

    Args:
        current: The cap in force.
        ceiling: The highest cap this governor may set (the stage's planned jobs).
        floor: The lowest.
        means: Windowed means of :data:`WATCHED`, in percent; a missing key is not judged.
        limit_pct: The ceiling on every mean.
        calm_for_s: How long every quantity has been under the raise threshold.
        raise_after_s: How long that must be before a raise.

    Returns:
        The cap to write (``current`` when nothing changes).
    """
    if any(value > limit_pct for value in means.values()):
        return max(floor, current - 1)
    if calm_for_s >= raise_after_s and current < ceiling:
        return current + 1
    return current


def _snapshot(meter: CpuMeter, disk: Path) -> Dict[str, float]:
    resources = sample(disk, meter)
    out = {"cpu_pct": resources.cpu_pct, "ram_pct": resources.ram_pct}
    if resources.gpu is not None:
        out["vram_pct"] = resources.gpu.memory_pct
        out["gpu_pct"] = resources.gpu.utilization_pct
    return out


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def run(jobs_file: Path, ceiling: int, log: Path, *, floor: int = 1, limit_pct: float = 90.0,
        raise_below_pct: float = 75.0, interval_s: float = 30.0, window: int = 4, cooldown_s: float = 900.0,
        raise_after_s: float = 1800.0) -> None:
    """Sample every ``interval_s`` and move the cap in ``jobs_file`` until killed; one CSV row per sample in ``log``."""
    meter = CpuMeter()
    samples = deque(maxlen=window)  # type: Deque[Dict[str, float]]
    last_change = calm_since = time.time()
    new_log = not log.is_file()
    with log.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_log:
            writer.writerow(("utc", "jobs", "decision") + WATCHED + tuple("mean_" + k for k in WATCHED))
        while True:
            time.sleep(interval_s)
            now = time.time()
            snap = _snapshot(meter, jobs_file.parent)
            samples.append(snap)
            means = {k: sum(s[k] for s in samples) / len(samples) for k in WATCHED if all(k in s for s in samples)}
            if any(snap.get(k, 0.0) >= raise_below_pct for k in WATCHED):
                calm_since = now
            current = _read_int(jobs_file)
            decision = "hold"
            if current is not None and current > 0 and len(samples) == window and now - last_change >= cooldown_s:
                target = decide(current, ceiling, floor, means, limit_pct, now - calm_since, raise_after_s)
                if target != current:
                    jobs_file.write_text("%d\n" % target)
                    decision = "%s %d->%d" % ("lower" if target < current else "raise", current, target)
                    last_change = calm_since = now
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), current, decision]
                            + ["%.1f" % snap[k] if k in snap else "" for k in WATCHED]
                            + ["%.1f" % means[k] if k in means else "" for k in WATCHED])
            stream.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--ceiling", type=int, required=True, help="The highest cap to set (the stage's planned jobs)")
    parser.add_argument("--log", type=Path, required=True, help="CSV: one row per sample, with the decision")
    parser.add_argument("--floor", type=int, default=1)
    parser.add_argument("--limit-pct", type=float, default=90.0)
    parser.add_argument("--raise-below-pct", type=float, default=75.0)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--cooldown-s", type=float, default=900.0)
    parser.add_argument("--raise-after-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if not 1 <= args.floor <= args.ceiling:
        parser.error("need 1 <= --floor <= --ceiling")
    run(args.jobs_file, args.ceiling, args.log, floor=args.floor, limit_pct=args.limit_pct,
        raise_below_pct=args.raise_below_pct, interval_s=args.interval_s, window=args.window,
        cooldown_s=args.cooldown_s, raise_after_s=args.raise_after_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
