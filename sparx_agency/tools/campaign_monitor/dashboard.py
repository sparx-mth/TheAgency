"""One screen showing where a NavDP run has got to, refreshed in place.

Both halves of a full NavDP build take hours and neither prints a percentage:
``collect.py`` logs one line per flight with no idea how many the campaign
wants, and ``train.py`` logs a metrics table with no notion of wall-clock
remaining. This joins them to a target and shows the answer people actually
want, which is how long is left.

It is a reader. Nothing here writes to a campaign or a run directory, so it can
be started and killed freely, and several copies can watch the same run::

    python3 -m sparx_agency.tools.campaign_monitor.dashboard \\
        --recordings ~/data/sim/office --episodes 2000 \\
        --run ~/data/navdp/world_goal/run1 --dataset ~/data/navdp/world_goal/dataset

``--once`` prints a single frame and exits, which is what to use from a script
or over a connection that cannot hold a live screen.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import List, Optional

from sparx_agency.tools.campaign_monitor import (
    bars, collection, coverage, resources, training,
)


class _CoverageCache:
    """Coverage, recomputed no more often than ``interval_s``.

    Measuring walks every recording's poses, which is seconds on a large
    campaign and far too slow for a screen that redraws every five.
    """

    def __init__(self, interval_s: float = 300.0) -> None:
        self.interval_s = interval_s
        self._value: List[coverage.SceneCoverage] = []
        self._at = 0.0

    def get(self, root: Path) -> List[coverage.SceneCoverage]:
        """The most recent measurement, refreshing when it has expired."""
        now = time.time()
        if now - self._at >= self.interval_s:
            self._value = coverage.measure(root, coverage.default_scene_maps())
            self._at = now
        return self._value


_coverage_cache = _CoverageCache()

CLEAR = "\033[2J\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

UTILISATION_TARGET = 80.0
"""The share of each resource this run is trying to keep busy."""


def _heading(text: str, width: int) -> str:
    """A bold section title with a rule out to the terminal's width."""
    return f"{bars.BOLD}{text}{bars.RESET} {bars.DIM}{'─' * max(0, width - len(text) - 3)}{bars.RESET}"


def render_collection(progress: collection.CollectionProgress, used_bytes: int,
                      cap_bytes: Optional[float], live_workers: int) -> List[str]:
    """The collection panel: flights done, outcome mix, disk against the cap."""
    lines = [bars.line("flights", progress.fraction,
                       f"{progress.done:,}" + (f" / {progress.target_episodes:,}"
                                               if progress.target_episodes else ""),
                       colour=bars.CYAN)]

    if cap_bytes:
        share = used_bytes / cap_bytes
        lines.append(bars.line("disk", share,
                               f"{bars.gigabytes(used_bytes)} / {bars.gigabytes(cap_bytes)}",
                               colour=bars.capacity_colour(100 * share)))
    else:
        lines.append(f"  {'disk':<12} {bars.gigabytes(used_bytes)}")

    rate = progress.rate_per_hour()
    landed_share = progress.landed / progress.done if progress.done else 0.0
    lines.append(bars.line("landed", landed_share,
                           f"{progress.landed:,} of {progress.done:,}",
                           colour=bars.GREEN if landed_share > 0.8 else bars.YELLOW))

    summary = [f"{live_workers} live worker(s)", f"{progress.frames:,} frames"]
    if rate:
        summary.append(f"{rate:.0f} flights/h")
    if progress.in_flight:
        summary.append(f"{progress.in_flight} in the air")
    lines.append(f"  {bars.DIM}{'  ·  '.join(summary)}{bars.RESET}")

    outcomes = progress.outcomes
    if outcomes:
        parts = [f"{name} {count}" for name, count in list(outcomes.items())[:6]]
        lines.append(f"  {bars.DIM}{'  '.join(parts)}{bars.RESET}")

    eta = progress.eta_seconds()
    if eta is not None:
        lines.append(f"  {bars.BOLD}ETA {bars.duration(eta)}{bars.RESET}"
                     f"  {bars.DIM}(collection){bars.RESET}")
    return lines


def render_coverage(scenes: List[coverage.SceneCoverage]) -> List[str]:
    """Per-building coverage — the number that decides when to stop collecting.

    Neither disk nor flight count says whether more flying is still worth
    anything; this does. Both bars stalling between checks means the campaign is
    producing near-duplicates.
    """
    if not scenes:
        return [f"  {bars.DIM}no recordings in a surveyed scene yet{bars.RESET}"]
    lines = []
    for scene in scenes:
        lines.append(bars.line(
            scene.scene[:12], scene.fraction,
            f"{scene.cells_seen:,}/{scene.cells_reachable:,} m²  "
            f"{scene.flights} flights  "
            f"headings {scene.mean_headings:.1f}/{coverage.HEADING_BINS}",
            colour=bars.load_colour(100 * scene.fraction, 80.0), width=24,
            label_width=13))
    return lines


def render_stages(stages: List[training.StageProgress]) -> List[str]:
    """The offline pipeline as six ticks, so a resumed run shows what it skipped."""
    done = sum(1 for stage in stages if stage.done)
    cells = []
    for stage in stages:
        mark = f"{bars.GREEN}✓{bars.RESET}" if stage.done else f"{bars.DIM}·{bars.RESET}"
        label = stage.name if stage.done else f"{bars.DIM}{stage.name}{bars.RESET}"
        cells.append(f"{mark} {label}")
    return [bars.line("stages", done / len(stages), "  ".join(cells), colour=bars.BLUE)]


def render_training(progress: training.TrainingProgress, running: bool) -> List[str]:
    """The training panel: steps, losses, and the two navigation metrics."""
    if not progress.exists:
        # A mistyped --run and a run that has genuinely not begun are the same
        # empty reading, and the first is far more common. Say which.
        if not progress.run_dir.is_dir():
            return [f"  {bars.YELLOW}no such directory{bars.RESET} {progress.run_dir}",
                    f"  {bars.DIM}check --run; nothing below it can be read{bars.RESET}"]
        return [f"  {bars.DIM}not started{bars.RESET} {bars.DIM}({progress.run_dir}"
                f" has no run.json){bars.RESET}"]

    total = f" / {progress.total_steps:,}" if progress.total_steps else ""
    lines = [bars.line("steps", progress.fraction, f"{progress.step:,}{total}",
                       colour=bars.CYAN)]

    if progress.total_epochs:
        lines.append(bars.line("epochs", min(1.0, progress.epoch / progress.total_epochs),
                               f"{progress.epoch:.2f} / {progress.total_epochs}",
                               colour=bars.BLUE))

    numbers = []
    if progress.train_loss is not None:
        numbers.append(f"train {progress.train_loss:.4f}")
    if progress.val_loss is not None:
        numbers.append(f"val {progress.val_loss:.4f}")
    if progress.best_val_loss is not None:
        numbers.append(f"best {progress.best_val_loss:.4f}")
    if progress.min_clear_m is not None:
        numbers.append(f"clearance {progress.min_clear_m:.2f} m")
    if progress.collide_pct is not None:
        numbers.append(f"in geometry {progress.collide_pct:.1f}%")
    if numbers:
        lines.append(f"  {bars.DIM}{'  ·  '.join(numbers)}{bars.RESET}")

    state = []
    rate = progress.steps_per_second
    if rate:
        state.append(f"{rate:.2f} steps/s")
    if progress.resumed_from_step:
        # Otherwise "elapsed 1h 36m" next to step 66,000 reads as a machine
        # three times faster than it is.
        state.append(f"elapsed {bars.duration(progress.wall_s)} "
                     f"since resuming at {progress.resumed_from_step:,}")
    else:
        state.append(f"elapsed {bars.duration(progress.wall_s)}")
    if progress.checkpoints:
        state.append(f"saved {', '.join(progress.checkpoints)}")
    if not running and progress.finished:
        state.append("finished")
    elif not running:
        state.append(f"{bars.YELLOW}no metrics for 5 min{bars.RESET}")
    lines.append(f"  {bars.DIM}{'  ·  '.join(state)}{bars.RESET}")

    eta = progress.eta_seconds()
    if eta is not None and running:
        lines.append(f"  {bars.BOLD}ETA {bars.duration(eta)}{bars.RESET}"
                     f"  {bars.DIM}(training){bars.RESET}")
    return lines


def render_resources(sample: resources.Resources) -> List[str]:
    """The machine panel, coloured so an under-used resource stands out."""
    lines = [
        bars.line("cpu", sample.cpu_pct / 100,
                  f"{sample.cpu_pct:.0f}%",
                  colour=bars.load_colour(sample.cpu_pct, UTILISATION_TARGET)),
        bars.line("ram", sample.ram_pct / 100,
                  f"{sample.ram_used_gb:.0f} / {sample.ram_total_gb:.0f} GB",
                  colour=bars.load_colour(sample.ram_pct, UTILISATION_TARGET)),
    ]
    if sample.gpu is not None:
        lines.append(bars.line("gpu", sample.gpu.utilization_pct / 100,
                               f"{sample.gpu.utilization_pct:.0f}%  {sample.gpu.name}",
                               colour=bars.load_colour(sample.gpu.utilization_pct,
                                                       UTILISATION_TARGET)))
        lines.append(bars.line("vram", sample.gpu.memory_pct / 100,
                               f"{sample.gpu.memory_used_mb / 1024:.1f} / "
                               f"{sample.gpu.memory_total_mb / 1024:.1f} GB",
                               colour=bars.capacity_colour(sample.gpu.memory_pct)))
    lines.append(f"  {bars.DIM}filesystem {sample.disk_free_gb:.0f} GB free{bars.RESET}")
    return lines


def frame(args, cpu_meter: resources.CpuMeter, size: resources.CachedDirectorySize) -> str:
    """One complete screen."""
    width = min(shutil.get_terminal_size((110, 40)).columns, 110)
    recordings = Path(args.recordings).expanduser()
    progress = collection.scan(recordings, target_episodes=args.episodes or None)
    used = size.get()

    out: List[str] = [
        f"{bars.BOLD}NavDP campaign{bars.RESET}  {bars.DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  ·  {recordings}{bars.RESET}",
        "",
        _heading("1  data collection", width),
    ]
    out += render_collection(progress, used, args.max_bytes or None,
                             collection.live_worker_count(recordings))

    if args.coverage:
        out += ["", _heading("1b  how much of each building has been seen", width)]
        out += render_coverage(_coverage_cache.get(recordings))

    if args.dataset:
        dataset = Path(args.dataset).expanduser()
        run_dir = Path(args.run).expanduser() if args.run else dataset.parent / "run1"
        out += ["", _heading("2  offline pipeline", width)]
        out += render_stages(training.pipeline_stages(run_dir.parent, dataset, run_dir.name))
        out += ["", _heading("3  training", width)]
        out += render_training(training.read(run_dir), training.is_running(run_dir))

    out += ["", _heading("4  machine", width)]
    out += render_resources(resources.sample(recordings if recordings.exists() else Path.home(),
                                             cpu_meter))
    return "\n".join(out)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", required=True,
                        help="the campaign directory the workers write into")
    parser.add_argument("--episodes", type=int, default=0,
                        help="flight target, for the percentage and the ETA")
    parser.add_argument("--max-bytes", type=float, default=0.0, help="the campaign's disk cap")
    parser.add_argument("--dataset", default=None, help="the labelled dataset directory")
    parser.add_argument("--run", default=None, help="the training run directory")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between redraws")
    parser.add_argument("--no-coverage", dest="coverage", action="store_false",
                        help="skip the per-building coverage panel, which walks "
                             "every recording and is the slow part")
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point: redraw the screen until interrupted."""
    args = _parse_args(argv)
    cpu_meter = resources.CpuMeter()
    size = resources.CachedDirectorySize(Path(args.recordings).expanduser(), interval_s=30.0)

    if args.once:
        print(frame(args, cpu_meter, size))
        return 0

    print(HIDE_CURSOR, end="")
    try:
        while True:
            print(CLEAR + frame(args, cpu_meter, size), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        print(SHOW_CURSOR, end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
