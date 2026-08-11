"""Separate the up-front allocation from whatever grows afterwards.

That split is the whole point of measuring. FALCON's voxel map is a dense array
sized once on the first tick, so if the map is the cost, the trace steps early
and then lies flat. Anything that keeps climbing after that step is something
else — the frontier list, the connectivity graph, the per-cell vectors — and it
is the slope, not the step, that decides whether a long flight survives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from sparx_agency.tasks.planning.falcon_pegasus.memwatch.sample import Sample

# Readings before this are ignored: the node is still starting, and the
# allocation itself is not instantaneous.
DEFAULT_SETTLE_S = 20.0

# A run that ends normally ends with the node letting go of its map, so the last
# readings fall off a cliff. Anything below this fraction of the peak at the END
# of a trace is teardown, not a measurement, and counting it turns a flat trace
# into hundreds of megabytes of "shrinkage".
TEARDOWN_FRACTION = 0.5

# Below this much measured flight, a slope is noise. The node's working set
# swings by tens of megabytes between planning cycles, so a short window fits
# that oscillation instead of any trend.
MIN_GROWTH_WINDOW_S = 120.0


@dataclass(frozen=True)
class Summary:
    """What a run's memory trace shows.

    Attributes:
        samples: How many readings had a value.
        duration_s: Span from first to last reading.
        startup_bytes: The plateau just after the node settles — the allocation.
        peak_bytes: The largest reading seen.
        final_bytes: The last reading.
        growth_bytes_per_min: Least-squares slope after the settle point.
        growth_total_bytes: Final minus startup.
        teardown_dropped: Trailing readings discarded as the node exiting.
    """

    samples: int
    duration_s: float
    startup_bytes: Optional[int]
    peak_bytes: Optional[int]
    final_bytes: Optional[int]
    growth_bytes_per_min: Optional[float]
    growth_total_bytes: Optional[int]
    teardown_dropped: int = 0


def _slope_per_minute(points: Sequence[Sample]) -> Optional[float]:
    """Least-squares bytes-per-minute through the readings.

    Args:
        points: Samples with a value, in order.

    Returns:
        The slope, or None if there is not enough spread to fit one.
    """
    if len(points) < 3:
        return None
    xs = [point.elapsed_s for point in points]
    ys = [float(point.rss_bytes) for point in points]
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return (covariance / variance) * 60.0


def summarise(
    samples: Sequence[Sample], settle_s: float = DEFAULT_SETTLE_S
) -> Summary:
    """Reduce a trace to the numbers worth reporting.

    Args:
        samples: Everything sampled, in order.
        settle_s: Ignore readings before this, and take the allocation plateau
            from the first reading at or after it.

    Returns:
        The summary. Fields are None where nothing could be measured.
    """
    valued: List[Sample] = [s for s in samples if s.rss_bytes is not None]
    if not valued:
        return Summary(0, 0.0, None, None, None, None, None)

    peak = max(s.rss_bytes for s in valued)

    # Trim the exit. Only from the end, so a genuine dip mid-run still counts.
    keep = len(valued)
    while keep > 1 and valued[keep - 1].rss_bytes < peak * TEARDOWN_FRACTION:
        keep -= 1
    dropped = len(valued) - keep
    valued = valued[:keep]

    duration = valued[-1].elapsed_s - valued[0].elapsed_s
    settled = [s for s in valued if s.elapsed_s >= settle_s] or valued[-1:]

    startup = settled[0].rss_bytes
    final = valued[-1].rss_bytes

    return Summary(
        samples=len(valued),
        duration_s=duration,
        startup_bytes=startup,
        peak_bytes=peak,
        final_bytes=final,
        growth_bytes_per_min=_slope_per_minute(settled),
        growth_total_bytes=final - startup,
        teardown_dropped=dropped,
    )


def _mb(value: Optional[float]) -> str:
    """Bytes as MB, or a dash."""
    if value is None:
        return "     --"
    return "{:7.1f}".format(value / (1024.0 * 1024.0))


def format_summary(summary: Summary, expected_grid_bytes: Optional[int] = None) -> str:
    """Render a summary for a terminal.

    Args:
        summary: What :func:`summarise` produced.
        expected_grid_bytes: What ``mapsize`` predicted the voxel grid would
            cost, so the step can be checked against it.

    Returns:
        Text with no trailing newline.
    """
    if not summary.samples:
        return "  no readings — was the container running?"

    lines = [
        "  samples   {:d} over {:.0f} s".format(summary.samples, summary.duration_s),
        "  startup   {} MB   (allocation, once the node settles)".format(
            _mb(summary.startup_bytes)
        ),
        "  final     {} MB".format(_mb(summary.final_bytes)),
        "  peak      {} MB".format(_mb(summary.peak_bytes)),
    ]

    if summary.growth_total_bytes is not None:
        lines.append(
            "  growth    {} MB after startup, {} MB/min".format(
                _mb(summary.growth_total_bytes), _mb(summary.growth_bytes_per_min)
            )
        )
        if summary.duration_s < MIN_GROWTH_WINDOW_S:
            lines.append(
                "            (over only {:.0f} s -- too short to mean anything. "
                "The working set swings by tens of MB between planning cycles, "
                "so a slope needs {:.0f} s or more.)".format(
                    summary.duration_s, MIN_GROWTH_WINDOW_S
                )
            )

    if summary.teardown_dropped:
        lines.append(
            "            ({:d} trailing reading(s) dropped as the node exiting)".format(
                summary.teardown_dropped
            )
        )

    if expected_grid_bytes is not None and summary.startup_bytes is not None:
        share = 100.0 * expected_grid_bytes / float(summary.startup_bytes)
        lines.append("")
        lines.append(
            "  the voxel grid alone was predicted at {} MB, which is "
            "{:.0f}% of the startup figure.".format(_mb(expected_grid_bytes), share)
        )
        if share < 40.0:
            lines.append(
                "  most of the startup cost is therefore NOT the voxel map — look "
                "elsewhere before shrinking the box."
            )

    return "\n".join(lines)
