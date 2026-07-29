"""Drawing a progress bar in a terminal, and the units that go beside one.

Kept apart from what it draws so the readers in :mod:`collection` and
:mod:`training` stay free of formatting, and so a caller that wants the numbers
without a terminal — a log line, a JSON dump — can have them.

Bars use eighth-block characters, which give a 200-column terminal eight times
the resolution of whole blocks and degrade to something still readable in a
terminal that lacks the glyphs.
"""
from __future__ import annotations

from typing import Optional

_EIGHTHS = " ▏▎▍▌▋▊▉█"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def bar(fraction: Optional[float], width: int = 40, colour: str = "") -> str:
    """A ``width``-character bar for ``fraction`` in ``[0, 1]``.

    ``None`` renders as a dotted rule: an open-ended campaign has real progress
    but no denominator, and showing it as 0 % would be a lie.
    """
    if fraction is None:
        return DIM + "·" * width + RESET
    fraction = max(0.0, min(1.0, fraction))
    total_eighths = int(round(fraction * width * 8))
    full, remainder = divmod(total_eighths, 8)
    full = min(full, width)
    body = "█" * full
    if remainder and full < width:
        body += _EIGHTHS[remainder]
    return f"{colour}{body.ljust(width)}{RESET if colour else ''}"


def load_colour(percent: float, target: float = 80.0) -> str:
    """Green at or above the utilisation target, yellow near it, red when idle.

    The convention is deliberately the opposite of a health dashboard's: this
    run is trying to *use* the machine, so a busy GPU is the good outcome and an
    idle one is the problem worth colouring red.
    """
    if percent >= target:
        return GREEN
    if percent >= target * 0.6:
        return YELLOW
    return RED


def capacity_colour(percent: float, limit: float = 90.0) -> str:
    """Red as a bounded resource — disk, VRAM — approaches its limit."""
    if percent >= limit:
        return RED
    if percent >= limit * 0.85:
        return YELLOW
    return GREEN


def duration(seconds: Optional[float]) -> str:
    """``2d 04h``, ``3h 12m``, ``4m 05s`` — two units, widest first."""
    if seconds is None:
        return "--"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, second = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {second:02d}s"
    hours, minute = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minute:02d}m"
    days, hour = divmod(hours, 24)
    return f"{days}d {hour:02d}h"


def gigabytes(count: float) -> str:
    """Bytes as GB or TB, whichever reads better."""
    gb = count / 1e9
    return f"{gb / 1000:.2f} TB" if gb >= 1000 else f"{gb:.1f} GB"


def line(label: str, fraction: Optional[float], right: str, width: int = 34,
         colour: str = "", label_width: int = 12) -> str:
    """One labelled bar with a right-hand caption, aligned to a column grid."""
    percent = "  --" if fraction is None else f"{100 * fraction:3.0f}%"
    return (f"  {label:<{label_width}} {bar(fraction, width, colour)} "
            f"{percent}  {right}")
