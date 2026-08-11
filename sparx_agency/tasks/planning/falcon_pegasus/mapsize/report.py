"""The run's map geometry and memory bill, as a few lines on a terminal.

Printed before the container starts. The point is that the number which decides
whether the run survives is visible at the moment it can still be changed.
"""
from __future__ import annotations

from typing import List

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.expand import ExpandedRun


def _si(value: float) -> str:
    """A voxel count as a short human string."""
    if value >= 1e9:
        return "{:.2f}G".format(value / 1e9)
    if value >= 1e6:
        return "{:.1f}M".format(value / 1e6)
    if value >= 1e3:
        return "{:.0f}k".format(value / 1e3)
    return "{:.0f}".format(value)


def _bytes(value: float) -> str:
    """Bytes as MB or GB, whichever reads better."""
    if value >= 1024 ** 3:
        return "{:.2f} GB".format(value / 1024 ** 3)
    return "{:.0f} MB".format(value / 1024 ** 2)


def format_report(expanded: ExpandedRun, detailed: bool = False) -> str:
    """Render the geometry and cost of an expanded run.

    Args:
        expanded: The result of expanding a run file.
        detailed: Also break the memory down by array.

    Returns:
        Text with no trailing newline.
    """
    area = expanded.area
    cost = expanded.cost
    box, grid = area.box, area.map

    lines: List[str] = []
    for name, geometry in (("box", box), ("map", grid)):
        size = geometry.size
        lines.append(
            "  {:<5} {:6.1f} x {:6.1f} x {:5.1f} m   = {:>8.0f} m3".format(
                name, size[0], size[1], size[2], geometry.volume
            )
        )
    lines.append(
        "  {:<5} {:6d} x {:6d} x {:5d}     = {:>8} voxels @ {:.2f} m".format(
            "grid", cost.shape[0], cost.shape[1], cost.shape[2],
            _si(cost.voxels), cost.resolution,
        )
    )
    lines.append("  {:<5} {}".format("RAM", _bytes(cost.total_bytes)))

    if detailed:
        lines.append("")
        for name, total, note in cost.breakdown():
            lines.append(
                "        {:<16} {:>9}   {}".format(name, _bytes(total), note)
            )

    for warning in expanded.warnings:
        lines.append("")
        lines.append("  note: " + warning)

    return "\n".join(lines)
