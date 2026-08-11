"""Set the exploration area in five numbers, and know what it costs.

    from sparx_agency.tasks.planning.falcon_pegasus.mapsize import expand_run, load_run

    expanded = expand_run(load_run(Path("runs/6_whole_office.yaml")))
    print(expanded.cost.megabytes)

From a shell:

    python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml
"""
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.area import (
    Box,
    ExplorationArea,
)
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.expand import (
    ExpandedRun,
    expand_any,
    expand_map,
    expand_run,
    load_and_expand,
    load_any,
    load_map,
    load_run,
    write_expanded,
)
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.memory import (
    GridCost,
    grid_cost,
    implicit_resolution,
)
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.report import format_report

__all__ = [
    "Box",
    "ExpandedRun",
    "ExplorationArea",
    "GridCost",
    "expand_any",
    "expand_map",
    "expand_run",
    "format_report",
    "grid_cost",
    "implicit_resolution",
    "load_and_expand",
    "load_any",
    "load_map",
    "load_run",
    "write_expanded",
]
