"""Compatibility shim: the navmesh stair-connector reader now lives in ``habitat/``.
It reads scene structure from any habitat-sim navmesh and is not Gibson-specific,
so the shared simulator bridge imports it from
:mod:`sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors`.
Gibson tools and tests keep importing this name.
"""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors import *  # noqa: F401,F403
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors import (  # noqa: F401
    LEVEL_BIN_M, MIN_CONNECTOR_RISE_M, MIN_LEVEL_AREA_M2, MIN_LEVEL_SEPARATION_M, OFF_LEVEL_M,
    SAMPLE_SPACING_M, _anchor, _cluster_xy, enu_to_habitat, floor_levels,
    habitat_to_enu, scene_structure, scene_structure_from_pathfinder, stair_connectors, surface_samples,
    triangles,
)
