"""The view-point geodesic moved to ``habitat/geodesic.py`` (2026-11-11): habitat-lab's measure, shared by HM3D and MP3D.

This module stays so that ``hm3d.geodesic`` imports keep working; new code imports
:mod:`sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.geodesic`.
"""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.geodesic import *  # noqa: F401,F403
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.geodesic import (  # noqa: F401
    ViewPointDistance, view_point_positions)
