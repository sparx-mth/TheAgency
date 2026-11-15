"""The smoke cover planner moved to ``objnav_benchmark_runtime/smoke_plan.py`` (2026-11-11): it is every campaign's, not HM3D's.

This module stays so that ``hm3d.smoke_plan`` imports keep working; new code imports
:mod:`sparx_agency.tasks.planning.objnav_benchmark_runtime.smoke_plan`.
"""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.smoke_plan import *  # noqa: F401,F403
from sparx_agency.tasks.planning.objnav_benchmark_runtime.smoke_plan import cover_plan, describe  # noqa: F401
