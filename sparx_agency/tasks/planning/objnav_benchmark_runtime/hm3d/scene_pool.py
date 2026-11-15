"""The scene process pool moved to ``objnav_benchmark_runtime/scene_pool.py`` (2026-11-11): it is every campaign's, not HM3D's.

This module stays so that ``hm3d.scene_pool`` imports and ``python -m ...hm3d.scene_pool``
keep working; new code imports :mod:`sparx_agency.tasks.planning.objnav_benchmark_runtime.scene_pool`.
"""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.scene_pool import *  # noqa: F401,F403
from sparx_agency.tasks.planning.objnav_benchmark_runtime.scene_pool import (  # noqa: F401
    HEARTBEAT_FILES, STALLED_RETURNCODE, console_log_path, main, max_workers, run_concurrently, scene_heartbeat)

if __name__ == "__main__":
    raise SystemExit(main())
