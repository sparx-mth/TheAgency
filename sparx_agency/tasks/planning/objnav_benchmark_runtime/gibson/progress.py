"""Gibson's import path for the shared run progress and lean records (``..progress``); also its CLI (``run_benchmark.sh``)."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.progress import (  # noqa: F401
    LeanHeadlessObjNavAgent, ProgressEnv, ProgressTracker, describe_progress, describe_summary, human_duration,
    lean_episode_info, main)

if __name__ == "__main__":
    raise SystemExit(main())
