# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SPARX Agency — an autonomous drone/robot navigation stack (perception, mapping, planning, localization, multi-robot coordination). The Python package is `sparx_agency` (Poetry, Python 3.12). It targets both x86 and Jetson and integrates with ROS2 Humble, but the *core algorithms are deliberately ROS-free* so they can be tested and reused without a ROS environment.

## Environment & common commands

- Interpreter: use the project venv `.venv/bin/python` (Python 3.12). NOTE: `.agent/GEMINI.md` hardcodes `/home/daphnaa/venvs/ros_py310/bin/python` — that is a stale per-developer path, do not use it.
- **`core/` must stay Python 3.8-compatible.** Although the venv is 3.12, the FALCON ROS1/Noetic adapter imports `core` under Python 3.8 (it is mounted read-only at `/opt/sparx_agency` in the container — see `tasks/planning/falcon/run_falcon.sh`). So in `core` avoid 3.10+ syntax: no `@dataclass(slots=True)`, no `match`/`case`, no PEP 604 `X | Y` unions outside `from __future__ import annotations`. These pass the 3.12 tests but crash the Noetic nodes at import.
- Install (algorithms only): `poetry install` (uses `pyproject.toml`). `requirements.freeze.txt` is the full pinned environment including ROS/CV/torch deps.
- ROS2 setup: `./scripts/install_ros_deps.sh ./ros2_ws` then `source ./ros2_ws/install/setup.bash`. This clones+builds `apriltag_ros` (humble) for localization. Always `source /opt/ros/humble/setup.bash` before running any ROS2 node.
- Tests: pytest, no central config — run by path, e.g. `.venv/bin/python -m pytest sparx_agency/core/mapping/costmap/tests/test_potential_mapper.py`. A single test: append `::test_name`. Test files live next to the code in `tests/` subdirs or as `test_*.py`.
- `pyproject.toml` declares `benchmark`/`analyze` console scripts pointing at `core.planning.evaluation.*` — those modules do not exist yet (the `evaluation/` dir is empty); the scripts are aspirational.

## Architecture (the big picture)

The repository follows a very clear division to keep concerns separated:

1. **`sparx_agency/core/`** — The lean algorithmic core. Pure, ROS-free, and drone-agnostic. No `rclpy`/`rospy` or environment specifics belong here. **Keep this core as short and lean as possible: try to use existing external libraries for algorithms rather than writing them from scratch.** It is divided into 3 main components:
   - `core/mapping/` — the perception→map pipeline (see below). `interfaces/depth_model.py` defines the `DepthModel` ABC; `costmap/` holds occupancy/log-odds/probabilistic grids, SDF/distance fields, and the potential-field stack.
   - `core/planning/` — planners (`astar`, `rrtstar`, `informed_rrtstar`, `bitstar`), `smoothers` (hermite, minsnap), `trackers` (pure_pursuit), `behaviors`, `local_planners`, `safety`, and `exploration` (frontier, random_walk). Orchestrated by `planning/pipeline/planning_pipeline.py`.
   - `core/localization/` — AprilTag azimuth/triangulation.
   - *Note:* `core/common/types/` provides the shared vocabulary (`Pose2D`, `State3D`, `Path2D`, `Trajectory`, `ControlCommand`, `KinematicLimits`, etc.). Almost everything flows through these dataclasses.
2. **`sparx_agency/tasks/`** — Mission-level tasks we need to perform. These execute logic by activating and composing the components from `core`. Each task's `ros2/` subdir holds the ROS2 `Node` wrappers (e.g. `tasks/mapping/ros2/potential_mapper_node.py`).
3. **`sparx_agency/robots/`** — Specific drones or simulations we work with (`XTEND`, `ROBOTICAN`, `SJTU`, `common`). Each has its own camera configuration, sensors, and communication setup. Adapters translate platform telemetry/control into the core `types`. ROS lives at this boundary.
4. **`sparx_agency/demos/`** — Self-contained demos of our code, built around the tasks we perform.

`sparx_agency/agents/internnav_ros2_bridge/` is a standalone ROS2 package. `pre_baseline/` is archived prior work — generally not active code.

### Registry pattern

Behaviors, trackers, and smoothers are pluggable via registries (`*/registry.py`, e.g. `BehaviorRegistry.from_behaviors([...])` then `.get("go_to_pose")`). Register new implementations there rather than wiring them in by hand.

### Mapping/perception pipeline (most active area)

RGB → metric depth → point cloud → 2D occupancy grid → potential field → control vector. Details in `.agent/docs/ARCHITECTURE.md`. Key conventions:
- **Coordinate frame** (matches the C++ reference): `X` = Left, `Y` = Up, `Z` = Forward.
- Depth backends implement `DepthModel`: `DepthEngineTRT` (verify `trt_engine` exists before inference) and `DepthAnythingV2DepthModel` (HuggingFace, CPU/GPU). Output is HxW float32 meters.
- Accumulation is EMA over occupancy: `M_acc = (1-α)·M_acc + α·M_temp`.
- Potential field = Gaussian-falloff repulsion + parabolic goal attraction. Defaults: 10cm cells, height band 5cm–2.0m, σ=0.3m.

## Conventions & Rules

- **Code Style:** Keep the code clear, short, and as simple as possible.
- **Documentation:** Everything must be documented. Write high-quality, Google-style docstrings, maintain inline documentation files, and add a `README.md` in subdirectories if necessary for new developers.
- **Domain Standards:** Strictly maintain standard robotics and aviation conventions for drones.
- **Failures:** Prefer raising errors over silent fallbacks to default values.
- **Paths:** No hardcoded absolute paths — use `pathlib`/`os.path.join` relative to repo root.
- **Single Responsibility Principle (SRP):** Each file, class, and function must have a single, clearly defined purpose. Do not group multiple distinct algorithms or unrelated logic into a single file, even if the file is short. Break complex logic into separate, focused modules.
- **Refactoring:** PEP8. Prefer splitting a file before it passes ~300 lines and refactoring functions over ~50 lines.
- **Safety:** Ask before `rm -rf`, global package installs, or pushing to a remote.
- **Testing:** Run the relevant tests after changing code; any change to potential-field logic should be followed by a simulation/demo run.

## Git workflow

Feature branches are named `<short_feature>_<your_name>` (e.g. `baseline_nadav`, the current branch). Merge into `main` at milestones. See `pre_baseline/TheAgency-GitWorkFlow.md` for the full submodule workflow.

## Custom Rules

- **Language:** Always respond, write code, and generate comments/documentation in English only. Do not use Hebrew under any circumstances.
- **File Access Permissions:** You have explicit, pre-approved permission to read and analyze any files or directories located under `PycharmProjects/` and `GIT/` on the local machine. Do not ask for permission before reading files in these paths.
- **Editing & Output:** When writing or modifying code, strictly use the file-editing tools to apply changes directly to the files. Do not print code blocks, snippets, or diffs in your conversational text response.
- **Summarization:** At the end of the assignment, provide a high-level explanation of the changes made without printing any lines of code. Explain what changed conceptually.
- **Version Control & Commits:** Do not run `git commit` or `git push`; the user will handle version control manually. However, immediately after your high-level explanation, provide a short, concise suggested Git commit message summarizing the work you just completed.