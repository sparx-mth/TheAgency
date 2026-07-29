# 005 - Navigate to a YOLO-detected labeled object (e.g. "barrel")

**Branch:** `feat/yolo-object-navigation` (not yet created — see Notes; work so far is on
the shared `create_devcontainer_daphna` checkout)
**Status:** in-progress
**Roadmap item:** "YOLO-detected object label + position (\"barrel\") as a FALCON navigation goal"

## Goal
Given a target label (e.g. "barrel"), detect it via YOLO in the live camera feed, compute its
real-world position, and drive FALCON to navigate there using the occupancy map for a safe
route — same click-to-fly pipeline as 004, but the goal comes from object detection instead
of a BEV click.

## Why
This is the natural next step after occupancy-aware click-to-fly (004): instead of a human
clicking a point, the system identifies *what* it's looking at and navigates to it.

## Steps
- [x] Locate the existing YOLO integration — turned out to be bigger than "confirm it runs on
      Rooster": XTEND already has a *complete* select-then-go mission stack built around
      `yolo_world_trt` (`run_object_mission.sh` + `config/mission.yaml` +
      `adapter/launch/object_mission.launch` — detector sidecar + ROS1↔ROS2 bridge + FALCON nav
      + `mission_director_node`), but it only runs real_drone.launch's XTEND path. Nothing
      Rooster-specific existed. Steps below reflect porting that stack, not building detection
      from scratch (see Notes for the restructure).
- [x] Fork the stack for Sphera/Rooster, same `_sphera`-duplicate convention as
      `sphera_drone.launch`/`run_falcon_sphera.sh`/`maps/sphera_jail.yaml`:
  - [x] `adapter/launch/object_mission_sphera.launch` — includes `sphera_drone.launch` instead
        of `real_drone.launch`; adds the 4 args that fork newly needs
        (`real_depth_path_topic`, `real_rgb_path_topic`, `sync_tolerance`, `max_interp_gap`)
  - [x] `config/mission_sphera.yaml` — `map: sphera_jail`, Rooster's `real_pose_topic`
        (`/R1/localization`) and camera intrinsics (540×360), sphera_jail's actual BEV/box
        bounds, `controller: multi_axis` (documented as moot — `sphera_drone.launch` hardcodes
        it regardless), goal/stage points near Rooster's real spawn, the 4 new forwarded keys
  - [x] `run_object_mission_sphera.sh` — defaults to `mission_sphera.yaml`/`sphera_jail`, calls
        `run_falcon_sphera.sh` + `object_mission_sphera.launch`
  - [x] Dry-validated via `mission_config.py` against the new launch file (exit 0, every key —
        including the 4 new ones — resolved correctly) and `./run_object_mission_sphera.sh
        --help` (exercises the full config-load path with zero side effects)
- [x] Get the detector sidecar itself actually running on this PC — was blocked entirely:
      `ultralytics` missing, `torch` broken (`libnccl.so.2`/`libcudnn.so.9` — orphaned cu12
      NVIDIA packages conflicting with torch 2.11.0's cu13 requirement, plus two cu13 packages
      with dist-info present but library files missing — an interrupted/corrupted earlier
      install). Fixed the venv (removed orphaned cu12 cluster, force-reinstalled the two broken
      cu13 packages, installed `ultralytics`), downloaded `yolov8s-worldv2.pt`, and ran
      `yolo_world_trt/build_all.sh s` end-to-end: ONNX export, exact backbone/head parity
      (0.00e+00), both TensorRT engines built on the RTX 4090. `./run_object_mission_sphera.sh
      --detector-only` then actually started the sidecar cleanly (TRT engines loaded, publishing
      `/object_approach/detections` @ 2Hz) — confirmed via live process check, not just files
      existing. `mission_sphera.yaml`'s `detector.model`/`weights_dir` corrected to match
      (`s`, `/home/user1/Downloads` — the previous default was `/home/user`, wrong username for
      this PC). Only `s` is built; `m`/`l`/`x` are not (mission.yaml/XTEND's own default is
      still `x`, untouched).
- [ ] Get a 2D detection (label + bounding box) reliably for a test object — **open question
      answered: no labeled object is known to exist in `sphera_jail` yet** (no room-mapper
      catalog for this map — see `objects_sphera.json`, a placeholder catalog with 2 fictional
      entries near spawn so the selection/staging/aim plumbing is exercisable even though
      nothing will actually be confirmed)
- [ ] Back-project the detection into a real-world 3D/2D position — reused as-is from the
      ported stack (`object_approach_node.py`), not yet exercised live
- [ ] Feed that position into the same goal-setting path the BEV click uses — reused as-is
      (the ported stack already does this via `mission_director_node` + `astar_planner_node`),
      not yet exercised live
- [ ] Handle the "no detection" / "label not found" case explicitly — reused as-is (the ported
      stack's sweep→give-up-land path), not yet exercised live
- [ ] Live test: start the stack (`./run_object_mission_sphera.sh`) against a live Sphera
      session and confirm the mission actually runs (arms, holds at goal, GO flies to stage,
      aims/sweeps) — the real object-detection part will not confirm anything until either a
      real object exists in the scene or a room-mapper catalog is generated for `sphera_jail`

## Open questions
- Is a "barrel" actually placed anywhere in the current `sphera_jail` scenario, or does the
  scenario need updating first to have a real target object? **Answered (2026-07-29): not
  known to exist.** No room-mapper sweep has been flown/processed for `sphera_jail`, so there
  is no real catalog — `objects_sphera.json` is a placeholder, not a real answer to this.
  Needs either an actual labeled object placed in the Sphera scene, or a room-mapping session
  (see `Demo_No4_XTEND_MapRoom/room_mapper`) to generate a real catalog for this map.
- Single-shot detection-then-navigate, or continuous re-detection while flying (in case the
  object moves or the first detection's position estimate was off)? Unchanged from the ported
  stack's existing behavior — not Sphera-specific, so not addressed by today's work.
- goal_x/goal_y/stage_x/stage_y in `mission_sphera.yaml` are a rough placeholder (spawn + a
  small fixed offset, mirroring XTEND's own numbers) — not tuned against `sphera_jail`'s real
  room geometry, since that isn't known either. Expect these need adjusting once actually
  flown.

## Notes
- 2026-07-29: `run_object_mission_sphera.sh` (inherited unchanged from `run_object_mission.sh`)
  sources `/opt/ros/humble/setup.bash`, which does not exist on this PC (it has ROS Jazzy) —
  the sourcing failure is non-fatal (`set +u +e` around it) and the detector sidecar started
  fine regardless (rclpy works without it), but this is worth fixing properly rather than
  relying on the silent tolerance if it ever causes a real problem.
- 2026-07-29: Reworked the Steps list mid-flight (see project-workflow policy) — the original
  steps assumed detection infrastructure needed to be located/confirmed piecemeal; it turned
  out XTEND already has the *entire* mission built (detector + bridge + FALCON nav + director),
  just never ported to Rooster/Sphera. Today's work was that port (config + two new launch/
  script forks), not building detection logic from scratch.
- 2026-07-29: Did **not** create/switch to the `feat/yolo-object-navigation` branch this entry
  names — another agent was concurrently active on the shared `create_devcontainer_daphna`
  checkout, and switching branches on a shared working directory without coordinating already
  caused that agent a real, confusing bug once this session (see
  `feedback_no_branch_switch_shared_workdir` memory). All new files from today are additive
  (new files only, zero edits to existing tracked files), so leaving them uncommitted on the
  current branch carries no real risk — branching/committing is left for the user to decide.
- 2026-07-29: Confirmed via `mission_control.py`'s "Rooster Falcon Adapter" service (the most
  up-to-date source of Rooster's actual live parameters, per the `fly-rooster-sphera` skill's
  own advice to check there over the skill doc) exactly which values to port: `real_pose_topic`,
  camera intrinsics, `bev_xmin/ymin/xmax/ymax`, `sync_tolerance`/`max_interp_gap`, and the
  `goal_x`/`goal_y` "3m from spawn" offset convention.

## Result
