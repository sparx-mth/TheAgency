# 002 - Move all Rooster host-side nodes into robotican_dev

**Branch:** `worktree-rooster-containerize` (isolated worktree — a second agent was
concurrently working `feat/rooster-frame-jetson-relay` on the shared checkout; see Notes)
**Status:** done
**Roadmap item:** "Move every Rooster node still running on the bare host into `robotican_dev`/`theagency:robotican`"

## Goal
Nothing in the Rooster/Sphera pipeline runs on the bare host Python venv anymore — every
node either runs inside `it` (Sphera-side, ROS2/Foxy), `falcon` (FALCON, ROS1/Noetic), or
`robotican_dev` (`theagency:robotican`, ROS2/Humble). "Done" = a fresh bring-up needs zero
`/home/user1/GIT/TheAgency/venv/bin/python ...` invocations for anything Rooster-related.

## Why
Confirmed live 2026-07-28 during the bring-up checklist: `rooster_frame_dir_publisher.py`
still runs directly on the host venv, and (per the twist-control adapter's own docstring)
`rooster_twist_control_adapter.py` is *designed* to run there too. `rooster_depth_processor.py`
turned out to already be containerized (`robotican_dev`, confirmed by inspecting its actual
process tree) — good, that one's already model to copy. The host-venv path is exactly the
kind of "reproducible nowhere else" setup the dev-container plan (see `docs/plans` / prior
session) is meant to eliminate — a fresh machine or the Jetson has no way to reproduce it.

## Steps
- [x] Confirm `robotican_dev`'s image (`theagency:robotican`) already has whatever
      `rooster_frame_dir_publisher.py` needs — GStreamer/PyGObject, cv2, numpy, rclpy all
      verified importable; no image changes needed
- [x] Move `rooster_frame_dir_publisher.py` to run via `docker exec robotican_dev ...`
      (`run_rooster_frame_dir_publisher.sh` rewritten)
- [x] Same for `rooster_twist_control_adapter.py` — worked cleanly first try inside
      `robotican_dev`, no ROBOTICAN-message-dep issue materialized; wrote a new
      `adapters/run_twist_control_adapter.sh` (none existed before) and added its first-ever
      `mission_control.py` Service entry
- [x] Update `fly-rooster-sphera` skill's startup sequence (steps 5 and 11) — done, plus a new
      Gotchas entry for the TRT engine rebuild this surfaced (see below)
- [x] Update `mission_control.py`'s service definitions — Frame Capture and Depth Processor
      both got `proc_container="robotican_dev"` added (they never had it — status checks were
      silently checking the wrong place); Twist Control Adapter is a new entry
- [x] Full clean bring-up test — all three confirmed working via their actual wrapper scripts,
      not just raw `docker exec` commands: frame publisher's GStreamer pipeline starts cleanly,
      twist adapter subscribes and logs ready, depth processor loads its (rebuilt, see Notes)
      engine and infers continuously at ~12-14ms/frame. No host-venv Rooster process running.

## Open questions
- Was resolved, not open: `robotican_dev` IS already started via a tracked file
  (`docker-compose.robotican.yml` + `docker/bake.hcl`) — my initial grep for the container
  name missed it. The running `robotican_dev` container just predates today and nobody had
  pointed the frame/twist scripts at it yet.

## Notes
- 2026-07-29: `rooster_depth_processor.py`'s existing engine (`DA3METRIC-LARGE_fp16_546x364.engine`,
  built Jul 8) failed to deserialize inside `robotican_dev` — a TensorRT version mismatch, not a
  bug in this task's scope, but blocking it. Rebuilt from the existing ONNX inside
  `robotican_dev` itself (so the build matches its exact TRT version); old engine backed up as
  `.engine.bak_20260729` rather than deleted. Full detail + reusable build script in the
  `fly-rooster-sphera` skill's new Gotchas entry.
- 2026-07-29: this work happened in an isolated git worktree
  (`.claude/worktrees/rooster-containerize`) because a second agent was concurrently working
  `entries/006-rooster-frame-jetson-relay.md` on the shared checkout, and a branch switch on
  that shared checkout had already disrupted one concurrent task that day (see
  `feedback_no_branch_switch_shared_workdir` memory). Will need a normal merge back into the
  main line once both efforts are ready — `mission_control.py` in particular was being edited
  by both tasks and will need reconciling by hand, not a blind merge.

## Result
All three Rooster nodes that ran on the bare host venv (`rooster_frame_dir_publisher.py`,
`rooster_depth_processor.py` — which turned out to be only accidentally/manually containerized,
not really via a tracked path — and `rooster_twist_control_adapter.py`, which had no wrapper at
all before today) now run inside `robotican_dev` via proper `run_*.sh` wrappers, each with a
`mission_control.py` Service entry with correct `proc_container` status tracking. Fixed a
latent bug along the way: `run_video_trigger.sh`/`run_ground_truth_localization.sh` (pre-existing,
`it`-targeting wrappers) used `docker exec -it`, which fails outright under any non-interactive/
background launch (exactly how `mission_control.py` or a backgrounded shell would run them) —
dropped `-it` from all five wrapper scripts, new and old. Also rebuilt the DA3 TensorRT engine
to match `robotican_dev`'s current TRT version (10.15.1.29), unblocking depth processing
entirely. Not yet done: merging this worktree's changes back into the main line (pending the
concurrent Jetson-relay work also touching `mission_control.py`).
