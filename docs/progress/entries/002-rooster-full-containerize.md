# 002 - Move all Rooster host-side nodes into robotican_dev

**Branch:** `chore/rooster-full-containerize`
**Status:** planning
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
- [ ] Confirm `robotican_dev`'s image (`theagency:robotican`) already has whatever
      `rooster_frame_dir_publisher.py` needs (ROS2 Humble, no exotic deps expected — verify,
      don't assume)
- [ ] Move `rooster_frame_dir_publisher.py` to run via `docker exec robotican_dev ...`,
      matching the existing `rooster_depth_processor.py` invocation pattern
- [ ] Same for `rooster_twist_control_adapter.py` (currently not run in most sessions — verify
      it actually works inside `robotican_dev` once first used there, don't assume the
      docstring's "host-side, no ROBOTICAN message deps" claim survives the move unchanged)
- [ ] Update `fly-rooster-sphera` skill's startup sequence (steps 5 and 11) to reflect the new
      `docker exec robotican_dev` invocations instead of bare host commands
- [ ] Update `mission_control.py`'s service definitions for these two, if they exist there
- [ ] Full clean bring-up test end-to-end with zero host-venv Rooster processes running

## Open questions
- Does `robotican_dev` need any additional mounts/env for the frame publisher or twist
  adapter specifically (it already has `/tmp/rooster_frames`, `/tmp/rooster_depth`,
  `~/.cache/sparx_agency` — check if anything else is missing before assuming it "just works")?
- Is `robotican_dev` itself started by a tracked script, or was it brought up manually and
  never captured anywhere? If the latter, that itself needs fixing before this entry is done.

## Notes

## Result
