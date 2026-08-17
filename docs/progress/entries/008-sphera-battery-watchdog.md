# 008 - Sphera battery watchdog

**Branch:** `feat/sphera-battery-watchdog`
**Status:** done
**Roadmap item:** "Automatic Sphera restart on dead battery (no Falcon/node reconnect logic)"

## Goal
Automate the one manual action the user does every time Rooster's battery in Sphera runs
out: exit the simulator and relaunch it. Nothing more — explicitly not the reconnect
cascade (`ros1_bridge`, `video_trigger.py`, `rooster_command_unit.py`, ground-truth
localization) documented in `.claude/skills/fly-rooster-sphera/SKILL.md`'s "After a Sphera
restart" section, and not `falcon`/the planner at all.

## Why
"Battery running out, manually exit and restart the simulator" was a purely mechanical,
repetitive step blocking comfortable iteration on anything else in Sphera. The user asked
for infrastructure to remove that step specifically, and — after two rounds of scoping
questions turned up a real ambiguity in the existing runbook (a numbered checklist vs. a
"FIRST RULE, no exceptions" Gotcha disagreeing on whether `falcon` needs recreating too) —
explicitly asked to keep this first piece narrow: restart Sphera, full stop, decoupled from
Falcon or any Rooster ROS node. Broader reconnect automation is left for a future entry if
requested.

## Steps
- [x] Found the existing headless restart mechanism already in place
      (`~/.sphera/sphera-restart.sh` — `docker compose down` / `up -d run_sphera`, already
      wired to a desktop icon) — no need to reverse-engineer manual GUI clicks, the
      automation primitive already existed.
- [x] Confirmed `RoosterState.msg`'s `percentage` field (flat `float32` in `[0, 1]`,
      `rooster_manager_interfaces`) as the battery source, matching the manual check in
      fly-rooster-sphera/SKILL.md (`docker exec it ... ros2 topic echo /R1/state`).
- [x] Wrote `sparx_agency/tools/sphera_battery_watchdog.py`: polls battery every
      `--poll-interval` (default 10s), triggers `sphera-restart.sh` at `--threshold`
      (default 10%), re-arms at `--recovery-threshold` (default 80%, since a fresh restart
      resets battery to ~99-100%). Handles "topic unreadable" (Sphera/`it` down) as a
      normal skip, not an error.
- [x] Added a "Sphera Battery Watchdog" `Service` entry to `mission_control.py`
      (`ROBOTICAN_SERVICES`, new `rooster_watchdog` group) plus a "Watchdog" section in the
      ROBOTICAN tab — reuses the existing card/Start/Stop/Logs/auto-restart-if-dies UI, no
      new Streamlit plumbing needed.
- [x] Smoke-tested: `read_battery_fraction()` returns `None` cleanly with Sphera/`it` down
      (confirmed live, no crash); `mission_control.py` still renders (HTTP 200, no
      exceptions in server log) with the new service card present.
- [x] Wrote `.claude/skills/sphera-battery-watchdog/SKILL.md` (gitignored `.claude/`, not
      part of this branch's commits) documenting the deliberately-narrow scope so a future
      session doesn't fold the reconnect cascade in without being asked.
- [x] **Live-tested end-to-end once Sphera was up** (2026-08-17). First pass caught two real
      bugs the dry run couldn't have found:
      1. `ros2 topic echo --once` isn't supported by this container's ROS2 Foxy CLI at all
         (silent argparse failure, not a "topic not up" case) — fixed to match
         fly-rooster-sphera/SKILL.md's own plain-streaming + `timeout` pattern.
      2. `sphera-restart.sh` alone does not reset the drone's battery — `R1` is a sibling
         container (spawned via the host Docker socket) that survives a `drone_simulator`
         bounce untouched. Fixed by force-removing `R1` before calling `sphera-restart.sh`.
         See LESSONS.md's 2026-08-17 entry for the full root-cause and the confirmed
         `docker inspect ... StartedAt` evidence.
      Second pass (with both fixes): `R1` respawned fresh (`StartedAt` after the restart),
      battery read 99%. Full loop confirmed working.
- [x] GUI walkthrough for entering the scenario after a restart, confirmed live with the
      user's correction: Continue → choose role (Manager, only enabled option) → Start →
      Standalone → **click the drone (`Rooster_1`) in the "Drones Assignment" panel** →
      Play. Skipping the drone-click step raises a "no operators associated" confirmation
      dialog instead of failing outright. Not automated as part of the watchdog itself yet
      at this point — done manually via `xdotool` + screenshots for this test, one-off. See
      `entries/009-sphera-gui-reentry-automation.md` for automating this step too.

## Notes
- This branch was based on `feat/falcon_exploration_sphera_nadav` rather than `main`,
  because `mission_control.py` (and the rest of the ROBOTICAN/Sphera stack this depends on)
  only exists on that branch line, not yet merged to `main`.
- `xdotool` was installed on the host (with the user's confirmation) to drive the post-restart
  GUI screens for this test. At the time this entry was written the watchdog itself did not
  depend on or invoke `xdotool` — that changed the same day, see
  `entries/009-sphera-gui-reentry-automation.md`.
