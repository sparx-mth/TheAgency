# 009 - Automate Sphera's post-restart GUI re-entry

**Branch:** `feat/sphera-battery-watchdog`
**Status:** done
**Roadmap item:** "Fold Sphera's post-restart GUI walkthrough into the battery watchdog, unattended"

## Goal
Extend `sphera_battery_watchdog.py` (008) so the whole recovery — container restart AND
getting a flyable drone back — happens with no one at the keyboard: force-remove `R1` ->
`sphera-restart.sh` -> drive Sphera's own GUI (Continue -> Manager -> Standalone -> select
Rooster_1 -> Play) -> verify `R1`/battery actually came back, all inside one function call.

## Why
008 deliberately left the GUI walkthrough manual/undecided ("an open question for a future
entry"). The user then said explicitly they will not be at the computer when this watchdog
fires, so a human clicking through Sphera's welcome/role/scenario screens is not an option —
the whole thing needs to be self-contained. They also asked, separately, for the reboot cycle
to be *faster*: most of the perceived slowness in the 008 testing session turned out to be
conversational round-trips (screenshot -> read -> decide -> click, repeated per step), not
Sphera's own boot time — folding the sequence into one script call removes that overhead
entirely, independent of the unattended-operation requirement.

## Steps
- [x] Split the GUI-clicking mechanics into their own module, `sphera_gui_automation.py`
      (window-finding, proportional-coordinate clicking, the click sequence itself) — kept
      out of `sphera_battery_watchdog.py` to keep that file's one responsibility (battery
      polling + trigger logic) intact, matching this repo's SRP convention.
- [x] Window identification: `xdotool search --name Sphera` also matches unrelated windows
      (hit a live Chrome/Teams tab with "Sphera" in its title) — filtered to an exact
      (trimmed) name match, then to the smaller-by-area of the two exact matches (Sphera's
      own outer decorated frame vs. its inner content window), confirmed stable across
      multiple reboots.
- [x] Coordinates as fractions of the window's live geometry (not fixed screen pixels),
      queried fresh per click — cheap, and adapts to minor size differences without needing
      a screenshot round-trip per step.
- [x] First live end-to-end test of the fully automated path failed silently: `R1` never
      came back, no exception raised. Root-caused to a real race — the window exists before
      Sphera is actually rendering/accepting input, so every click fired against a
      still-loading screen. Full detail + fix (10s warmup + an idempotent redundant first
      click) in `LESSONS.md`'s second 2026-08-17 entry.
- [x] Added a structural (not pixel-level) success check in `sphera_battery_watchdog.py`:
      after the GUI sequence, poll for `R1` to exist AND report a readable battery before
      declaring success; `notify()` + log a clear failure otherwise. This is what actually
      caught the race-condition bug above instead of it silently passing as "done".
- [x] Re-tested twice more after the fix: both fully unattended runs succeeded end-to-end
      (real low-battery trigger -> verified fresh `R1` + restored battery), ~38s each,
      consistent.
- [x] `--no-gui-reentry` CLI flag added to `sphera_battery_watchdog.py` for the case where
      someone IS at the keyboard and wants the old (008) narrower behavior.
- [x] Built end-to-end timing into `restart_and_reenter()` itself (not just an ad-hoc test
      wrapper) — every real run now logs and notifies its own elapsed time, so drift is
      visible without re-measuring by hand. Confirmed via the actual CLI (not a test
      script): a real low-battery trigger through `sphera_battery_watchdog.py` logged
      `scenario re-entered, R1 fresh, battery restored — 35.8s end-to-end`, consistent with
      the two prior manual measurements (38.1s, 38.2s).
- [x] Startup check: hard-fails with a clear message if `xdotool` isn't installed and GUI
      re-entry wasn't opted out of (this repo's "raise, don't silently fall back" convention).

## Notes
- The 10s warmup and per-step settle delays are empirically tuned against this one machine's
  observed Sphera boot behavior, not a documented Sphera property — if re-entry starts
  failing on different/slower hardware, raise `_WINDOW_WARMUP_SEC` in
  `sphera_gui_automation.py` first before re-investigating from scratch.
- Explicitly out of scope, same as 008: this still does not touch Falcon, the ROS1 bridge, or
  any Rooster node. "Unattended" here means "gets Sphera itself back to a flyable drone",
  not "resumes whatever mission was running before the battery died."
