# 003 - Integrate incoming FALCON/planning update

**Branch:** `chore/falcon-planning-update` (or `fix/`/`feat/` once the actual diff is known)
**Status:** planning
**Roadmap item:** "Incoming updated FALCON/planning drop from the user — integrate and re-verify"

## Goal
The user will hand over an updated version of FALCON and/or the planning stack. Land it
cleanly and re-verify nothing from today's fixes (X/Y localization sign, rotation-freeze,
demo-mode handshake, exploration_node watchdog, video-freshness watchdog) regressed.

## Why
Today's session fixed several real, hard-to-reproduce bugs deep in this stack (see
`LESSONS.md`, entries dated 2026-07-28). An external update landing on top of them is the
single most likely way to silently reintroduce one — this entry exists so that re-verification
is a planned step, not an afterthought.

## Steps
- [ ] Get the actual update from the user and diff it against the current state before
      merging blind — identify anything it touches that overlaps today's fixes
      (`rooster_ground_truth_localization.py`, `sphera_drone.launch`, `nav_stack.launch`,
      `sphera_jail.yaml`, `mission_control.py`'s Rooster service, the `fly-rooster-sphera`
      skill's documented values)
- [ ] If it touches any of the above, reconcile explicitly rather than silently overwriting —
      today's X-axis fix in particular has a wide blast radius (map bounds, goal, BEV bounds,
      RViz view) that an external update won't know about
- [ ] Re-run the full bring-up checklist and re-verify: rotation-freeze doesn't stick
      (`gate=FUSING`), `exploration_node`/video watchdogs still apply if the update changes
      those nodes, a real "right" move still reports the correct (non-mirrored) sign
- [ ] Update `LESSONS.md`/the skill if the update changes any documented behavior

## Open questions
- Scope and exact contents of the update are unknown until it arrives — this entry is a
  placeholder for the re-verification process, not a real plan yet.

## Notes

## Result
