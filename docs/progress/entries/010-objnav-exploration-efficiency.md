# 010 - ObjectNav exploration efficiency (spin, inspection and route churn)

**Branch:** `feat/objnav-grounded-vlm-nadav` (fix work; the user decides on a split)
**Status:** in-progress
**Roadmap item:** [ObjectNav exploration efficiency](../ROADMAP.md)

## Goal

The frontier/RPT ObjectNav policy spends its 500-action allowance moving toward
unmapped space: no multi-rotation spins after a room is swept, no camera
inspection that interrupts a committed route, frontier goals ranked by what they
are worth from where the robot stands, and a recorded reason every time a route
is replaced.

## Why

Video analysis of `runs/grounded_vlm_gibson_3x3_20260922T084950Z/.../Ranchester/
recordings/e7c4f2ad5402` (472 actions, target `toilet`, not found) measured from
`steps.jsonl`: 24 % of actions were idle in-place turns with no route, 17 % were
stair-inspection LOOK/TURN actions and a further 7 % were turns recovering the
heading an inspection had thrown away; only 28 % were `MOVE_FORWARD`. 54 routes
were adopted, 19 of them replacing a route that still had more than a metre to
run, and 13 new goals lay roughly behind the robot.

## Steps

- [x] Extract the per-step timeline from the recording and attribute every wasted
      action to a code path (supervisor stall clock, timer-driven inspection,
      size-only frontier ordering, single-goal-per-action planning).
- [x] Core supervisor: let the caller end a room the moment its own goal
      generator is exhausted (`frontier_exhausted`) or its planner refuses the
      transit goal (`route_failed`), instead of waiting on 30/90 s clocks.
- [x] Core frontier ranking: geodesic-distance and heading-aware utility over the
      same passable graph the arc weights use; unreachable clusters dropped.
- [x] Runtime `FrontierSweep`: bounded one-rotation room scan, several ranked
      goals tried per action, goal persistence keyed on "still informative",
      release-and-reselect on the same action.
- [x] Route memory records why the previous route was replaced.
- [x] Stair inspection only at a natural pause (no committed route), triggered by
      floor exhaustion, a stair detection or a long periodic cadence -- never a
      per-cooldown timer while following a path.
- [x] Regressions for each behaviour; existing exploration/runtime suites green.
- [x] README, CHANGELOG, LESSONS updated.
- [ ] Re-run the Ranchester episode and compare the action mix (user decides when).

## Open questions

- Whether a full in-place rotation on room arrival is worth 12 actions for the
  detector, or whether a shorter sector scan suffices (`room_scan_turns`).
- Whether descending stairs need any unprompted periodic inspection at all once
  the terrain module flags drop-offs itself.

## Notes

- 2026-09-22: The 009 batch is finished (no episode process running); the
  detector and Ollama services are still up and were left alone.
- 2026-09-22: `gibson/stair_diagnostic.py --spent-floor-budget` still forces
  portal selection immediately, but its camera inspection now waits for the first
  action without a committed route (the idle gate applies there too).

## Result

Shipped (code only, no re-run yet):

- `core/planning/exploration/object_search_supervisor.py`: `frontier_exhausted` →
  `EXHAUSTED` (productive) and `route_failed` → `UNREACHABLE`, both immediate.
- `core/planning/exploration/frontier_ranking.py` (new) + `room_costs.frontier_cluster_cells`
  shared with the size-ordered `in_room_frontier_goals`.
- `tasks/planning/objnav_benchmark_runtime/methods/frontier_sweep.py` (new) replaces the
  inline `_local_frontier/_search_state/_frontier` in `rpt_policy.py`; `SweepSettings`
  reported under `configuration()["frontier_sweep"]`, diagnostics under
  `episode_info()["frontier_sweep"]`.
- `route_memory.py`: `replaced`, `cleared:<reason>` stats, and `reusable()` keeps the clear
  reason; `rpt_policy._navigate` emits `route_replaced`.
- `camera_control.py`: `periodic_inspection_actions` (100) + `periodic_due()`;
  `multifloor_policy._inspect` with the idle gate and named triggers (`inspection_started`
  events).
- Tests: 5 new supervisor tests, `test_frontier_ranking.py` (8), `test_frontier_sweep.py`
  (13); `test_method.py` regression re-pointed at the sweep. Full runtime + core exploration
  suites: 443 passed (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/python -m pytest
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests
  sparx_agency/core/planning/exploration/tests`).

Expected effect on the analysed recording, by construction: the 114 idle turns collapse to
≤ 12 per swept room; the 10 inspections to ≤ 3 (allowance 140 + cadence 100 within 500
actions) plus any stair detections, none mid-route; no throwaway route on release; failed
A* goals no longer cost an action. Whether the target is then found is for the re-run.

CHANGELOG: `[Unreleased]` → Fixed/Changed (ObjectNav frontier explorer). LESSONS: "ObjectNav
spent half its actions spinning".

