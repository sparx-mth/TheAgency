# navigation — the narrow-passage layer

Doorways and tight corridors were the method's bottleneck on the HM3D
recordings of 2026-10-28: the failed episodes wedged in them, scraped the jambs
and painted open doorways shut on the occupancy grid; the successful ones
burned their steps oscillating and re-aligning through every opening. This
package is the answer, in four pieces that meet in `policy.py`:

```text
 2D route (planner, with the centreline layer)
      │
      ▼
 kinematic_follower.py   the route becomes the exact actions that walk it (planners/lattice), re-planned when the
      │                  agent leaves them, a fresh frame blocks them or a step is refused; the converter executes them
      ▼
 bottlenecks.py          where the route squeezes: channels (doorway / corridor), throat, axis, width, off the EDT
      │
      ├── scrape_guard.py      a refused step in a channel, aligned with it, is a scrape: NOT written on the map,
      │                        remembered as a refused pose for the lattice instead
      └── doorway_look_down.py the camera pitched on the approach to a channel; the plan through it re-derived on
                               the pitched frames (CameraController owner DOORWAY)
```

| module | holds |
|---|---|
| `settings.py` | `KinematicSettings` (the lattice, the corridor, strict tracking, re-anchoring, the lattice floor), `ScrapeSettings`, `LookDownSettings` -- `RPTSettings.kinematic`, `.doorway_scrape`, `.doorway_look_down`; with `RPTSettings.centreline_weight` and `.channel_half_width_m` every knob is in the frozen configuration |
| `bottlenecks.py` | `route_channels(points, clearance, ...)` → `Channel(kind, entry/exit/throat arc lengths and points, axis, width, door_id)`; `route_position`, `nearest_channel`, `heading_error` |
| `kinematic_follower.py` | `KinematicFollower.command(obs, world, planner, command)` verifies the state, re-anchors or re-plans, and attaches the plan suffix and `plan_pose`; `note_action` advances the cursor; `note_refused` and `invalidate` drop it |
| `scrape_guard.py` | `DoorwayScrapeGuard.judge(obs, world, route)` → the verdict, or None to write the map as before |
| `doorway_look_down.py` | `BottleneckLookDown.pitch_request(obs, world, route)` → the pitch to hold or None; `refreshed(obs)` fires once on the first pitched frame |

## 1. The centreline (medial axis) layer -- in the planner

`WeightedAStarParams.centreline_weight` (4.0 here, 0 for the aircraft
planner) adds `centreline_weight * (1 - d / channel_half_width_m)` to every
cell whose clearance `d` -- the exact Euclidean distance transform of the
confirmed obstacles -- is below `channel_half_width_m` (0.6 m: passages
narrower than 1.2 m, every doorway and no room). The existing soft band is a
gentle slope that told a doorway's centre and its jamb-side cell apart by a
tenth of a cell's cost, and an 8-connected route cut the corner through the
jamb; the new slope is steepest where the passage is tightest and its minimum
across any passage is the ridge of the distance transform, the medial axis.
See `planners/astar/cost_grid_2d.py`.

## 2. The kinematic follower

Every `_navigate` command -- the loop's transits, the frontier sweep, the
fallback, the closing's standoff approach -- goes through
`KinematicFollower.command`. On a new route it asks the lattice
(`planners/lattice`) for the action sequence from the agent's pose along the
route, at the standoff the route was planned at (its `inflate_used_m`), then
at the floor standoff when that fails; the command leaves with the actions and
the pose they start from, and the converter executes the first one. Every
step after that: the agent is where the plan says (within the converter's
plan tolerances) and every remaining forward step still clears the map in
force → the suffix is sent on; otherwise a fresh plan from where the agent is
(`off_plan`, `plan_obstructed`, `route_changed`). A refused step
(`notify_blocked`) and a safety veto (`notify_action` saw a turn where the
plan said forward) are remembered as refused poses, so the next plan does not
ask for the same step from the same place. When no plan exists -- the lattice
found none inside the corridor, or the last attempt from this route failed
fewer than `replan_backoff_actions` ago -- the command goes out as before and
the ladder walks it; the command's `info["kinematic"]` says which.

**Strict tracking** (the revision of 2026-10-28, after the first recorded
series): the executed trajectory is the plan, and the plan is the route.

- *The plan rides the medial-axis route.* The lattice corridor is 0.4 m
  (`corridor_m`, under the route memory's 0.5 m cross-track rule) and every
  forward step pays `route_offset_cost` (2 actions per metre) for its lateral
  offset from the 2D route, so the plan leaves the route's line only to line
  up a lattice heading through a doorway and returns at once. On the first
  recorded series the plan-driven steps already stood a median 5 cm (p90
  16 cm) off the route; the wide corridor's remaining liberty is gone.
- *The full state is verified every step.* The converter executes a planned
  action only at the plan's pose (`plan_position_tolerance_m` 2 cm,
  `plan_heading_tolerance_deg` 1); the follower checks the same before it
  sends the suffix. A heading the plan did not predict is a re-plan from the
  exact pose, which begins with the in-place turns the correction needs. A
  displacement of centimetres at the planned heading (the simulator's
  navmesh snap) keeps the plan: it is re-anchored on the true pose -- the
  remaining poses move with the agent, the same actions follow -- when the
  moved trajectory still clears the map (`resync_m`, 0.10 m; `resynced` in
  the stats).
- *No corner is cut by the fallback.* `strict_tracking` sets the converter's
  `strict_plan_tracking`: its ladder aims one forward step along the path
  (never the 0.5 m lookahead, which turned a step early at every kink) with
  no heading hysteresis, and a plan the agent has left the start of is
  answered with the turn toward the plan's heading before any step. The
  ladder is reached only when the lattice has no plan, and
  `replan_backoff_actions` is 0: the lattice is asked again every action.
- *The last standoff rung.* The lattice samples the continuous step every
  5 cm where the 2D planner joins cell centres; an 8-connected route that
  passes an obstacle corner diagonally is 0.224 m from it, lethal to the
  lattice at 0.25 m. The follower's ladder of standoffs is the route's own,
  the planner's floor, then `lattice_floor_m` (0.20 m: the body radius and a
  cell): on the first series' recordings 17 of the 24 known-map failures
  planned there. `squeezed` counts them; the plan's radius is in the
  command's `info["kinematic"]`.

Measured on the synthetic doorway rig: the ladder's rollout 184 actions and
53 turns on a 32 m route through three 0.8 m doorways; the plan 148 and 15
with the wide corridor, 156 and 25 riding the route (0.4 m corridor, offset
cost), at 0.30-0.32 m clearance through every doorway, in 5 ms.

## 3. The scrape guard

`policy.notify_blocked` asks the guard before `ObservedMap.notify_blocked`.
The guard reads the route's channels off the clearance field; when the point
one step ahead of the body (or the body) is within `reach_m` (0.75 m) of a
channel's throat and the heading is within `max_heading_deg` (40) of its
axis, the refusal is a scrape: the map is not told, the escape probe and the
goal-bump ledger are not charged, the route is cleared and re-planned (with
the refused pose excluded). A channel that refuses `scrapes_before_mark` (3)
alignments within `window_actions` (40) is closed after all: the next refusal
is written as before. Confirmed door landmarks within `door_match_m` of the
throat are named in the verdict; the geometry decides.

## 4. The look-down

Under the search owner, on a walk of the search's own (not the takeover's
approach, not a stair transaction, not a scan or a glance), when the route's
next channel entry is within `approach_m` (1.5 m) along the route and the
agent faces its throat, the camera is asked for `pitch_deg` (30: the floor
from half a metre out in the bottom of the image, the room still in the top)
and holds it to `exit_margin_m` past the channel's exit, at most
`max_hold_actions`. The first frame taken at that pitch drops the follower's
plan, so the entry is re-planned on the refreshed map -- the pre-entry
correction. A released channel is never asked for again (two LOOKs per
doorway, once), a walk interrupted by a hold survives two steps before the
request is released, and channels are matched by throat (within 0.6 m): the
smoke run of 2026-10-28 spent 178 of 500 actions on LOOKs before those three
rules.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/object_search/navigation -q
```

`tests/rig.py` builds a real policy (the package's `tests/fixtures.py`) on a
doorway world the planner, the follower and the guards share. Disable any
piece with `{"enabled": False}` in its settings; the method is then exactly
what it was before 2026-10-28 for that piece.
