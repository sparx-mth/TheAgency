# `recovery` — stuck detection & back-out recovery (controller-agnostic)

The drone keeps getting stuck in doorways. The controller flying it does not
notice, because from its side the command went out fine — the failure is in the
*world*: it told the drone to go and the drone did not go, usually because it
clipped a door frame or pinned itself against something the camera never saw.

This package is the missing feedback loop, wrapped around **any** follower:

1. **Notice** — a command was issued and the pose did not follow.
2. **Recover** — take the command away and fly a short, open-loop back-out
   ("exit the wall in the other direction").
3. **Replan from the real position** — once clear, the drone is off its planned
   trajectory, so ask the planner for a fresh route starting from *where the drone
   actually is*, and mark the spot it clipped so the new route avoids it.

Everything here is pure, ROS-free and Python-3.8 / numpy-1.17 safe, so it imports
cleanly inside the Noetic FALCON container alongside the rest of `core`.

## The three primitives

| file | class | job |
|---|---|---|
| `stuck_detector.py` | `StuckDetector` | Over a trailing window, compare the distance the pose actually moved *along the commanded direction* with the distance commanded. Below a fraction for enough consecutive ticks ⇒ stuck. Forward and yaw tracked separately (a drone wedged one way can often still move the other). |
| `escape_maneuver.py` | `EscapeManeuver` | A scripted, timed reflex: **brake → back off → (probe sideways) → settle**. Capability-aware: the lateral probe is skipped for a one-axis follower that cannot command `linear.y`, leaving the universal back-off. |
| `recovery_supervisor.py` | `RecoverySupervisor` | Ties the two together into one per-tick `RecoveryDecision`: pass through, own the command with an escape, or (once a recovery concludes) raise `request_replan`. |

## Why "along the commanded direction"

Projecting the displacement onto the command, rather than taking the raw distance
moved, is the whole trick. A drone shoved sideways by drift while its forward axis
is blocked would otherwise look like it is making progress. The projection sees
that none of the motion went the way it was told to go. (This is the same idea
`drift_pid` uses — see below.)

## Frame convention

Body **FLU (REP-103)**: `vx` forward, `vy` left, `wz` CCW — the frame every FALCON
follower already commands in. An escape only ever drives `vx` (reverse) and `vy`
(probe); it **never rotates** (`wz == 0`), because the axis it is escaping may be
the yaw axis itself.

## How it wires into FALCON

`tasks/planning/falcon/adapter/scripts/waypoint_follower_node.py` builds a
`RecoverySupervisor` for every controller **except `drift_pid`** (which carries its
own reflex — see below) and, each control tick, before stepping the follower:

```
recov = supervisor.update(raw_pose, last_published_vx, last_published_wz, dt,
                          pose_trustworthy=<pose fresh?>, frozen=<held?>)
```

* It is fed the **raw** pose, never the estimator's command-propagated one — a
  pose advanced by the drone's own commands would mask a stuck drone.
* It is fed the command **published last tick** — that is the command whose effect
  this tick's pose reflects.
* `frozen=hold` covers every "the drone was told to stop" case (no GO, lost
  localization, mid-turn re-observe), so a held drone that keeps "trying" cannot
  invent an obstacle and box itself in.

Then:

* `recov.override` → publish `(vx, vy, wz)` straight to `/cmd_vel` and **return**,
  freezing the follower for the duration of the back-out.
* `recov.request_replan` → publish a `geometry_msgs/PointStamped` at the **real**
  pose on `/falcon/blockage`.

### "Replan from the real position" — the end-to-end path

The blockage report reuses the pipeline `drift_pid` already established:

```
follower  --/falcon/blockage (real pose)-->  astar_planner_node._blockage_cb
    → BlockageMemory.add()                 (remember the spot)
    → BlockageMemory.stamp() + confidence  (mark it lethal on the map)
    → _forced_replan()                     (replan NOW, start = self.pose = live odometry)
```

Because the planner already starts every plan from its live odometry pose, and the
escape has just backed the drone out to a clean spot, the new route is planned from
where the drone *actually is* — not from the stale point it should have reached on
the trajectory, which is exactly what was asked for. Marking the clipped spot also
nudges the next route away from that doorway edge, toward the middle.

## Relationship to `drift_pid`

`drift_pid` (`core/planning/trackers/drift_pid/`) already contains a tighter
version of this loop — `blockage.py` + `escape.py` — wired into its confidence
model (`LocalizationQuality`, `cmd_effectiveness`). This package is its
**controller-agnostic sibling** for the four controllers that have no such reflex
(`waypoint`, `multi_axis`, `pure_pursuit`, `roll_assist`). Two deliberate
differences:

* **Interface.** This one takes a plain `pose_trustworthy` flag instead of the
  whole `LocalizationQuality`, so it does not depend on drift_pid internals and can
  wrap any follower.
* **When it escalates.** drift_pid only reports to the planner once its reflexes
  are fully *spent* on a pinned drone. This supervisor reports after **every**
  concluded recovery, because re-centring the route from the recovered pose — and
  teaching the planner the spot it clipped — is what stops the drone clipping the
  same doorway edge twice.

**Run one or the other, never both on the same command.** The node enforces this by
building the supervisor only when `controller != drift_pid`.

## Tuning (rosparams, via `nav_stack.launch`)

| launch arg | node param | default | meaning |
|---|---|---|---|
| `recovery` | `~recovery_enabled` | `true` | master on/off (ignored by `drift_pid`) |
| `recovery_window_s` | `~recovery_window_s` | `1.2` | progress window (s) |
| `recovery_progress_frac` | `~recovery_progress_frac` | `0.30` | below this achieved/commanded ratio ⇒ "not moving" |
| `recovery_confirm_ticks` | `~recovery_confirm_ticks` | `5` | consecutive bad ticks to confirm |
| `recovery_brake_s` / `recovery_back_s` / `recovery_back_speed` | same | `0.4` / `0.7` / `0.10` | escape brake, then reverse for `back_s` at `back_speed` |
| `recovery_probe_s` | `~recovery_probe_s` | `0.8` | sideways probe (holonomic/lateral controllers only) |
| `recovery_settle_s` | `~recovery_settle_s` | `0.5` | hold still after, for a clean re-observe |
| `recovery_max_attempts` | `~recovery_max_attempts` | `1` | back-outs per episode before escalating to a replan |
| `recovery_pose_max_age_s` | `~recovery_pose_max_age_s` | `0.5` | a pose older than this is not stuck evidence |

`recovery:=false` restores the old behaviour (no detection, no escape, no report).

> **Airframe caveat.** The back-out commands **reverse** `vx` (and, on holonomic
> platforms, lateral `vy`). This mirrors `drift_pid`'s deployed escape, so reverse
> is known-good on the XTEND. On a new platform, confirm the airframe accepts a
> reverse velocity in its current flight mode before trusting the reflex, and fly a
> simulator/demo run after any change to the escape — the unit tests prove the
> *logic*, not that a given aircraft flies it.

## Tests

```
pytest sparx_agency/core/planning/recovery
```

Covers each primitive plus one end-to-end episode: a drone driven into a wall it
cannot see is noticed, backed out, and reported exactly once for a replan.

