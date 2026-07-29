# turn_anticipation_rig — does leading the nose into a corner actually help?

An offline flight rig for the drift-PID controller's **turn anticipation** (yaw
lookahead, `core/planning/trackers/drift_pid/yaw_lookahead.py`). It flies the
real controller, with the tuning the real drone is flying, over routes the real
planner produces, with the feature off and then on, and prints what changed.

No ROS, no GPU, no simulator, no container. The whole comparison runs in about a
second; the survey routes add a few seconds of A*.

```sh
.venv/bin/python -m sparx_agency.tasks.planning.turn_anticipation_rig.compare
.venv/bin/python -m sparx_agency.tasks.planning.turn_anticipation_rig.compare \
    --survey --plot /tmp/turns.png
.venv/bin/python -m sparx_agency.tasks.planning.turn_anticipation_rig.compare \
    --no-yaw-bite            # the idealised airframe; see "Reading the table"
```

## Why this exists rather than a simulator run

The obvious way to watch a controller change is to fly it in Isaac Sim. That is
not available for this one: `tasks/planning/falcon_pegasus/` runs FALCON
complete and unmodified, and the trajectory there is flown by
`core/planning/trackers/reference_tracker_3d/`. Nothing in the Pegasus graph
ever constructs a `DriftPidFollower` — `drift_pid` is deployed only on the real
XTEND, through the ROS1 stack in `tasks/planning/falcon/`. A Pegasus run would
prove a great deal about FALCON and nothing whatsoever about this change. (The
one exception: `drift_pid/pid.py` is shared — `ReferenceTracker3D` builds its
three axis loops from it — so a change *there* is exercised by Pegasus and by
`falcon_pegasus/stub/check.sh`.)

So the rig models the part that matters instead, and is explicit about which
part that is.

## What is modelled, and what is not

`airframe.py` is a first-order velocity-tracking body — the same shape as the
controller's own closed-loop tests — plus the one thing those tests leave out
and that decides this particular comparison: **the yaw this airframe delivers
depends on the translation under it.** From the flight logs the controller was
tuned on (2026-07-21): ~11% of the commanded rate standing still, 30-68% while
translating forward, and *inverted* under a backward translation.

That coupling is the whole reason turn anticipation is worth having, so a rig
without it would hand a stop-and-spin the same free rotation a flying drone gets
and conclude the two manoeuvres are much closer than they are. `--no-yaw-bite`
turns it off, and quoting both numbers is the honest way to present the result.

Deliberately **not** modelled, and worth remembering when reading the output:

* **No obstacles and no collision.** Routes are planned clear and the controller
  is not being asked to avoid anything. A crab flies the drone sideways into
  space its forward camera is not looking at; this rig cannot tell you whether
  that space was empty.
* **No localization noise or latency.** Both are covered by the controller's own
  suite, which measures settled cross-track error under a 100 ms delay.
* **No roll-versus-yaw coupling.** The logs say YAW+ROLL is worse than
  YAW+forward but never say by how much, and inventing a number would flatter
  or punish the crab on no evidence. The crab therefore gets, if anything, a
  slightly easy ride here while the in-place spin is modelled at its measured
  worst.

## The routes

**Hand-built corners** (`routes.py`) are the controlled experiment: one turn, of
a known angle, with nothing else going on — a single 90, two corners a metre
apart (the pair that must *not* be anticipated as one), an S-bend, a gentle bend
the feature is supposed to leave alone, and a hairpin.

**Survey routes** (`--survey`) are the reality check: the same weighted A* the
FALCON stack plans with, on the committed 10 cm survey of the office
(`robots/PEGASUS/maps/office_alt0150cm.npz`), through the same trajectory
simplifier the stack runs before the follower sees a waypoint. Whatever corner
distribution that chain produces is the one the controller meets in the air.

The controller tuning is read out of `tasks/planning/falcon/config/mission.yaml`
at run time (`tuning.py`) rather than copied here, so the rig cannot quietly
fly last month's gains. The run says which set it used.

## Reading the table

```
route                          yla    secs   TURN    esc   spin   stop   xtrack  arrive   lead
right turn                     off    24.5    8.2    0.0    0.0    0.0    0.131   0.287      0
right turn                     ON     24.6    0.0    0.0    0.0    0.0    0.260   0.290     70
```

* `secs` — time to the goal. Expected to go **up**: a crab is capped by the weak
  lateral axis, so the stretch into each corner is slower than a cruise.
* `TURN` — seconds in the controller's TURN regime: pointed the wrong way,
  translation suppressed, station-keeping while the nose comes round. **This is
  the number the feature exists to drive down.**
* `esc` — seconds in an escape reflex. Must stay 0: a reflex here is a false
  alarm, and a false alarm teaches the planner a phantom obstacle.
* `spin` / `stop` — rotating with nothing under it / commanding no translation.
* `xtrack` — worst distance off the line. The nose is allowed to leave the leg;
  the body is not.
* `lead` — how far round the nose ever led. 0 with the feature off.

Result as of the tuning in `mission.yaml` (6 corners + 5 survey routes):
flight time **345.7 s → 351.7 s (+1.7%)**, time in TURN **83.1 s → 14.7 s**,
escapes **0 → 0**. Add `--no-yaw-bite` and the *same routes* cost **+13.5%**
(300.0 s → 340.6 s, TURN only 28.9 s → 2.9 s), because a drone that can rotate
freely on the spot loses much less by doing so. That gap *is* the value of the
feature, stated as a number — and it is only meaningful compared like for like,
so quote both figures from one route set. (The 6 corners alone give +5.3% and
+24.1%; the pairs are not interchangeable.)

`--plot` draws both tracks with the nose direction marked once a second, which
is the part a table cannot show: the classic run points along its track
throughout, and the anticipating one is visibly pointed round the corner while
still travelling down the corridor.

## Files

| File | Owns |
|---|---|
| `airframe.py` | The modelled drone, including the yaw/translation coupling. |
| `routes.py` | Hand-built corners, and real A* + simplifier routes off the survey. |
| `tuning.py` | The controller dials, read from the deployed `mission.yaml`. |
| `flight.py` | One flight, and the metrics that score it. |
| `compare.py` | The CLI: fly everything both ways, table, totals, plot. |

## What a green run here does and does not license

It says the manoeuvre is geometrically sound over a set of real corners
(including a corner-pair), that it does not trip the controller's own
stop-and-turn latch or its blockage reflexes, and roughly what it costs in time
on this airframe model.

It does **not** cover the republished route: the rig calls `set_path` once per
flight, so that case is pinned by
`core/planning/trackers/drift_pid/tests/test_yaw_lookahead.py` instead. And it
does not say the drone will not fly sideways into a chair — that question needs
the map, and the map needs a real flight.
