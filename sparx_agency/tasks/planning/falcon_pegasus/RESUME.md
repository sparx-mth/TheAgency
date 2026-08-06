# Where this was left, and what to do next

Working note for the "ten clean full-office explorations in a row" task. Delete
it when the streak is done and the README carries the conclusions.

## State

**0 / 10.** Three soak rounds, each stopped at attempt 1. The crashes that used
to end every flight are gone; what stops it now is the aircraft.

| round | image contained | ended | crashes | coverage |
|---|---|---|---|---|
| 1 | grid-index + LKH isolation + A* inflation | `stalled` | 1 (mapper `CHECK`) | 595 m³ |
| 2 | + raycast guard | `stalled` | **0** | 382 m³ |
| 3 | + tilt/throttle in the status line | killed mid-warm-up to free the machine | — | — |

Logs kept, videos stripped: `~/falcon_pegasus_recordings/soak_evidence/`.

## The open question, and the one measurement that answers it

Round 2 stopped at (−21.8, −5.2, 1.45) and never moved again. Two things are
already ruled out:

* **not boxed in** — the surveyed map is clear for 1.4 m in every direction
  there, at the height it was flying;
* **not lagging** — the split read `lag=0.00m xte=1.03m`, and zero lag with the
  whole gap cross-track is the signature of a reference whose *planned velocity
  is zero*: a trajectory endpoint, not a moving target.

So it sat one metre from a stationary reference, in free space, still tracking
the yaw plan, and produced no translation. That is either **a controller not
asking** or **an airframe not answering**, and they want opposite fixes.

`isaac/mission.py`'s status line now prints the commanded tilt and throttle
beside the error for exactly this. Fly one run and read a stalled line:

```
t=  50.0s pos=(-21.82, -5.23, 1.45) err=1.03m lag= 0.00m xte=1.03m tilt= ?.?deg thr=0.?? traj#69
```

* **tilt ≈ 0°** — the outer loop is not commanding a lean. Look at
  `TrajectoryTracker.update`: the position term should give
  `kp * clamp(1.0) = 2.0 m/s²` ≈ 11.5° of tilt, so something is cancelling it.
  Suspect the integrator, the saturation flag, or `follow=False` sticking.
* **tilt ≈ 10–12°** — the controller is asking correctly and PX4 is not
  delivering. Look at the attitude cut's PX4 parameters
  (`isaac/px4_exploration_params.ATTITUDE_CUT_OVERRIDES`) and whether
  `SET_ATTITUDE_TARGET` is being accepted; note the `MPC_*` tilt limits are
  bypassed in this mode, so a limit would have to be an `MC_*` one.

Do not guess between these. One stalled status line decides it.

## Resuming

```bash
docker start isaac-sim && sleep 25
cd sparx_agency/tasks/planning/falcon_pegasus
./soak.sh 6_whole_office 10          # ~25 min per attempt; stops at the first dirty one
```

The image `falcon-pegasus:noetic` already contains all four C++ patches — check
with:

```bash
docker run --rm falcon-pegasus:noetic bash -lc \
  'strings /catkin_ws/devel/lib/exploration_manager/exploration_node | grep -c "the LKH solver died on"'
```

## Two things that will waste your time if you do not know them

**Builds fail randomly on this machine.** `docker build` and even in-container
`catkin_make` die with gcc internal compiler errors and `ld terminated with
signal 11`, in a different translation unit each time, and succeed on retry —
the last build took three attempts. Always wrap the build in a retry loop, and
build via `docker run` + `docker commit` rather than `docker build` (see the
README's rebuild section). This looks like failing hardware; a memtest is
overdue.

**The coverage bar is an estimate, not a measurement.** `MIN_COVERAGE_M3=2200`
of a 2424 m³ box is ~91%, chosen on the assumption that full coverage fits the
20-minute budget. No flight has yet exceeded ~600 m³. If attempts start ending
`flight_timeout` with coverage plateauing well under the bar, the answer is a
longer budget — not another bug fix — and the bar should be re-derived from a
flight that actually ran to the end, rather than quietly lowered to whatever was
achieved.
