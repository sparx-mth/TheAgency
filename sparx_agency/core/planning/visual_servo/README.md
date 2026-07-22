# Visual Servo

The control half of "lock onto a named object and approach it": confirm a target,
turn its tracked box into a body-frame velocity, recover when lost, and decide when
this node owns `/cmd_vel`. Pure, ROS-free, trajectory-free (reactive, not a
`Trajectory` tracker).

## Pipeline

```
RGB ─► detector ─► confirmation gate ─► tracker ─► servo ─► ControlCommand
                        (N frames)      (Track2D)   │
depth ──────────────────────────────────────► range│  (approach / terminal logic)
                                                    ▼
                              state machine  ◄─────  at_target / track_valid
                              (SEARCH/APPROACH/HOVER_LOCK/RECOVER)
```

The tracker feeding this lives in `core/mapping/tracking`; the detector in
`core/mapping/detection`.

## Components

- **`TargetConfirmationGate`** — the search→approach trigger. Pose-free: acquire
  once the detector reports the target in `n_confirm` consecutive frames above
  `min_score` (a `miss_tolerance` bridges one dropped frame). Label match is
  fuzzy-lite (exact / substring / shared token). Returns the best matching
  `Detection2D` per frame to seed the tracker. Deliberately simpler than the
  reference stack's world-frame aggregation + LLM matcher (those needed
  localization).

- **`VisualServoController`** — `bbox` (+ optional depth `range_m`) → REP-103 body
  velocity. Two modes:
  - `holonomic` (default): yaw-to-centre, forward gated by how centred we are,
    optional lateral crab + vertical centring — all at once. For platforms like
    the XTEND.
  - `yaw_forward_xor`: reference behaviour — pure-yaw XOR pure-forward with
    hysteresis + a brake tick per switch, for platforms that reject mixed
    forward+yaw Twists.

  Depth-aware: with a metric range it ramps forward on true distance and stops at
  `target_range_m`; without depth it falls back to the box-area fraction.
  **Success (`at_target`) = centred *and* close** — the mission's hover-lock
  condition. Stateless per-axis math is in `algorithm.py`; the controller owns only
  mode/hysteresis state + output EMA.

- **`ReSearchPolicy`** — where to look when the track is lost. Reads the last
  track's position + image-plane velocity to infer the exit side and actively yaws
  the camera back toward it (vs the reference stack's blind hover). A short
  `hold_before_search_s` lets an in-flight re-detection recover first; gives up
  after `max_search_s`.

- **`VisualApproachStateMachine`** — `SEARCH / APPROACH / HOVER_LOCK / RECOVER`.
  Decides per tick whether this node drives `/cmd_vel`: in `SEARCH` it stays
  passive so the existing A*/NavDP follower flies the route while the detector
  scans; on confirm+lock it hands control to the servo. Driven only by booleans
  the node already has (`confirmed`, `track_valid`, `at_target`) + a lost-timer —
  no clock, no I/O. `drive_cmd_vel` is True in every state but `SEARCH`;
  `reset_acquisition` fires on the RECOVER→SEARCH give-up edge so the node clears
  the gate + tracker.

Tuning lives in the `*Config` / `VisualServoParams` dataclasses (sign conventions
documented in `params.py`) — see them, not restated here.

## Usage

```python
from sparx_agency.core.planning.visual_servo import (
    TargetConfirmationGate, VisualServoController, VisualServoRequest,
    ReSearchPolicy, VisualApproachStateMachine, SEARCH,
)

gate  = TargetConfirmationGate(target="refrigerator")
servo = VisualServoController()
fsm   = VisualApproachStateMachine()
research = ReSearchPolicy()

# each control tick (node owns the tracker + detector cadence):
conf = gate.update(detections)                        # streak / confirmed / best
track = tracker.on_frame(rgb, stamp_s=t)              # from core.mapping.tracking
dec = fsm.update(conf.confirmed, track.valid, at_target, dt)

if not dec.drive_cmd_vel:
    pass                                              # SEARCH: follower flies the route
elif dec.mode == "RECOVER":
    cmd = research.command(track, dec.lost_for_s, W, H).command
else:                                                 # APPROACH / HOVER_LOCK
    res = servo.step(VisualServoRequest(track=track, intrinsics=intr, range_m=range_m, dt=dt))
    cmd, at_target = res.command, res.at_target
```

Compute `range_m` via `core/mapping/depth/depth_bbox_fusion.py`. `ControlCommand`
/ `Track2D` / `Intrinsics` are in `core/common/types`.
