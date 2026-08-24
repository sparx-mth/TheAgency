# sjtu_internvla_n1 — InternVLA-N1 flying the SJTU Gazebo drone

InternVLA-N1 controls the SJTU warehouse drone in Gazebo, the **NavDP way**: the
policy answers with a body-frame trajectory, we anchor it in the world and fly it
as a route, and a separate follower tracks it into the drone's one control input.
Nothing here is N1-specific below the wire; swap the policy node for the NavDP one
and the follower would not notice.

**Everything runs on the CPU except the InternVLA-N1 network, which owns the GPU
(~8 GB) alone.** That is the whole design constraint, and it is now enforced
rather than assumed. Both ROS2 nodes are pure numpy with
`CUDA_VISIBLE_DEVICES=""`, and **Gazebo renders on the CPU** — `bringup_world.sh`
passes no `--gpus` and forces llvmpipe.

That last part is not a preference. Gazebo Classic has no GPU physics, but it
renders its camera sensors through OGRE, and with `--gpus all` that context
landed on the same card the server was holding at ~7.2 of 8.1 GB. Asking for a
GL context in the remaining ~900 MB **hard-locked the host** — twice. Software
rendering turned out to be *faster* here anyway, because it stops competing:
RGB went from 5.6–9.0 Hz to 12.5–14.9 Hz and depth from 4.0–5.0 to ~6.3 Hz on
this 32-thread box, with the card reading 13 MiB before the server starts.

`SJTU_SIM_GPU=1` gives the simulator the card back, for a machine with a second
GPU or no VLA resident. Do not set it on this one.

## The chain

```
Gazebo — SJTU no_roof_small_warehouse (Docker, ROS2 Humble, domain 20, CPU)
  /simple_drone/front/image_raw        (RGB 600×600)
  /simple_drone/front_depth/.../image_raw (depth 32FC1 m)
  /simple_drone/odom                   (pose + body twist)
        │
        ▼   (host ROS2, CPU, CUDA hidden)
  n1_policy_node ──HTTP──►  InternVLA-N1 server  (GPU, conda `internnav`, ~8 GB)
   • InternVLAN1Policy.step(obs, LanguageGoal) → body-frame trajectory (T,≥2 FLU)
   • PlanCommitExecutor: anchor at the capture pose, commit ~half, re-infer
   • publishes /simple_drone/n1/trajectory   (nav_msgs/Path, world frame)
        │
        ▼   (host ROS2, CPU)
  trajectory_follower_node
   • PurePursuitTracker3D pursues the path → world-frame velocity
   • rotate world→body, clamp with the SJTU velocity adapter
   • DepthProximityBrake caps the FORWARD component on raw depth
   • refuses to command a capsized airframe at all
   • publishes /simple_drone/cmd_vel   (geometry_msgs/Twist, body FLU)
```

The route is drawn on the hospital's **own occupancy map** — ground truth built
from the world's collision meshes, not a flight — so a recording shows which
corridor the drone went down and which rooms it never entered. See
"The map the route is drawn on" below.

This is FALCON's `navdp_click_node → path → waypoint_follower_node` split, in
ROS2, with no ROS1 bridge because N1 is already ROS2.

## Where the trajectory actually comes from

InternVLA-N1's System 1 predicts a fan of candidate trajectories per call, which
`core/.../internvla_n1/trt/postprocess.mean_path` (upstream: `vln_utils.py::
traj_to_actions(..., use_discrate_action=False)`) integrates into one body-frame
XY curve — the *exact* shape NavDP emits. **That continuous curve is what this
stack flies.** It is `S1Output.trajectory`, carried to the HTTP response by the
`trajectory` patch in `tasks/planning/vlas/internvla_n1/upstream/` (see the README
there); `InternVLAN1Policy` reads it and prefers it over the discrete action.

Stock InternVLA-N1 discretizes that curve into a single VLN-CE action
(STOP / FORWARD / TURN_LEFT / TURN_RIGHT) and returns only that — which is why
the patch exists. **Apply the upstream patch and restart the model server**, or
the server emits no `trajectory` field and the policy falls back to the discrete
action (below).

On a **pure-S2 step** — a turn or a look-down, where the model genuinely emits
no curve — there is no continuous trajectory, so the policy renders that one
discrete action as a short followable body step
(`geometry.trajectory_from_action`): a forward action advances 0.25 m, a turn
places a waypoint one step ahead bent by **30°** — VLN-CE's own turn granularity,
and far enough off the nose to cross the follower's `stop_turn_rad`, so the
aircraft actually rotates instead of creeping 0.25 m sideways. The follower keeps
moving and the next S1 step returns to flying the curve.

In practice most steps are this fallback, for the reason set out in "What
decides between a curve and a discrete action" below: System 2 answers with an
arrow more often than with coordinates, and System 1 runs only on the
coordinate branch. A recording that shows `2 pts, 0.25 m` commitments
interleaved with `17 pts, 1.4 m` ones is the system working; the ratio between
them is the number worth watching, and it is drawn on the video.

## What decides between a curve and a discrete action

This is the question the deployment lives or dies on, and the answer is one
`if` in the model. `internvla_n1_policy.py::s2_step` ends:

```python
if bool(re.search(r'\d', self.llm_output)):   # the VLM wrote digits
    output.output_pixel  = <pixel goal>
    output.output_latent = self.model.generate_latents(...)   # -> System 1 runs
else:
    output.output_action = self.parse_actions(self.llm_output)  # -> no System 1
```

**System 1 never runs without a latent from System 2, and System 2 only makes a
latent when its text reply contains a number.** The prompt it is given asks for
exactly that:

> *"You are an autonomous navigation assistant. Your task is to `<instruction>`.
> Where should you go next to stay on track? Please output the next waypoint's
> coordinates in the image. Please output STOP when you have successfully
> completed the task."*

Its whole vocabulary for the other branch is `STOP`, `↑`, `←`, `→`, `↓`
(indices 0, 1, 2, 3, 5). So three observations that look like separate
mysteries are one mechanism:

* **Why so many turns?** Because System 2 replied `→→→→` instead of coordinates
  — four right-turns queued from one call. It does that when it cannot name a
  point in the current frame worth flying to: a wall filling the view, or a
  corridor it has to rotate into before anything navigable is visible.
* **Why is there only a pixel goal in forward flight?** Because the pixel goal
  *is* the forward branch. A coordinate reply means "fly to this point"; a
  turn, a look-down or a stop produces no pixel goal at all. The patched agent
  keeps the last goal alive on screen while System 1 flies toward it, which is
  why the ring is visible exactly during the forward legs.
* **Why does it say STOP so often?** It does not. Measured over 168 System-2
  replies, System 2 emitted a real `STOP` **zero** times. What the screen used
  to call STOP was action index **-1**, which the agent emits while a look-down
  is in progress and whenever System 1 returns an empty action list. It means
  "no decision this tick", not "the task is finished" — see the next section.

### The look-down is the prelude to every curve

The measurement that matters, over 168 System-2 replies in the hospital:

| System 2 replied | share |
|---|---:|
| pixel coordinates → **System 1 runs** | 30.4% |
| `↓` look-down | 32.7% |
| `→` / `←` / `↑` | 36.9% |
| `STOP` | **0%** |

and **92.7% of look-downs are immediately followed by coordinates**. `↓` is not
a failure or a stall: it is the trained prelude to a pixel goal, and it is very
nearly a promise of one. The model asks to tilt its camera down, then names the
waypoint.

Two consequences. First, this stack used to decode that `↓` as STOP and reset
the plan executor — **it braked the aircraft on the one step that precedes
every continuous trajectory**, which is about the worst possible place to put a
spurious stop. Second, the SJTU drone's camera is fixed, so the look-down is
never actually performed; the model gets the same forward frame again and
names a waypoint in it. That it still works 92.7% of the time is luck we are
living on, and it is the most promising thing left to improve — the airframe
does publish a downward camera on `/simple_drone/bottom/image_raw`.

### The look-down, performed

System 2 asks for a lower view before ~93% of its pixel goals, and computes
those coordinates **in that lower frame**. This airframe's camera is bolted
forward, so the stack used to hand it the same cruise-altitude frame and let it
name a waypoint in a view it believed was different.

It now **dips instead**, using the same forward camera from lower down — the
airframe does carry a downward camera, and it is deliberately not used: it looks
straight down, which is not the view the model was trained on either. The
sequence is: PATCH 6 puts a `look_down` flag on the wire (the action index
cannot carry it — the agent overwrites the look-down action with `-1`, which is
also what an empty System-1 list reports); the policy node asks the follower for
a 0.5 m altitude offset, waits until the aircraft is actually down at 0.70 m,
sends exactly that one frame, and clears the offset.

Two details that are load-bearing. The offset is **ramped**, not stepped —
a 0.5 m step puts the altitude error straight past `altitude_release_m` and the
follower drops out of route tracking to go up or down, turning every look-down
into a hover at each end. And the arrival test is **one-sided**: a symmetric
window goes true the instant the aircraft enters it, which sent the frame after
0.35 m of a 0.50 m dip.

Measured over a 110 s flight: four clean dips to 0.68–0.72 m, each returning to
the 1.20 m cruise, and the only ground contact is the landing.

### What a commitment costs, and how long it is allowed

Two settings decide whether the aircraft flies or waits, and both were wrong.

**`commit.fraction` is now 1.0.** NavDP's half-commitment exists because NavDP
re-infers at 3 Hz and its far half is extrapolation; InternVLA-N1 here needs
2.6–6.8 s per decision, so committing half a curve threw away the half the
aircraft had time to fly. Published routes went from 17 waypoints to **33** —
the whole prediction — and arc lengths from ~1.5 m to 2.5 m.

**The expiry deadline is now sized by the route.** A flat `max_commit_s` has to
be long enough for the longest curve, which makes it ten times too long for the
0.25 m step that is most commitments: one whose arrival never registers sits out
the whole ceiling. At `max_commit_s: 12` that was five twelve-second stalls in a
ninety-second flight. `expected_speed_mps` turns the deadline into
`arc / speed + commit_grace_s`, capped by `max_commit_s` — 2.6 s for a 0.25 m
step, 8.2 s for a 2.5 m curve.

And `follower.goal_tolerance_m` has to be small enough for arrival to *fire*:
`_reason` needs `peak_arc >= commit_arc - min(arrive_radius_m, 0.25 * commit_arc)`,
so on a 0.25 m route it needs 0.1875 m flown. A follower halting 0.10 m short
left 0.153 m and the commitment could only ever expire. It is 0.06 now.

Measured on the same 90 s atrium leg, cumulatively:

| | moving | ground track | replan reasons |
|---|---:|---:|---|
| half plan, flat 12 s deadline | 42% | 6.4 m | mostly *took too long* |
| full plan, route-sized deadline | 56% | 10.7 m | 17 flown / 4 expired |
| + arrival actually firing | **77%** | **10.8 m** | **23 flown / 2 expired** |

### Getting more curve and less action

`policy_params.sys2_max_forward_step` (default 8) is the knob. It is how many
System-1 steps the agent plays out before calling System 2 again, and since a
System-2 call is very nearly one chance at a curve, it is how many discrete
steps separate two chances. Lower is more continuous and slower — System 2 is
~98.5% of the per-decision budget.

The instruction is **not** the knob, which is worth saying because it is the
obvious guess. Counterbalanced A/B, same start pose every arm, order reversed
on the second pass, ~21 System-2 calls per arm:

| instruction | coordinate replies |
|---|---:|
| "explore the entire hospital, find every room, enter and exit every room" | 28.6% |
| "Walk forward down the corridor, pass the reception desk…" | 27.3% |
| "Go to the doorway ahead of you and stop in front of it." | 45.0% |
| "Head for the seating area on the far side of the room." | 27.3% |

The concrete-referent instruction trended higher, but at these counts the
difference is not significant (Fisher's exact against the other three pooled,
p ≈ 0.17). **An open-ended exploration order is not why the output is mostly
discrete.**

### Why the route is short and the aircraft keeps waiting

Measured off two recorded runs' bags (54 decisions, 2962 `cmd_vel` samples):

| published committed route | share |
|---|---:|
| 2 waypoints — the 0.25 m step rendered from a discrete action | **80%** |
| 17 waypoints — the committed half of a System 1 curve | 20% |

So four commitments in five are 0.25 m long, because that is what upstream's
`step_size` means by one forward action. Even a real curve is halved before it
flies, by `commit.fraction: 0.5`.

That has a second-order consequence that is easy to miss and was costing more
than the first: the pursuit's `slow_down_distance` defaulted to **1.0 m**, which
is longer than 80% of the routes, so the aircraft was inside the braking zone
for the *whole* of every action step and never exceeded 0.19 m/s against a
0.40 m/s cruise — measured mean while moving, 0.12 m/s. `slow_down_distance_m`
is now 0.15 and the command reaches full cruise.

**The aircraft is stationary about 43% of the time, and none of it is a STOP.**
It flies its 0.25 m in about a second and then waits for the next decision, and
System 2 takes 2.6–6.8 s to produce one. That is the honest shape of the
system: a 7B VLM on an 8 GB card is the clock. Raising the commanded speed
helped less than the arithmetic suggests (52% → 57% moving, 0.12 → 0.14 m/s)
because the airframe's own lag — 0.181 s delay, tau 0.51 s — means it barely
begins accelerating inside 0.25 m.

The levers, in order of effect: `commit.fraction` (fly more of each curve),
`sys2_max_forward_step` (more curves, fewer decisions per second), and
`policy_params.step_m` — though raising the last one makes the aircraft travel
further than the model asked for, which is a different thing from making it
follow the model better.

## -1 is not STOP

`INDEX_TO_ACTION.get(index, "STOP")` is the obvious way to decode an action, and
it is wrong here: it turns every index the map does not know — including the -1
the agent emits seventeen times in five flights — into a request to stop. This
stack used to act on that, resetting the plan executor and publishing an empty
route, so **a routine look-down abandoned a route the aircraft was halfway
through flying** and it braked for nothing.

`core/planning/vlas/internvla_n1/types.py` now names every index the agent can
emit (`NO_ACTION` -1, `LOOK_DOWN` 5) and `NON_TERMINAL_IDLE_INDICES` marks the
two that carry no motion and are not a stop. `InternVLAN1Policy.step` reports
them as `metadata["idle"]` with `stop=False`, and the policy node's rule is
**keep flying the current commitment** — the plan already in the air is still
the best one there is. Only index 0, System 2 literally answering `STOP`, resets
it.

`metadata["from_curve"]` says which producer each decision came from; the policy
node publishes it and a running `curve_share_pct` on `/simple_drone/n1/info`,
and the recorder draws both on the camera panel ("S1 CURVE" in green against
"action step" in orange). That number is how you read a recording and know what
you actually got.

## Layering (nothing here breaks it)

| layer | what | where |
|---|---|---|
| policy | wire contract + the body-frame trajectory | `core/planning/vlas/internvla_n1/` (`policy.py`, `geometry.py`, registered `"internvla_n1"`) |
| task | the ROS2 nodes, the follower glue | this package |
| robot | topics, camera intrinsics, actuation limits | `robots/SJTU/` + the binding YAML |

The policy never names SJTU and the SJTU robot layer never names N1; they meet in
`robots/SJTU/config/vla/internvla_n1.yaml`, which both nodes read via their
`config_file` parameter. The follower reuses the platform's own
`robots/SJTU/adapters/velocity_command.py` for the body-twist clamp, so the sign
and saturation logic lives once, next to the drone.

## Running it

```bash
export SJTU_PROJECT_DIR=~/GIT/sjtu_project   # the external sim checkout
export DISPLAY=:1                            # Gazebo Classic needs an X display

# one command: GPU preflight → Gazebo warehouse (CPU) → wait for N1 server (GPU)
#            → the two CPU nodes → takeoff → instruction
sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/run_sjtu_n1.sh \
    no_roof_small_warehouse "go to the far shelves and stop"
```

The script **refuses to start unless the GPU is empty** (`check_gpu_free.py
--require-empty`), gives the card to N1, and pins everything else off it. It does
not vendor the simulator or the model server:

* **Gazebo** comes up via `robots/SJTU/setup/bringup_world.sh` (Docker, CPU). Set
  `START_SIM=0` to manage the world yourself in another terminal.
* **The InternVLA-N1 server** must be runnable on the GPU. The script waits for it
  at `127.0.0.1:8087`; start it yourself (conda env `internnav`) or hand the
  script `N1_SERVER_CMD='<command>'` to start it. `~/GIT/InternNav` is not assumed.

Verify the split at any time:

```bash
python3 sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/check_gpu_free.py \
        --allow internnav --allow python     # only N1 may hold the card
nvidia-smi                                    # everything else is CPU
```

Redirect the drone mid-flight:

```bash
ros2 topic pub --once /simple_drone/navigation/instruction std_msgs/msg/String \
    "{data: 'turn around and go back to the door'}"
```

### Running the nodes alone

With the world and the server already up, and `ROS_DOMAIN_ID`/`RMW_IMPLEMENTATION`
matching the sim (domain 20, and the DDS profile of "Middleware" below):

```bash
ros2 launch sparx_agency/tasks/planning/sjtu_internvla_n1/launch/sjtu_internvla_n1.launch.py
```

## Recording a run

```bash
# hospital world, the exploration order, recording on. Ctrl-C to stop, or set a duration.
sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/record_run.sh
RECORD_SECONDS=180 sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/record_run.sh \
    hospital "Explore the entire hospital, enter all the rooms, reach every area at least once"
```

It produces two artifacts and prints the measured S1/S2 FPS at the end:

* **an MP4** — a two-panel video: the **drone camera on the left** (with the
  instruction, the action, the System-2 pixel goal and the System-1/System-2 FPS
  drawn on) and **N1's route top-down on the right** (the committed route in
  yellow, the speculative tail in orange, the trail it has flown in green). The
  top-down view is the honest way to show a *drone's* route — an aircraft flies
  at camera height, so a ground path projects to the horizon in first person and
  only reads clearly from above. Written with OpenCV's `mp4v` encoder, so no
  system `ffmpeg` is needed.
* **a rosbag** — every relevant topic, for replay and offline analysis.

The rendering is ROS-free (`recording.py` for the camera panel, `top_down.py`
and `map_backdrop.py` for the route on the map) and testable; see the format
without any of the stack:

```bash
python -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.demo_recording \
    --output /tmp/sjtu_n1/demo.mp4 --seconds 12
```

Adding the recorder to a manual bring-up takes three launch arguments, not one:

```bash
ros2 launch .../sjtu_internvla_n1.launch.py record:=true \
    record_output:=/tmp/sjtu_n1/run.mp4 record_seconds:=90.0
```

`record_seconds` is the one that matters and the one that is easy to miss: the
recorder closes its own file at that deadline, which is what makes the MP4
playable even when the shutdown does not go to plan. It must be written with a
decimal point — the parameter is a DOUBLE, and `record_seconds:=90` is parsed as
an INTEGER and rejected before the node starts. `0.0` records until stopped.

## FPS: System 1 and System 2

The two systems run at very different rates, and the split is the point of the
design — System 1 is the small, fast trajectory policy; System 2 is the 7B VLM.

**Live:** the `trajectory` patch to the InternNav agent also times each system
(`s1_ms`/`s2_ms` in the response). The policy node turns them into a smoothed
rate, publishes them on `/simple_drone/n1/info`, draws them on the recorded
video, and logs a line every 5 s:

```
[n1_policy_node] N1 FPS  System1=22.8 Hz  System2=1.4 Hz  (action=MOVE_FORWARD)
```

**Measured (this machine, `~/trt/internnav/REPORT.md`, RTX 5070 Laptop, sm_120):**

| | before | after (TensorRT S1) |
|---|---:|---:|
| **System 1 alone** | 6.77 Hz (p50 147.6 ms) | **22.99 Hz** (p50 43.4 ms) |
| **Whole dual-system pipeline** | 1.36 Hz (734.5 ms/decision) | **1.41 Hz** (707.9 ms/decision) |

System 2 is **~98.5% of every decision's time** and, because it is autoregressive
(a Qwen2.5-VL-7B behind a KV cache) and 16.6 GB at bf16, it is deliberately *not*
TensorRT-converted — so making System 1 3.4× faster moved the whole pipeline only
1.04× (the Amdahl ceiling with System 1 free is 1.05×). Cadence: System 2 fires
once every `sys2_max_forward_step` (8) System-1 steps, so per control decision
System 1 runs ~0.25× and System 2 ~0.125×.

> The live number reflects whatever the running server actually uses. If it
> serves the torch System 1, expect ~6–7 Hz; the 22.99 Hz is the TensorRT S1 from
> the optimization workspace. Either way the pipeline is System-2-bound at ~1.4 Hz.

## The map the route is drawn on

`robots/SJTU/maps/hospital.{pgm,yaml,npz}` — a 544x1182 grid at 5 cm, origin
(-13.60, -36.10), in **the same world frame `/simple_drone/odom` reports**. It is
built by `tasks/mapping/gazebo_world_occupancy` straight from `hospital.world`:
every model's collision mesh is transformed into the world, clipped to the
0.30-2.00 m band the drone can hit, and rasterised. It is therefore *ground
truth*, not an explored map — there are no unknown cells and nothing depends on
having flown anywhere first, which is what makes it usable as a backdrop from
the first frame of the first recording.

Validated against the live simulator: 100.00% of back-projected depth returns
land within one cell of an occupied cell, median distance 0.000 m, and it is not
mirrored, shifted, rotated or scaled. Rebuild it with the command in
`robots/SJTU/maps/README.md`.

## The two reflexes, and why they exist

Neither is optional in this world, and both were added after a recording proved
the point.

**The depth brake.** N1 decides *where* to go; nothing in this stack decides
whether the way is clear, and unlike `falcon_sjtu` there is no map in the
control loop. `core/planning/safety/depth_proximity_brake` takes the minimum
depth inside the corridor the airframe sweeps and turns it into an allowed
forward speed; the follower clamps only the forward component with it, so the
aircraft can still slide and climb out of trouble. Tuned under `brake:` in the
binding YAML.

**The capsize guard.** The SJTU plugin thrusts along body z. Past about 35
degrees of roll or pitch it cannot climb, translate or yaw — while still
reporting `FLYING` and a healthy 30 Hz of odometry. Every command is silently
ignored and **no topic says so**. Measured here: an aircraft that clipped the
reception desk lay at roll -83 degrees for the rest of a 60 s recording while
the policy cheerfully committed sixteen routes to it and the follower published
a twist on every axis. `/simple_drone/reset` does **not** right it; only
restarting the world does. The follower now reads attitude off odom, stops
commanding, and logs `CAPSIZED` — which is also what `record_campaign.sh` greps
for to fail a run.

## Recording a campaign: five areas, five recordings

All the commands below are written from the repo root.

```bash
sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/record_campaign.sh   # 5 areas, 60 s each
SCRIPTS=sparx_agency/tasks/planning/sjtu_internvla_n1/scripts
$SCRIPTS/record_campaign.sh 90 atrium south_hall   # 90 s each, two named areas
REPEAT=4 $SCRIPTS/record_campaign.sh               # four passes over the five areas
REPEAT=0 $SCRIPTS/record_campaign.sh               # keep cycling until stopped
```

`REPEAT=0` is the unattended mode: as one recording ends the next begins, world
restart and ferry included, until the script is killed.

Every run is hermetic: **the world is restarted**, the aircraft is ferried above
the interior walls to the area, and only then does the policy get the
instruction. The restart is not tidiness — a capsized airframe cannot be
recovered any other way, so without it one bad contact poisons every subsequent
recording.

The areas are in `config/hospital_areas.yaml`, one per part of the building,
each carrying the clearance **measured off the occupancy map** rather than
guessed:

| area | (x, y) | clearance |
|---|---|---|
| `atrium` | (-0.09, 11.84) | 4.11 m |
| `north_wing` | (-0.09, 14.04) | 3.01 m |
| `reception` | (1.07, 2.08) | 1.54 m |
| `east_wards` | (8.47, -7.28) | 1.75 m |
| `south_hall` | (-1.61, -27.96) | 1.85 m |

```bash
$SCRIPTS/goto_area.py --list                          # what is available
$SCRIPTS/goto_area.py atrium                          # ferry there, by hand
.venv/bin/python $SCRIPTS/area_clearance.py           # re-measure them
.venv/bin/python $SCRIPTS/area_clearance.py --propose # pick fresh ones off the map
```

`goto_area.py` climbs to 4.5 m first — above the 3 m interior walls, which this
world has no ceiling over — crosses under a plain world-frame P controller, and
descends. Explicitly **not** the plugin's own position mode: `pid_controller.cpp`
clamps a setpoint to the controller's `Limit`, which for `Position XY` is 5, so
the aircraft flies confidently to y = 5.00 and hovers there for ever with healthy
odometry and no error anywhere. It is a ferry, not a planner, it confirms takeoff
and aborts rather than descending below wall height if it could not cross, and it
must never run while the follower is up: both publish `cmd_vel`.

## Configuration

One file: `robots/SJTU/config/vla/internvla_n1.yaml`. The knobs that matter:

* `server.host` / `server.port` — where the N1 model server is.
* `camera.*` — the SJTU front pinhole (600×600, fx=fy=390.64). Passed to the
  server so it projects its pixel goal correctly; a wrong intrinsic is this
  platform's most expensive bug.
* `commit.*` — how much of each prediction to fly before re-inferring (NavDP's
  plan-commitment discipline). Kept short for the 0.25 m action-fallback steps.
* `follower.cruise_speed` / `follower.target_altitude_m` / `follower.max_*` — the
  pursuit speed, the altitude held after takeoff, and the SJTU airframe clamps
  (well under its 2 m/s ceiling).

## Topics

| topic | type | dir | note |
|---|---|---|---|
| `/simple_drone/front/image_raw` | `sensor_msgs/Image` | in | RGB 600×600 |
| `/simple_drone/front_depth/depth/image_raw` | `sensor_msgs/Image` | in | 32FC1 metres |
| `/simple_drone/odom` | `nav_msgs/Odometry` | in | pose + body twist (the feedback source) |
| `/simple_drone/navigation/instruction` | `std_msgs/String` | in | the language goal |
| `/simple_drone/state` | `std_msgs/Int8` | in | 0 landed, 1 flying — the follower will not command a landed aircraft |
| `/simple_drone/n1/trajectory` | `nav_msgs/Path` | out | the committed route (world), what the follower flies |
| `/simple_drone/n1/trajectory_full` | `nav_msgs/Path` | out | the whole prediction, for RViz |
| `/simple_drone/n1/info` | `std_msgs/String` | out | JSON: action, S1/S2 FPS, the S2 pixel goal — what the recorder overlays |
| `/simple_drone/cmd_vel` | `geometry_msgs/Twist` | out | body FLU — the drone's only control input |

## Reading a run back

Every run writes a rosbag beside its MP4, and it is a real bag: `metadata.yaml`
and a finalised mcap footer, so `ros2 bag info` opens it and any reader can seek
it.

That took a fix worth knowing about, because the failure is invisible and
total. rosbag2 finalises on **SIGINT** -- flushing its cache, closing the mcap
footer and writing `metadata.yaml` -- but a **non-interactive shell starts every
background job with SIGINT set to IGNORE** (POSIX: it is what stops a Ctrl-C in
a script from killing its own children). So `ros2 bag record ... &` inside a
script cannot be stopped with `kill -INT` at all; the teardown waits out its
grace and SIGKILLs it, and what lands on disk is a bag with a truncated final
record and no metadata. `ros2 bag info` refuses it outright. Measured before the
fix: four of five runs in a campaign unreadable, and the fifth only partly.

The fix is to un-ignore the signal in a subshell before exec'ing:

```bash
( trap - INT; exec ros2 bag record -o "${BAG_DIR}" "${BAG_TOPICS[@]}" ) &
```

The same reset is applied to the node launch. SIGTERM also finalises the bag and
would have worked; SIGINT is what rosbag2 documents, so that is what is sent.

## Tests

ROS-free, in the plain `.venv`:

```bash
pytest sparx_agency/core/planning/vlas/internvla_n1 \
       sparx_agency/tasks/planning/sjtu_internvla_n1
```

The policy translation, the trajectory shaping and the path→trajectory timing are
all unit-tested without ROS or Gazebo; the ROS2 nodes are thin wiring over them.

## Middleware

The sim is Humble in a container; the host is Jazzy. `run_sjtu_n1.sh` picks the
RMW **from what is installed** — CycloneDDS when
`ros-<distro>-rmw-cyclonedds-cpp` is present, Fast DDS otherwise — and brings the
world up on the same one. Asking for an RMW that is absent aborts rclpy with a
dlopen error three screens into a redirected log, so it is chosen rather than
assumed.

Either way **shared memory has to be off**. Humble's Fast DDS and Jazzy's agree
that the shared segment is available (the container runs `--ipc=host` with
`/dev/shm` bound) and then fail to decode each other's samples: discovery
succeeds over multicast, so `ros2 topic list` is full, and every sample is
dropped, so `ros2 topic echo` hangs forever. It is indistinguishable from a
simulator that never started. The two profiles that fix it are
`robots/SJTU/setup/fastdds_udp_only.xml` and `cyclonedds_no_shm.xml`; the run
script exports whichever applies. Working by hand, export it yourself:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/sparx_agency/robots/SJTU/setup/fastdds_udp_only.xml
export FASTDDS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE
```

## Known limitations

* **The upstream patches must be applied and the server restarted.** There are
  more of them than there used to be, and two are the difference between a
  server that answers and one that lies — see
  `tasks/planning/vlas/internvla_n1/upstream/README.md`. Without the trajectory
  patch the policy falls back to the coarse discrete action; without the
  quantiser patch System 1 returns HTTP 500 on every step that matters while the
  server stays "healthy".
* **Never run two followers.** Both publish `/simple_drone/cmd_vel`, the drone's
  only control input, and the result is not a bad flight but a meaningless one.
  `run_sjtu_n1.sh` now launches its nodes in their own process group, signals the
  group, and sweeps for strays at both ends — because a teardown that only
  signalled `ros2 launch` was orphaning every node it started, and five runs of
  orphans is what capsized the aircraft nobody could explain.
  `scripts/goto_area.py` publishes `cmd_vel` too: it is for use *between*
  recordings, never during one.
* **Takeoff is confirmed, not commanded.** The plugin silently drops
  `/simple_drone/takeoff` from the wrong state, and a whole recording has been
  lost to an aircraft sitting on the floor at state 0 while the policy committed
  routes to it. `scripts/ensure_flying.py` retries until `/simple_drone/state`
  says 1, and refuses outright if the aircraft is capsized.
* **There is no landing sequence during flight.** The teardown stops the
  aircraft, then lands it; interrupt it any other way and the plugin holds the
  last twist for ever.
* **The obstacle reflex is a brake, not a planner.** It caps forward speed on raw
  depth; it does not route around anything. Keep `cruise_speed` conservative --
  eighteen of the hospital's twenty-six doorways are 0.93 m clear against a
  0.63 m airframe.

