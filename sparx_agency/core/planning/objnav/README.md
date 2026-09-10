# ObjectNav layer — the method, detached from ROS, physics and perception noise

The shared foundation for benchmarking our object search (scene graph + LLM room
probabilities + RPT\*) against zero-shot ObjectNav SOTA on MP3D, HM3D v1/v2,
Gibson and RoboTHOR. It holds everything those benchmarks have in common and
nothing any one of them owns.

The premise is isolation. A benchmark step here receives the simulator's
**ground-truth RGB-D** and **perfect pose** — no depth network (no DA3), no
localiser, no dynamics — so the numbers measure the search logic and nothing
else. The method speaks continuous world geometry; the benchmark speaks six
discrete actions; this package is the translation between them.

```text
 ObjNavEnv  (one adapter per benchmark branch)         HeadlessObjNavAgent  (this package)
 ─────────────────────────────────────────────         ─────────────────────────────────────────────
 reset(id) ─► ObjNavEpisode + ObjNavObservation ─────► reset(episode)
                                                          LabelMapper ─► TargetLabels ─► SearchPolicy.reset
 step(action) ─► ObjNavObservation ──────────────────► act(observation)
      ▲                                                   SearchPolicy.plan ─► NavigationCommand   (where)
      └──────────────── DiscreteAction ◄───────────────── DiscreteActionConverter.step              (how)
 measure() ─► EpisodeMeasurement  (privileged: to the harness, never to the agent)
```

Scoring, logging and the run loop live in the harness,
[`tasks/planning/objnav_benchmark`](../../../tasks/planning/objnav_benchmark/).

Python 3.8 syntax, numpy only, no ROS — enforced by `tests/test_python38_contract.py`
(which also scans every repo module the package pulls in) and
`tests/test_import_weight.py`, because a benchmark adapter may run in the Habitat
conda env (Python 3.9) and `core/` must also import in the FALCON Noetic
container (Python 3.8).

## The parts

**The contract** (`types/`, `interfaces/`, `errors.py`). Frozen dataclasses,
validated on construction, and four interfaces:

| interface | kind | who implements it |
|---|---|---|
| `ObjNavEnv` | ABC | each benchmark branch, once per simulator |
| `ObjNavAgent` | ABC | `HeadlessObjNavAgent`, or any other agent under test |
| `SearchPolicy` | Protocol | the method — and every baseline — that decides where to go |
| `LabelMapper` | ABC | one table per dataset vocabulary |

The episode an agent sees (`ObjNavEpisode`) has no typed field for a goal
position or a geodesic distance. Its free-form `metadata` mapping is handed to
the policy as is, so privileged values (shortest paths, geodesic distances,
goal positions) must never be put there. The adapter is responsible; the
harness's reset check refuses the known leaking keys as a tripwire, not a
guarantee.

**The action converter** (`action_converter/`). A path of world waypoints in, one
discrete action out, assuming perfect execution and re-deciding from the true
pose every step, so the real simulator's collisions and RoboTHOR's actuation
noise are corrected rather than accumulated. Each step walks one ladder, in
order: stop if asked → tilt the camera if asked → rotate toward a lookahead
point on the path when the heading error exceeds half a turn → step forward
while that brings the agent closer → face the requested heading at the end.

How it follows a path, and why:

- **Progress only moves forward, and stays on the leg being walked.** It is a
  high-water mark in arc length, kept on the first leg within
  `parallel_offset` (0.134 m for Habitat) of the agent, so a later leg that
  passes near it cannot capture it. Without that, near-reversing corners
  livelocked and room tours skipped rooms. An aim point that lands within half
  a step of the agent, short of the goal, is moved on along the path, so exact
  folds and retraces are walked, not stepped off sideways.
- **Arrival means the end of the path.** The command is complete only when the
  aim is the last waypoint and the agent is within
  `max(goal_tolerance_m, reach_floor)`, where `reach_floor` = step / (2 cos(half
  a turn)) = 0.129 m for Habitat is the closest a half-turn-aligned step can
  get. A path that passes near its own end is walked in full, and arrival is
  absorbing: turns after it do not undo it.
- **The heading dead band is exactly half a turn**, so under exact execution a
  turn always lands inside it and the converter never swings left-right-left.
  RoboTHOR's turn noise (σ 0.5°) occasionally makes an overshooting turn be
  undone — a wasted step, never a livelock.
- **Nothing to do is not STOP.** A converter with nothing left to do returns no
  action; ending the episode is the policy's decision. A command built with
  `stop_on_arrival=True` gets its STOP on the exact step it is fully executed —
  still the policy's decision, taken in advance, and the only way to stop right
  after facing something without an idle turn in between.

A MOVE_FORWARD that does not move the agent is reported (`forward_blocked`, one
definition, on every result). Until a forward step moves again (or `reset()`),
the converter aims one step ahead instead of `lookahead_m`, so a corner cut into
a wall is not simply repeated. It never plans a detour, and it cannot rescue a
path that hugs the navigable boundary: its heading is quantised to half a turn,
so a zero-clearance route still gets blocked (on the fake building a
zero-clearance geodesic route livelocked in 15 of 90 episodes; a
clearance-weighted one was never blocked). **Plan with clearance, and implement
`notify_blocked`.**

It is not a replacement for `core/planning/vlas/internvla_n1/trt/postprocess.py`.
That module reproduces upstream InternNav's walker exactly (15-degree turns,
lookahead counted in path indices, a body-frame path from the origin) as a
TensorRT quality gate. This one works in the world frame, from the current pose,
with a lookahead in metres, STOP, camera tilt and a final heading.

**The headless agent** (`agent/`). Composes a `SearchPolicy`, a `LabelMapper`
and a converter behind the `ObjNavAgent` step API. It validates every
observation against its episode (target, camera, step order), calls the policy's
optional `notify_blocked(observation)` before `plan()` whenever the last step was
blocked, and spends an idle step on a turn (`TURN_LEFT` by default; only the two
turns are accepted, never STOP). Each step's reasons come back to the caller of
`act()` in `AgentDecision.info`; the harness stores only `episode_info()` —
statuses, idle steps, blocked steps and notifications — in the results row.

**Label mapping** (`labels/`). A dataset names its goals in its own words
(`tv_monitor`, `Television`, `chest_of_drawers`). The mapper turns one into a
`TargetLabels`: a phrase for the LLM prompts, the prompts the open-vocabulary
detector must be given (in a fixed order, so a list or tuple — never a set), and
the exact set of detector labels that count as the target. Matching is exact
membership after `normalize_label`, never the fuzzy `label_matches`, because on a
benchmark a false accept is a false STOP. A table that would let one detector
label count for two categories is refused.

**Camera geometry** (`camera_geometry.py`, `camera_intrinsics.py`). The executable
definition of what a pose and a depth pixel mean: world-from-camera transforms,
depth back-projection and projection. `intrinsics_from_hfov` takes Habitat's
horizontal field of view; `intrinsics_from_vfov` takes AI2-THOR's `fieldOfView`,
which is **vertical** (RoboTHOR's 63.453° is 79° horizontal at 640×480).

## Frames

| quantity | convention |
|---|---|
| world | ENU, `z` up, right-handed (the frame of `Pose2D`, A\* and every map) |
| yaw | radians, counter-clockwise about `+z` from world `+x`; wrap it at the adapter (beyond 1e6 rad it is refused) |
| camera pitch | REP-103: about body `+y`, **positive looks down** (Habitat's sensor pitch is positive up; AI2-THOR's horizon is positive down) |
| `AgentPose.z` | the floor under the agent; the camera sits `CameraSpec.height_m` above it |
| depth | optical-frame `z` in metres; `NaN` no reading; `+inf` no surface within range. A reading equal to a clip bound is a clipped return, not a measurement: a near clip or a zero/hole pixel becomes `NaN`, a far clip `+inf`. `backproject_depth` keeps only `0 < d` and `min_depth_m < d < max_depth_m` |
| pixels | integer index = pixel centre; symmetric pinhole principal point `((W-1)/2, (H-1)/2)` |

Every simulator converts into these at its own adapter, once. Nothing
downstream knows which simulator it is talking to.

## Plugging in a method

A method is a `SearchPolicy`: `reset(episode, target)` and
`plan(observation) -> NavigationCommand`. It is told the target as
`TargetLabels` — `query` for the LLM, `detector_prompts` for the detector,
`accepts(label)` to decide whether a detection *is* the target — and answers in
world coordinates: `NavigationCommand.follow(path)`, `.hold(camera_pitch=...,
final_yaw=...)` or `.stop_here()`.

- **Stop only when the benchmark's own success criterion holds.** For the final
  approach send `follow(path, final_yaw=..., stop_on_arrival=True)` (or a `hold`
  with it): STOP then comes on the step the facing completes. On RoboTHOR the
  object must be in view in the STOP frame, and a `stop_here()` sent one step
  later comes after the idle turn has rotated the view away.
- **One floor per command.** Waypoints are 2-D and the converter is planar: a
  `follow()` path must stay on the agent's current floor. To change floors, send
  the path to and up the stairs, then the next floor's leg.
- **Plan with clearance, and implement `notify_blocked`** (see the converter).
- Build maps from the ground-truth depth with `camera_geometry.backproject_depth`,
  which already folds in the camera's pitch.

## Adding a benchmark

Each environment branch (`feat/objnav-habitat-{hm3d,mp3d,gibson}-nadav`,
`feat/objnav-ai2thor-robothor-nadav`) does four things:

1. **Implement `ObjNavEnv`** for its simulator:
   - Convert frames, the pitch sign, and action names — by name, never by integer.
   - Convert the depth encoding. Habitat normalises depth to `[0, 1]` over
     0.5–5 m by default: a normalised 0 becomes `NaN`, 1 becomes `+inf` (not
     0.5 m and 5 m).
   - Build the camera with `intrinsics_from_hfov` (Habitat) or
     `intrinsics_from_vfov` (AI2-THOR), and wrap yaw and pitch.
   - Configure the simulator to execute the advertised `DiscreteActionSpec`:
     AI2-THOR with `rotateStepDegrees=30`, Habitat with the benchmark yaml's
     turn and tilt. The harness checks the realised motion after every action
     and refuses a mismatch. RoboTHOR adapters may round `cameraHorizon` to 0.1°.
   - Make `episode_ids()` unique within (benchmark, split): Habitat ObjectNav
     numbers episodes from 0 in every scene file, so scene-qualify them.
   - Never copy dataset rows into `metadata`.
   - Refuse or exclude, up front, any episode whose shortest path the benchmark
     cannot compute: `EpisodeMeasurement` refuses a non-finite `l`, and a run
     would otherwise stop at that episode on every resume.
   - Report `EpisodeMeasurement` with the benchmark's own `l`, `d0`, `dT`, `p`
     and success. Put the simulator's own metric values under the `NATIVE_*`
     keys only where they use the same definitions.
2. **Add its label table** as `labels/datasets/<dataset>.py` (a
   `TableLabelMapper`; prompts and accept labels as lists or tuples) and register
   it in `labels/registry.py`.
3. **State its protocol** to the harness: whether success requires STOP (Habitat
   and RoboTHOR: yes; SemExp's Gibson evaluator: no), the action spec and pitch
   limits (Habitat: unclamped; RoboTHOR LoCoBot: ±30 degrees, and a LOOK beyond
   fails; limits are judged with a 1e-4 rad float-noise slack), the step budget.
4. **Record the SOTA numbers** it compares against, with their sources.

## Files

| path | what it holds |
|---|---|
| `errors.py` | the error family: `ObjNavError` and its subclasses, and `ObjNavInternalError` |
| `types/` | actions and their geometry, pose, camera, observation, episode, navigation command, target labels, decision, measurement; `angles.py` holds the one angle bound |
| `interfaces/` | `ObjNavEnv`, `ObjNavAgent`, `SearchPolicy`, `LabelMapper` |
| `camera_geometry.py` | camera transforms, depth back-projection and projection |
| `camera_intrinsics.py` | intrinsics from Habitat's horizontal or AI2-THOR's vertical field of view |
| `action_converter/path_geometry.py` | path length, forward-monotone projection with the leg-keeping band, point at arc length |
| `action_converter/action_choice.py` | heading error, and the one-step turn and tilt choices |
| `action_converter/checks.py` | the number checks both of those share |
| `action_converter/path_progress.py` | the adopted path, its forward-only progress, and the aim point |
| `action_converter/ladder.py` | the stateless decision ladder, `reach_floor` and `parallel_offset` |
| `action_converter/transition.py` | what one action does to the pose, under perfect execution |
| `action_converter/converter.py` | `DiscreteActionConverter`: STOP, path state, blocked-step recovery, rollout |
| `action_converter/params.py`, `types.py` | its tuning, and its results |
| `labels/` | the table mapper, its registry, and (on the benchmark branches) `datasets/` |
| `agent/headless_agent.py` | `HeadlessObjNavAgent` |
| `agent/episode_checks.py` | its checks on the mapped target and on every frame |
| `agent/episode_log.py` | what it records per step and per episode |
| `agent/params.py` | its parameters |
| `tests/` | the contract types (one file per family), the interfaces, camera geometry and intrinsics, the docstrings' promises (`test_contract_docs.py`), the Python 3.8 and import-weight rules |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/objnav -q
```

## Not in here, on purpose

- **No simulator.** Habitat and AI2-THOR adapters live on their own branches.
- **No method.** The scene graph, the LLM oracle and RPT\* plug in later as a
  `SearchPolicy`; this layer only defines the seam.
- **No scoring.** SPL, SoftSPL and the statistics are in the harness, so that
  every benchmark is scored by one piece of code.
- **No mapping from depth.** `camera_geometry` defines the frames; building an
  occupancy grid from ground-truth depth belongs to the policy that needs one.
- **No 3-D arrival.** Paths are planar, one floor per command (see above).
