# MP3D ObjectNav runtime

The method is unchanged: **scene graph -> LLM room probabilities -> RPT* room
order -> weighted A* -> discrete action converter**. Habitat supplies RGB-D and
pose, exactly as it does for every method in the comparison tables. **FALCON
itself is not running**: it is a 3-D exploration/mapping system, not the object
detector, and its local exploration is represented here by the existing 2-D
host frontier sweep. No navigation-specific training is performed.

This adapter is the MP3D half of the [shared ObjectNav runtime](../README.md);
the execution path, provenance, locks, recording and dashboards are shared with
every other benchmark, so a Gibson number and an MP3D number come out of the
same loop and the same arithmetic. Goal positions, view points and distances
reach the scorer and the recorder only, never the policy.

## The benchmark

The published **Habitat ObjectNav MP3D v1 validation split**, as used by the
2021 Habitat Challenge and by both comparison papers: **11 scenes, 21 goal
categories, 2,195 episodes**. Both papers state that split explicitly
(SG-Nav p.7: "the validation set, which contains 11 indoor scenes, 21 object
goal categories and 2195 episodes"; ApexNav Sec. V-A: "MP3D (Matterport3D from
2021 Habitat Challenge, 2195 episodes, 11 scenes, 21 goal categories)").

The 21 categories are in [`labels/datasets/mp3d.py`](../../../../core/planning/objnav/labels/datasets/mp3d.py).
The loader cross-checks that table against the release's own
`category_to_task_category_id`, so a different release fails at load rather
than scoring 21 categories against another vocabulary. The **scene names are
read from the installed release**, never assumed here.

## Protocol

Transcribed from habitat-lab `configs/tasks/objectnav_mp3d.yaml` (v0.2.2) and
`habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml` (v0.2.4), which
agree with habitat-challenge 2021's `challenge_objectnav2021.local.rgbd.yaml`.
Changing any of them is a different experiment.

| | |
| --- | --- |
| RGB and depth | 640x480, HFOV 79 deg, mounted at `[0, 0.88, 0]` |
| Depth range | 0.5 - 5.0 m |
| Body | height 0.88 m, radius 0.18 m |
| Actions | STOP, MOVE_FORWARD 0.25 m, TURN_LEFT/RIGHT 30 deg, LOOK_UP/DOWN 30 deg |
| Step budget | 500 actions |
| Sliding | **disabled** |
| Success | STOP **required**, within **0.1 m** of a published goal view point |
| Distance | geodesic, to the nearest view point (`DISTANCE_TO: VIEW_POINTS`) |
| Path | 3-D chords between successive base positions, zero initial accumulator |

The six-action space is the benchmark's, not the subset our policy uses: the
search never tilts the camera, but an episode advertises what the protocol
grants. Habitat does not clamp LOOK, so the pitch limits stay `None`.

### This is not the Gibson protocol

The Gibson adapter reproduces SemExp's evaluator: no STOP required, FMM
distance on a ground-truth semantic floor map, planar path length, a `1e-5`
initial accumulator, sliding on. MP3D is habitat-lab's evaluator, and every
one of those differs. Because habitat-lab's rules **are** the shared harness
defaults, `MP3DProtocol.evaluation()` overrides nothing but the motion
tolerances - see [`protocol.py`](protocol.py).

`distance.py` is an independent implementation of habitat-lab's
`DistanceToGoal` / `Success` / `SPL` / `SoftSPL` (`habitat/tasks/nav/nav.py`)
and `HabitatSim.geodesic_distance`: one `habitat_sim.MultiGoalShortestPath`
per episode, re-used across steps exactly as habitat-lab caches it on the
episode. habitat-lab itself is **not** a dependency - only `habitat_sim`.
Our values go into `native_metrics`, and the harness refuses the episode if its
own recomputation disagrees, so a frame, unit or wrong-goal bug shows up as a
failed check rather than as a suspicious SPL.

### Which navmesh the distances are measured on

This is the one detail most likely to make a number quietly incomparable, so it
is a protocol field (`navmesh_source`), not an implementation choice.

habitat-sim auto-loads `<scene>.navmesh`, compares its stored settings against
the agent's radius and height, and **silently recomputes it when they differ**.
The shipped Habitat navmeshes carry the library defaults (radius 0.1 m, height
1.5 m), and habitat-lab copies `AGENT_0.RADIUS`/`HEIGHT` straight into
`habitat_sim.AgentConfiguration`, so an ObjectNav agent of 0.18 m / 0.88 m
always runs on a *recomputed* navmesh. `DistanceToGoal` - hence `l`, `dT`, SR
and SPL - and the collision filter both come from that one.

Measured here on Collierville with habitat-sim 0.2.4:

| | navigable area |
| --- | --- |
| recomputed for the 0.18 m / 0.88 m ObjectNav agent | 44.145 m² |
| the shipped `.navmesh` (0.1 m / 1.5 m) | 58.006 m² |

16 of 25 sampled geodesic distances differ by more than 5 cm (up to 0.6 m), and
14 of 39 sampled point pairs are reachable **only** on the shipped navmesh. So
`navmesh_source = "agent_recomputed"`: force-loading the shipped file would
score a more permissive world than every published MP3D number was measured in,
and would let the agent walk through gaps habitat-lab's `step_filter` blocks.

Each episode's `info.geodesic_distance` is the cheapest possible check on all
of this, because the dataset generator wrote it. On the sibling HM3D branch,
measured against real episodes: the **recomputed** navmesh reproduces the
published `info.geodesic_distance` bit-exactly, while the shipped one comes out
about 5.5% short. So under `navmesh_source = "agent_recomputed"` the preflight's
`published_geodesic_gaps` should be close to empty, and a systematic gap is
evidence that something else is wrong - the wrong goals, the wrong frame, or the
wrong release. The preflight reports the gaps rather than failing on them.

For that check to mean anything the preflight has to measure on the navmesh the
run scores on, so `validate_starts()` loads each selected scene through the
simulator (a standalone `PathFinder` would read the published file and answer
for a navmesh the run never uses). That is the one part of `--preflight` that
needs the rendering GPU; `--skip-start-validation` opts out of it.

The same preflight refuses any selected start from which no view point is
reachable on the scoring navmesh, since `l` would be infinite and SPL
undefined. `--exclude-unreachable-starts` drops them, records them in the run
configuration and the audit, and **forfeits the full-split claim** - the
selection is then no longer the publisher's.

## Metrics

**SR** and **SPL** are the primary metrics: they are the two every MP3D row in
the comparison tables reports. We additionally record **DTG** (the final
geodesic distance to the nearest view point, metres), **SoftSPL**, steps and
wall time as supplementary.

**DTG, TF and NM** are the columns of *Open Scene Graphs* (arXiv:2508.04678):
DTG is distance-to-goal, **TF = training-free**, **NM = non-metric**. That
paper reports them for HM3D and Gibson, **not for MP3D**, so there is no
published MP3D DTG baseline to compare against - ours is reported as our own
supplementary number, not as a win over anyone. Our method is training-free
(TF) and metric (not NM). Ground-truth pose and depth do not mean ground-truth
semantics: detections stay predicted.

### Published rows we compare against

Transcribed in [`report.py`](report.py), each with the table it was read from.
Where the two papers disagree on the same method (VLFM on MP3D: 36.2/15.9 in
SG-Nav, 36.4/17.5 in ApexNav) **both rows are kept under their own sources**
rather than averaged or picked. They are reference values, not paired
re-evaluations; protocol equivalence with our run is not established, and no
SOTA claim is asserted by this package.

## Data

Two separate downloads, and only one of them is licensed.

**Episodes** (no licence gate, ~173 MB):

```bash
curl -L -o /tmp/objectnav_mp3d_v1.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
unzip /tmp/objectnav_mp3d_v1.zip -d ~/datasets/objectnav/mp3d
```

Note the publisher's `m3d` path typo; the `mp3d` spelling returns 403.

**Scenes** (Matterport3D Terms of Use required): sign the form linked from
<https://niessner.github.io/Matterport/>, then use the `download_mp.py` the
authors email you:

```bash
python2 download_mp.py --task habitat -o ~/datasets/scene_datasets
```

That ships `<scene>.glb` beside `<scene>.navmesh` for each scene. **The
published navmesh is authoritative** - this evaluator loads it and never
recomputes one, because a regenerated navmesh silently changes every geodesic
distance and therefore every SPL. The habitat task download is ~15 GB for all
90 scenes; only the 11 validation scenes are needed.

Expected layout:

```text
~/datasets/objectnav/mp3d/objectnav_mp3d_v1/val/val.json.gz
~/datasets/objectnav/mp3d/objectnav_mp3d_v1/val/content/<scene>.json.gz
~/datasets/scene_datasets/mp3d/<scene>/<scene>.glb
~/datasets/scene_datasets/mp3d/<scene>/<scene>.navmesh
```

Nothing here downloads, extracts or licence-gates anything for you: the CLI
reads local files only.

## Running

With the Habitat conda environment active (this workstation: `conda activate
habitat`, habitat-sim 0.2.4) and a dedicated detector and LLM service running:

```bash
export MP3D_EPISODES_DIR="$HOME/datasets/objectnav/mp3d/objectnav_mp3d_v1/val"
export MP3D_SCENES_DIR="$HOME/datasets/scene_datasets"
```

**Preflight** - checks data, assets, services, model identity, the habitat-sim
version and every selected start. All of it is render-free except the per-start
check, which loads each selected scene:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run --preflight
```

**Smoke** - one scene, a handful of episodes, recorded:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run \
  --scene 2azQ1b91cZZ --limit 3 --record \
  --output "$HOME/objnav_benchmark/mp3d/smoke"
```

**Sharded** - four shards of the full split, merged afterwards. Shards are
taken stride-wise so every shard covers every scene:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run \
  --shards 4 --shard-index 0 --output "$HOME/objnav_benchmark/mp3d/shard0"
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report \
  "$HOME"/objnav_benchmark/mp3d/shard[0-3] \
  --merge-output "$HOME/objnav_benchmark/mp3d/merged"
```

**Full validation** goes through the freeze-then-run workflow, never a bare
`run` - see [PROTOCOL_AND_TUNING.md](PROTOCOL_AND_TUNING.md#freeze-then-evaluate).

`--resume` continues the *same* output directory only while configuration,
data and source are unchanged; never mix attempts. `--print-vocabulary` emits
the exact detector class list, in order, that the dedicated detector service
must be started with:

```bash
VOCAB=$(python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.run --print-vocabulary)
python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --model "$HOME/GIT/TheAgency/yolov8s-worldv2.pt" --device cpu \
  --host 127.0.0.1 --port 18092 --conf 0.05 --classes "$VOCAB"
```

The detector runs on CPU, off Habitat's rendering GPU; preflight refuses a
rendering GPU that is already occupied unless you pass `--allow-shared-gpu`.
Door and doorway prompts need an emission threshold of 0.05; navigation goals
still require 0.35 and multi-view evidence before STOP.

## Method settings

The embodiment is the protocol's (0.88 m, 0.18 m) and is passed explicitly, not
inherited. The one genuine method choice is `stop_distance_m = 1.0`: MP3D
grants success for STOP within 0.1 m of a published view point, and those sit
within about a metre of the goal's surface, so the search drives to roughly
that distance of the landmark it believes in. `map_size_m` is raised to 100 m
because MP3D buildings are the largest of the five benchmarks and the observed
map is centred on the episode start. Both are in `METHOD_DEFAULTS` in
[`run.py`](run.py), both are recorded in the run configuration and the lock,
and both are overridable with `--policy-config`. Neither redefines success.

## Preflight checks worth knowing about

* **Unreachable starts.** If no published view point is reachable from a
  start, `l` is infinite and SPL is undefined. `validate_starts()` finds those
  before the run, on the scoring navmesh, and preflight refuses rather than
  scoring them; `--exclude-unreachable-starts` drops them with disclosure.
* **Publisher disagreement.** Each episode ships its own
  `info.geodesic_distance`. We recompute it and report every episode that
  differs by more than 5 cm: a systematic gap means a different navmesh, a
  different frame, or the wrong goals - the three bugs that move SPL without
  failing anything.
* **Quaternion order.** The release serializes `start_rotation` as
  `(x, y, z, w)`; habitat-sim's `quaternion.from_float_array` wants
  `(w, x, y, z)`. The conversion happens once, in
  `MP3DEpisode.start_rotation_wxyz()`, and is covered by a test.

## Simulation realism and recordings

Habitat renders a **3-D scanned mesh**, not stitched input frames; scan seams
and missing geometry can affect perception. Apparent jumps also come from the
published instantaneous 0.25 m / 30 deg actions played back at six decisions
per second. Changing either would change the benchmark, so smoothing lives in
a separate, visualization-only replay:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.smooth_replay \
  "$HOME/objnav_benchmark/mp3d/smoke/recordings/<episode>" \
  --episodes-dir "$MP3D_EPISODES_DIR" --scenes-dir "$MP3D_SCENES_DIR"
```

Run it only after the evaluation has released the GPU. `smooth_rgb.mp4` never
enters policy observations or scoring, and the original trajectories, metrics
and videos are untouched.

## Tests

No Matterport3D data, habitat-sim or GPU is needed: the release is built as a
synthetic fixture with the published *schema*, and the simulator and navmesh
are stubs.

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

`test_mp3d.py` covers the vocabulary, the protocol constants, the release
schema, the XYZW conversion and each evaluator rule; `test_mp3d_evaluation.py`
runs the whole shared path - loader, environment, headless agent, action
converter, the runner's kinematic and path cross-checks, the logger and the
report - against a scripted simulator that obeys the published action geometry
exactly. That proves the plumbing and the arithmetic. It is **not** a score.
