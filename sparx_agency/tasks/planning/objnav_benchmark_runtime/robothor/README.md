# RoboTHOR ObjectNav runtime

The method is unchanged: **scene graph → LLM room probabilities → RPT\* room
order → weighted A\* → discrete action converter**. AI2-THOR supplies RGB-D and
ground-truth pose. **FALCON is not running**: it is a 3-D exploration and
mapping system, not the object detector, and its local exploration is
represented here by the existing 2-D host frontier sweep. No navigation-specific
training is performed, and no mandatory opening panorama is added.

This optional runtime uses the [shared benchmark harness](../../objnav_benchmark/README.md)
and — unlike the older Gibson adapter — the [shared runtime](../README.md)'s
`run_evaluation`, so resume, configuration locks, provenance, recording and the
dashboard are the shared ones rather than a second copy. Goal positions, the
shipped shortest path and distance telemetry reach only the scorer and the
recorder, never the policy.

## The benchmark, and where the rules actually live

Everything below was transcribed from `allenai/robothor-challenge`
(`challenge_config.yaml`, `robothor_challenge/challenge.py`),
`ai2thor.util.metrics`, and the Unity source of the **pinned build**
`bad5bc2b250615cb766ffb45d455c211329af17e`.

| | value | where it comes from |
|---|---|---|
| Split | val, **1,800 episodes**, 15 scenes `FloorPlan_Val{1,2,3}_{1..5}`, 120 each | repo README; the challenge *web page*'s "1080" is wrong |
| Targets | 12 `objectType` strings, 150 episodes each | `TARGET_TYPES`; the docs' 13th, `RemoteControl`, is not a challenge target |
| Camera | 640×480, `fieldOfView: 63.453048374758716` — **vertical**, = 79° horizontal | config comment; the two agree to 1e-14 at 4:3, pinned by a test |
| Camera height | **0.8688 m** = `origin_height_m` − 0.0312, i.e. above the *floor*, not below the agent's reported y | pinned build's bot branch; measured 0.8688 on a live build, which is what corrected it |
| Body | LoCoBot capsule: radius **0.175 m**, height **0.9 m** | pinned build's `m_CharacterController` |
| Actions | `MoveAhead` 0.25 m, `Rotate{Left,Right}` 30°, `Look{Up,Down}` 30°, `Stop` | `ALLOWED_ACTIONS` + config |
| Horizon clamp | **±30°**, and every episode *starts* at +30 (fully down) | pinned build sets both look angles to `30f` |
| Budget | 500 steps; a **failed action still spends one** | `total_steps += 1` runs before the action is issued |
| Success | `stopped and target_obj["visible"]` — **no distance is computed** | `challenge.py`; the 1 m rule rides entirely on `visibilityDistance: 1.0` |
| `l` (SPL numerator) | the **shipped** `shortest_path_length` in the episode file | the evaluator never queries the simulator for it |
| `p` | 3-D chords over the trajectory, from the spawn point, `Stop` and turns adding 0 | `ai2thor.util.metrics.path_distance` |

### Three decisions this adapter had to make, not inherit

1. **Which instance must be visible.** Upstream's `get_object_by_type` returns
   the *first* object of the goal type in the metadata list and asks only
   whether that one is visible — so an episode in a scene with two Mugs is
   failed when the agent finds the second. AllenAct, and the plain reading of
   the published rule, accept **any** visible instance. This adapter takes the
   any-instance rule (`--success-rule`, default `any_visible_instance`) and
   records the first-instance verdict beside it on every episode, so the report
   states how many episodes the choice actually moved instead of assuming it is
   negligible.

2. **Ground-truth pose.** The official runner calls `event.metadata.clear()`
   and hands the agent only `{object_goal, rgb, depth}` — **no pose, no GPS**.
   ESC says so explicitly for its own RoboTHOR runs. We use the simulator's
   ground-truth pose, as the Habitat-benchmark comparisons do and as this whole
   evaluation layer is premised on. **That is more information than the official
   protocol gives**, and it is recorded as `pose_source` in every manifest. It
   is the largest protocol deviation here and must be stated in any comparison.

3. **The build.** See below — it is not a free choice.

## The build is not interchangeable, and the reference one is not headless

The challenge and AllenAct both pin Unity build
`bad5bc2b250615cb766ffb45d455c211329af17e`. Two things in its own source make
it non-substitutable with a modern one:

* its LoCoBot branch clamps the horizon to **±30°**; ai2thor 5.0.0 raised the
  downward limit to 60°, so a newer build legalises a `LookDown` that upstream
  refused;
* its depth shader returns `Linear01Depth` packed into **three 8-bit channels**,
  where 5.0.0's returns float32 metres. The Python client decodes whatever the
  build sends, so **the client version and the build commit are one choice**.

And the reference build has **no CloudRendering variant** — only a `Linux64`
one — so it needs an X display. `--platform` names the trade:

| `--platform` | reaches the GPU by | comparable to published numbers |
|---|---|---|
| `reference-linux64` (default) | X display + `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` | yes — the pinned build |
| `cloud-rendering` | headless Vulkan + `VK_DRIVER_FILES=…/nvidia_icd.json` | **no** — needs `--thor-build` ≥ 3.5 and `--allow-build-mismatch` |

The preflight refuses both silent failures: a `cloud-rendering` run that still
asks for the reference build, and a `reference-linux64` run on a PRIME-offload
machine without the offload variables (without them GLX renders on the
integrated GPU and nothing says so).

## Metrics

**SR and SPL are the only metrics any paper reports on RoboTHOR.** SoftSPL and
DTG are computed here because the harness computes them for every benchmark,
but they have no published counterpart on this benchmark and must not be
presented as if they did.

* **SR** — the challenge's own judgement, mean over all episodes.
* **SPL** — `S · l / max(l, p)`, identical to `ai2thor.util.metrics`. Its
  arithmetic is also recomputed in AI2-THOR's own frame and cross-checked, which
  is what would catch a frame conversion that stopped being an isometry.
* **DTG** — geodesic metres from the final position to the nearest goal
  instance, via the same `GetShortestPath` call that generated the shipped
  paths, with AllenAct's escalating tolerance ladder. `inf` is counted
  separately, never averaged.
* **SoftSPL** — the harness's shared definition, using the shipped `l` as `d0`.
* **TF / NM** — **not metrics.** They are OSG Navigator's Table 2 method flags
  (*training-free*, *non-metric*), hand-authored per method. Ours is
  training-free and **metric**: the search integrates depth into a 2.5-D
  occupancy grid and plans through it with weighted A\*.

### The zero-length-`l` trap

**108 of the 1,800 validation episodes (6.0%) ship `shortest_path_length == 0`**
— the agent spawns already inside the goal cushion. Upstream's SPL gives
`l / max(l, p) = 0` there unless `p` is exactly zero, so a *successful* episode
that took even one step scores SPL 0. That is upstream arithmetic, reproduced
faithfully; it caps achievable SPL and it applies equally to every published
number. The count is recorded in the dataset manifest.

## Comparison

`reported.py` carries every published RoboTHOR number, each read from the table
that first printed it. The one trap worth repeating:

> **ESC's Table 1 and SG-Nav's Table 1 both put ProcTHOR's 65.2/28.8 and
> ProcTHOR-ZS's 55.0/23.7 in a column headed "RoboTHOR", beside their own
> 1,800-episode validation numbers. Those are 2,040-episode *test*-split
> figures.**

They are kept in `TEST_SPLIT_RESULTS`, and the shared `comparison_table`
refuses to print a test row beside a validation run. Two more cautions:

* **ESC does not reproduce**: it reports 38.1/22.2; two later papers using
  "the official implementation" both report 34.5/18.2, and ESC's code is not
  public. ESC anchors almost every later RoboTHOR table.
* **L3MVN, VLFM and OpenFMNav never published a RoboTHOR number**; their rows
  exist only inside SG-Nav's Table 1.

Best published zero-shot on validation: **CogNav, 54.6 SR / 24.3 SPL**. The
honest trained reference on the *same* episodes is EmbCLIP at **52.2 / 26.0**,
recovered from the archived leaderboard's `val_*` fields — no ObjectNav paper
cites it.

## Data and running

```text
<root>/val/episodes/FloorPlan_Val{1,2,3}_{1..5}.json.gz
```

Either `git clone https://github.com/allenai/robothor-challenge` (the episodes
ship in-tree under `dataset/`) or extract AllenAct's
`robothor-objectnav-challenge-2021.tar.gz` (2.4 MiB). The CLI downloads
nothing — not the episodes, not the ~550 MiB Unity build, not any model.

```bash
export ROBOTHOR_EPISODES_DIR="$HOME/datasets/objectnav/robothor"
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run \
  --preflight --detector-url http://127.0.0.1:18092

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run \
  --scene FloorPlan_Val1_1 --limit 1 --record \
  --detector-url http://127.0.0.1:18092 \
  --output "$HOME/objnav_benchmark/robothor/smoke"
```

`--preflight` checks dataset, services, GPU, build and platform without
rendering. `--resume` continues a run only under an unchanged configuration and
source. `--shards/--shard-index` split the sweep deterministically. Never mix
attempts in one output directory.

The detector needs this benchmark's own vocabulary, in order, on CPU:

```bash
VOCAB=$(python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run --print-vocabulary)
python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --model "$HOME/GIT/TheAgency/yolov8s-worldv2.pt" --device cpu \
  --host 127.0.0.1 --port 18094 --conf 0.05 --classes "$VOCAB"
```

Keep it on `--device cpu` (the server's own default is `cuda:0`): the 8 GB GPU
has to hold the Unity renderer, and the preflight refuses a busy one unless
`--allow-shared-gpu` is passed deliberately.

### A known perception risk, stated up front

RoboTHOR's goals are **small objects** — Apple, Mug, AlarmClock, SprayBottle,
Vase — where Gibson's were furniture. Two consequences are recorded rather than
hidden: `AlarmClock` accepts the detector label `"clock"` and `BasketBall`
accepts `"sports ball"`, because without them those 300 episodes have
essentially no recall; both are a real false-positive surface, and a false
accept is a false STOP. Detection quality, not exploration, is the likely
bottleneck on this benchmark.

## Tests

None of the tests need ai2thor, a GPU or the dataset: the episode files are
written in the challenge's own shape, and the simulator is a fake controller
that moves the way AI2-THOR moves — in AI2-THOR's frame — so that a mirrored or
transposed conversion shows up as a motion the harness refuses.

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark -q
```

See [status, findings and remaining work](RESUME_STATUS.md), and [protocol,
tuning and interpretation](PROTOCOL_AND_TUNING.md) for what the
cited papers do and do not establish, and for the freeze-then-evaluate workflow.
