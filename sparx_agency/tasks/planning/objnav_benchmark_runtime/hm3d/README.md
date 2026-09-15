# HM3D ObjectNav runtime (v1 and v2)

The method is unchanged: **scene graph -> LLM room probabilities -> RPT\* room
order -> weighted A\* -> discrete action converter**. Habitat supplies
ground-truth RGB-D and a perfect pose, so what is measured is the search logic
and nothing else. **FALCON itself is not running** — it is a 3-D exploration and
mapping system, not an object detector; its local exploration is represented
here by the existing 2-D frontier sweep. No navigation-specific training is
performed, and no mandatory opening panorama is added.

This optional runtime composes the [shared benchmark harness](../../objnav_benchmark/README.md)
and the [shared runtime](../README.md). Goal positions, view points and distance
telemetry reach the scorer and the recorder only — never the policy.

> "ZSON" here means the zero-shot ObjectNav *task*. It is not a reproduction of
> the separately named ZSON paper, which appears in the comparison tables as a
> baseline.

## The two published datasets

One adapter, two versioned protocols, because only the episodes and the scenes
differ — the agent, the actions and the six categories are identical.

| | **HM3D-v1** | **HM3D-v2** |
|---|---|---|
| Episode dataset | `objectnav/hm3d/v1` | `objectnav/hm3d/v2` |
| Scene release | HM3D-Semantics **v0.1** | HM3D-Semantics **v0.2** |
| Challenge | Habitat 2022 | Habitat 2023 episodes, 2022 agent |
| Validation split | **2000 episodes / 20 scenes** | **1000 episodes / 36 scenes** |
| Scenes named | `hm3d/val/...` | `hm3d_v0.2/val/...` |
| Category balance | `plant` is only 84/2000 | balanced, 135–195 each |

The six goal categories are the dataset's own spellings, in its category-index
order: `chair`, `bed`, `plant`, `toilet`, `tv_monitor`, `sofa`.

The 20 v1 validation scenes are a subset of the 36 v2 ones, but the **geometry
is not identical** — HM3DSem v0.2 manually cleared debris in four validation
scenes, one of which (`00878-XB4GS9ShBRE`) is in the v1 set. Pointing both names
at one download, as ApexNav's README suggests, therefore runs one of the two
versions against scenes its episodes were not generated on. The loader refuses
it: every episode's `scene_id` prefix is checked against the protocol.

The **2023 challenge itself is not what the papers ran** — it moved to a
HelloRobot Stretch with velocity control. "HM3D-v2" throughout this package
means the v2 *episodes* under the v1 *agent*, which is what ApexNav's released
configs do.

## The protocol, and where it comes from

Every value is read off habitat-lab's own `objectnav_hm3d` benchmark
configuration, not inferred from a benchmark name:

| | |
|---|---|
| Steps | 500, STOP included |
| Actions | STOP, MOVE_FORWARD 0.25 m, TURN_LEFT/RIGHT 30°, LOOK_UP/DOWN 30° (pitch unclamped) |
| Agent | height 0.88 m, radius 0.18 m, sliding **off** |
| Camera | 640×480 RGB-D, HFOV 79°, 0.88 m above the floor, depth 0.5–5.0 m |
| Success | STOP **and** geodesic distance to the nearest goal **view point** < **0.1 m** |
| SPL | `S · l / max(l, p)`, `l` = that same distance at reset, `p` = 3-D chords |

`l` is the distance *habitat-lab measures at reset*, not the row's published
`info.geodesic_distance` — and it is measured to view points, not to the object.
The 1 m "within a metre of the object" of the task description is baked into
where the generator placed the view points; 0.1 m is only a snap tolerance on
top of them.

habitat-lab is **not installed here**; only habitat-sim is. So this adapter
reads the published episode files itself and reimplements `DistanceToGoal`
against habitat-sim's `MultiGoalShortestPath` — one number, measured once per
step, from which success, SPL, SoftSPL and DTG all follow. See
[`geodesic.py`](geodesic.py).

### The navmesh is not a detail

habitat-sim's `Simulator._config_pathfinder` loads the shipped
`<scene>.basis.navmesh` **and then recomputes it** whenever the mesh's recorded
settings differ from the agent's radius and height — which they always do here,
because HM3D's meshes are built at the library defaults (0.10 m / 1.50 m) and
ObjectNav's agent is 0.18 m / 0.88 m. So the reference pipeline runs on a
*recomputed* navmesh.

That changes what is reachable, and therefore `l`, and therefore every SPL.
Measured on an ObjectNav episode carrying a published `info.geodesic_distance`:
the mesh recomputed at 0.18/0.88 reproduced the publisher's number **bit-exactly**,
while the shipped mesh was 5.5% short. On one scene the ObjectNav settings cut
navigable area by 15% and made half of a sample of connected point pairs
unreachable.

`HM3DProtocol.navmesh` therefore defaults to `"agent"` — the reference
behaviour — and `"published"` exists only for a deliberate comparison. Whichever
is used, `navmesh_provenance()` records the settings, the navigable area and the
island count that were actually in force.

Because this lives in habitat-sim rather than habitat-lab, it is **not** a
difference between the papers we compare against: both run a habitat-sim that
recomputes, and so do we. What it does give us is a free check.
`info.geodesic_distance` is the publisher's geodesic to the nearest goal view
point — the same quantity as `l` — so `--validate-starts` reports how closely
our `l` reproduces it, and a systematic disagreement is a defect rather than a
difference of definition. `run.py` turns a bad agreement into a preflight
issue.

### The two deviations we name rather than apply

- **Success radius.** ApexNav's released configs use **0.2 m** where
  habitat-lab uses 0.1 m. Loosening it can only add successes. We score at 0.1 m
  and **re-score the same episodes** at 0.2 m from the rows on disk, so both
  rows exist from one run and cannot drift apart.
- **Depth near clip.** ApexNav opens the depth sensor to `min_depth 0.0`;
  SG-Nav's HM3D config raises `max_depth` to 10 m. We keep habitat-lab's
  0.5–5.0 m. This changes what the *method* can see, not how it is scored.

## Data

Episodes are public; scenes are licensed. Get the exact commands with:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets \
  --version v2 --data-root "$HOME/datasets" --instructions
```

and check an installation — counts, meshes, navmeshes, and the release-name
trap — with `--check`. Neither downloads anything, and neither takes a
credential: the scene meshes need Matterport's academic licence and an API
token, which belongs to the account holder and must never reach this repo.

Installed here already (public, no credentials):

```text
~/datasets/objectnav/hm3d/v1/val/{val.json.gz,content/<stem>.json.gz}   2000 ep / 20 scenes
~/datasets/objectnav/hm3d/v2/val/{val.json.gz,content/<stem>.json.gz}   1000 ep / 36 scenes
```

Still needed, ~3.8 GB (v0.1) and ~5.3 GB (v0.2):

```text
~/datasets/scene_datasets/hm3d/val/00NNN-<STEM>/<STEM>.basis.{glb,navmesh}
~/datasets/scene_datasets/hm3d_v0.2/val/00NNN-<STEM>/<STEM>.basis.{glb,navmesh}
```

habitat-sim's downloader only ever creates the name `hm3d`, and repoints it at
whichever release was fetched last — so the `hm3d_v0.2` name has to be made by
hand. `--instructions` spells that out; `--check` catches it if it was missed.

## Running

With the Habitat conda environment active:

```bash
export HM3D_EPISODES_DIR="$HOME/datasets/objectnav/hm3d/v2/val"
export HM3D_SCENES_DIR="$HOME/datasets/scene_datasets"

# does everything exist, and under exactly what configuration?  No GPU touched.
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
  --version v2 --preflight

# a few episodes of one scene, recorded
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
  --version v2 --scenes 00877-4ok3usBNeis --limit 3 --record \
  --detector-url http://127.0.0.1:18092 \
  --output "$HOME/objnav_benchmark/hm3d_v2/smoke"
```

`--split train --episodes-per-scene N` runs the development split instead. It
renames the protocol, files its rows separately, and can never be called
complete — see [protocol and tuning](PROTOCOL_AND_TUNING.md).

`--validate-starts` loads every selected scene once and measures `l` up front.
It is slow, and it is the only way to find an episode whose goals are
unreachable *before* the run stops at it; `--exclude-unreachable` then drops
those explicitly, and the dropped ids are recorded in the run manifest.

A full split is 20 or 36 buildings, and habitat-sim holds a scene's meshes and
its rendering context for the life of the process — so run it scene by scene,
each in its own child process:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign \
  "$HOME/objnav_benchmark/hm3d_v2/campaign" --version v2 \
  --episodes-dir "$HM3D_EPISODES_DIR" --scenes-dir "$HM3D_SCENES_DIR" \
  --resume -- --detector-url http://127.0.0.1:18092
```

Each scene gets a complete results directory of its own, so one that dies costs
one scene and can be resumed on its own; `--aggregate-only` then summarises
exactly the scenes that produced rows and names the ones that did not. Add
`--limit 1` for a one-episode-per-scene smoke across the whole split.

Alternatively shard a run with `--shards N --shard-index I` and merge with
`report.py --merge-output`, which refuses shards that disagree on data,
protocol, method or runtime, or that do not cover every episode exactly once.

Use the same `--output` with `--resume` only for an unchanged configuration and
unchanged data; the logger refuses anything else. A full validation run goes
through the [freeze workflow](PROTOCOL_AND_TUNING.md#freeze-then-evaluate).

The detector needs this benchmark's own vocabulary, in this order:

```bash
VOCAB=$(python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run --print-vocabulary)
python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --model "$HOME/GIT/TheAgency/yolov8s-worldv2.pt" --device cpu \
  --host 127.0.0.1 --port 18092 --conf 0.05 --classes "$VOCAB"
```

Preflight verifies the detector's model identity and vocabulary, not just a live
port, and refuses an emission threshold high enough to hide door candidates.
`LLM_BACKEND`, `LLM_BASE_URL` and `LLM_MODEL` configure the shared LLM client;
the API key stays in the environment and never reaches a result artifact.

## What we report, and what the papers report

SR, SPL and DTG are the three the comparison papers use; SoftSPL is
supplementary. All four are means over **every scored episode, failures
included**, as the leaderboards compare them. DTG is habitat-lab's
`distance_to_goal` — geodesic, to the nearest goal view point.

**`TF` and `NM` are not metrics.** They are property checkmarks in OSG's
*Gibson* table (its HM3D table has no such columns) meaning **training-free**
and **non-metric**. Our method is training-free; it is not non-metric, since it
builds a metric map. `RPTSearchPolicy.configuration()` records both flags.

Reference numbers live in [`sota.py`](sota.py) with the table each was printed
in. They are transcribed, never measured here, and the protocols behind them
differ — ApexNav's own results at 0.2 m against SG-Nav's 0.1 m, most of
ApexNav's HM3D-v1 baseline rows being quotes rather than its own measurements,
and the two papers printing different numbers for the same
baseline. OSG's HM3D row is a 400-episode self-sampled subset with no published
ids, no step cap, RGB-only observations and no STOP action; it is recorded for
context and excluded from the comparison table unless `--include-osg` is passed.

See [protocol, tuning and interpretation](PROTOCOL_AND_TUNING.md) before putting
any of our numbers next to theirs.

## Tests

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark -q
```

They run in the numpy-only environment: no habitat-sim, no scene meshes. The
simulator and the distance provider are injected, so the success rule, the path
accounting and the measurement arithmetic are tested on geometry the test picks.
One test loads the real installed episode split when it is present, which is the
only thing that proves the parser matches the publisher's actual format.

## Files

| file | what it holds |
|---|---|
| `protocol.py` | `HM3DProtocol`, `HM3D_V1`, `HM3D_V2`, the navmesh choice, the named ApexNav radius |
| `dataset.py` | the published episodes and goals, scene-qualified ids, XYZW→WXYZ, the manifest |
| `geodesic.py` | habitat-lab's `DistanceToGoal` to view points, on habitat-sim's pathfinder |
| `env.py` | `HM3DEnv`: the episode, the measurement, start validation, navmesh provenance |
| `assets.py` | what is installed, what is missing, and the commands that would fix it |
| `run.py` | preflight and the run, on the shared `run_evaluation` |
| `campaign.py` | one child process per scene, then an aggregate over what finished |
| `report.py` | the comparison table, the 0.2 m re-score, the audit, shard merging |
| `sota.py` | the published numbers, with the table each came from |
| `frozen_eval.py` | freeze an exact configuration, then run the full split under it |
| `smooth_replay.py` | visualization-only smooth video, after the fact, never scored |
