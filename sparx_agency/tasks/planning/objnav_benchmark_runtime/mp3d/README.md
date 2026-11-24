# MP3D ObjectNav — execution guide

The Matterport3D adapter of the shared ObjectNav search. **One algorithm, three
simulators**: the search policy, the perception services, the action converter,
the recorder and the scorer are the same code that runs on Gibson and HM3D; this
package is the *MP3D simulation driver* and nothing else. Everything that is
specific to the simulation lives here and only here:

| What | Where |
| --- | --- |
| Step size (0.25 m), turn (30°), tilt (30°), the six actions, the pitch the method keeps to | `protocol.py` → `MP3DProtocol.actions()` |
| Camera: 640×480, HFOV 79°, mounted at 0.88 m, depth 0.5–5.0 m, the depth-hole fill rule | `protocol.py` → `camera()`, `depth_holes()` |
| Body: 0.18 m radius, 0.88 m height; sliding off; the navmesh the run is scored on | `protocol.py` → `MP3DEnv.__init__` hands them to the shared Habitat bridge |
| Stopping criterion: STOP within **0.1 m geodesic of a published goal view point** (strict `<`) | `env.py` → `MP3DEnv.measure()` |
| Step budget 500; `l`, `d0`, `dT`, `p`, SR, SPL, SoftSPL as habitat-lab computes them | `env.py`, with the distance from `../habitat/geodesic.py` (habitat-lab's measure, shared) |
| The published episodes, goals and scene files; the 21-category vocabulary | `dataset.py`; `core/planning/objnav/labels/datasets/mp3d.py` |
| Kinematic self-check bounds for habitat-sim's recomputed navmesh | `protocol.py` → `kinematics()` |

The policy is built by the one shared `policy_factory.build_policy`; the only
things this adapter tells it are the label table and the protocol's body.
Nothing here came from an MP3D score.

## One command (the launcher)

For **2,195 MP3D v1 val episodes on a 24 GiB RTX 4090 / 64 GiB PC**, use
[`run_4090.sh`](run_4090.sh). It previews by default; see the [workstation run instructions](../README.md#full-hm3d-and-mp3d-validation-on-one-rtx-4090).

```bash
cd ~/GIT/TheAgency
sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh            # the full published val split
sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh --jobs max  # ... as many scenes at once as the card holds
```

`run_benchmark.sh` is the whole benchmark at the push of a button:

1. **Data.** Checks the installation (`assets.py --check`): the 2,195 episodes
   in 11 scenes with 21 categories, and every scene's `.glb` and `.navmesh`.
   The public episode archive is **fetched automatically** when absent (173 MB,
   no credentials). The scenes are licensed and must already be installed
   (section 1.2); the launcher names exactly what is missing.
2. **Services.** The GPU plan, Ollama with the two LLM roles, the YOLO-World
   detector started with the MP3D table's vocabulary on port 18093, the
   optional segmenter, FFmpeg for recordings — `../benchmark_services.sh`,
   shared with Gibson and HM3D.
3. **Preflight.** The frozen configuration without a mesh: protocol, runtime
   versions, method and services, dataset manifest (`<output>/preflight.json`).
4. **The campaign.** One `mp3d.run` process per scene (`campaign.py`), each
   validating its starts on the navmesh it scores on (`VALIDATE_STARTS=1`:
   every start's `l` must reproduce the publisher's own `info.geodesic_distance`
   to 1 mm, or the scene refuses to run), then the aggregate
   `campaign_summary.json` with `complete: true` only when all 2,195 episodes
   of all 11 scenes finished.
5. **Reports.** Per scene `comparison.md` (our row under SG-Nav's and ApexNav's
   transcribed rows, at the official 0.1 m radius *and* re-scored at ApexNav's
   0.2 m) and `audit.json`.

`--output runs/X` resumes an interrupted campaign (finished scenes kept,
started scenes continued). `--status runs/X` prints the live progress. Run
`run_benchmark.sh --help` for every option (`--scene`, `--limit`, `--smoke`,
`--record`, `--category`, `--episode-ids-file`, `--agent falcon-yolo`, ...).

## Ground-truth perception (the upper bound)

```bash
sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d/run_benchmark.sh --perception ground-truth --jobs 6
```

As on HM3D: habitat-sim's semantic sensor answers the detector and segmenter
contracts (tight boxes, labels, exact instance masks, confidence 1.0; instances
under 64 px not reported), the method is otherwise unchanged, no detector or SAM
is started, and the run is **non-publishable**. Output defaults to
`runs/gt_oracle_mp3d_2195ep_<stamp>`.

**The semantic config is required.** The scenes' `_semantic.ply` and `.glb` are
in different frames; the publisher's `mp3d.scene_dataset_config.json` registers
one to the other, and habitat-sim's built-in `"default"` does not (measured
2026-11-04, habitat-sim 0.2.4: 8 of 418 rendered ids lay on their objects in
X7HyMhZNoso under `"default"`, 604 of 630 under the config). The launcher installs
it (`assets.py --fetch-semantic-config`, 809 bytes, public) into
`$MP3D_SCENES_DIR/mp3d/`, and the preflight refuses a ground-truth run without it.
It changes nothing a service run sees: navmesh, RGB, depth and poses are
identical with and without it.

**The label table** (`../ground_truth/mp3d_categories.py`), and where each part
comes from:

| Part | Source |
| --- | --- |
| The 21 goal classes and their mpcat40 ids | `objectnav_mp3d_v1`'s `category_to_mp3d_category_id`, equal to Matterport's `metadata/mpcat40.tsv` (checked at preflight) |
| Each instance's raw name and mpcat40 class | the scene's `.house` `C`/`O` lines, read by habitat-sim |
| Which instances are a scene's goals | the episodes' `goals_by_category[...].object_id` (= `.house` object index = rendered id; all 1,316 val goals checked) |
| Context words (doors, stairs, appliances, ...) | mpcat40 class where it is one word (door, stairs, mirror, curtain), else every raw spelling Matterport's `metadata/category_mapping.tsv` cleans to the word |

Tiers: the publisher's goal ids first (an instance of a goal class the episodes
do not list is not reported, as on HM3D); then, in a scene with no episodes of a
class, its instances by mpcat40 name; context words never land on a goal-class
instance. Object 0 renders as background and is never reported (no val goal is
object 0).

**What the official classes contain** (raw names of the 1,316 val goals): `table`
includes desks, side and coffee tables; `chest_of_drawers` is mostly nightstands;
`cushion` mostly pillows; `stool` includes ottomans; `bed` includes headboards;
`shower` goals are mostly shower walls and floors; `tv_monitor` (4 goals) includes
a bulletin board. And unlisted same-class instances are common for `picture`
(253 against 114 goals) and `cabinet` (37 against 125): they are not reported.

## 1. Setup & downloads

### 1.1 Environments

Identical to HM3D's (see `../hm3d/README.md` §1.1): the habitat-sim 0.2.4 conda
environment (`HAB_PY`; the launcher finds `~/miniconda3/envs/objnav-habitat` or
`~/miniconda3/envs/habitat` itself), the detector venv (`DETECT_PY`,
`~/.venvs/objnav-detector`), Ollama, FFmpeg for recordings. The lightweight
test venv needs neither simulator nor GPU.

### 1.2 Datasets

```bash
# Episodes: public, fetched by the launcher when absent, or by hand:
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets --fetch-episodes --check
#   -> ~/datasets/objectnav/mp3d/v1/{train,val,val_mini}/

# Scenes: licensed by Matterport. Sign the terms of use (https://niessner.github.io/Matterport/),
# request the Habitat task download (mp3d_habitat.zip, 15 GB, 90 scenes) and unpack it so that
#   ~/datasets/scene_datasets/mp3d/<scene>/<scene>.glb  (+ .navmesh, .house, _semantic.ply)
# Then verify:
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets --check
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets --instructions
```

`DATA` (default `~/datasets`) is the root; `MP3D_EPISODES_DIR` and
`MP3D_SCENES_DIR` override the two halves.

### 1.3 Model weights

As HM3D: `$MODELS/yolov8x-worldv2.pt` (YOLO-World), the Ollama models named by
`LLM_MODEL` / `LLM_REASONING_MODEL` (pulled by the launcher when
`OLLAMA_MODE` is `docker` or `native`).

## 2. Single run & video recording

```bash
mp3d/run_benchmark.sh --preflight                                 # services + data + configuration, no episode
mp3d/run_benchmark.sh --scene 2azQ1b91cZZ --limit 1 --record      # one recorded episode
mp3d/run_benchmark.sh --scene pLe4wQe7qrG --limit 3 --record --save-frames
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.smooth_replay runs/mp3d_v1_pLe4wQe7qrG_<stamp>
```

Without the launcher, `mp3d.run` is the CLI (`--help`): `--episodes-dir`,
`--scenes-dir`, `--detector-url`, `--scenes`, `--limit`, `--episode-ids`,
`--category`, `--record`, `--validate-starts`, `--exclude-unreachable`,
`--resume`, `--preflight`, `--print-vocabulary`. The method switches
(`--agent`, `--explorer`, `--policy-config`, `--segmentation`) are the shared
ones.

## 3. The full published validation split

```bash
mp3d/run_benchmark.sh --jobs max                 # 11 scenes, as many at once as fit; ~200 episodes each
mp3d/run_benchmark.sh --status runs/mp3d_v1_val_<stamp>
```

The result is `runs/mp3d_v1_val_<stamp>/campaign_summary.json`: SR, SPL,
SoftSPL and DTG over every scored episode, `complete: true` only for the whole
split, `campaign_metrics.csv` per scene, and in the summary the split per goal
category (`by_category`) and the same rows re-scored at ApexNav's 0.2 m
(`apexnav_radius`). Each scene's `comparison.md` puts our row under the papers'
rows at both success radii.

### Sharded: no scene becomes the tail, and the LLM is spread over servers

One process per scene makes the campaign as long as its slowest scene. With
`--shard-episodes N` every scene is cut into shards of at most N episodes
(`mp3d.run --shards K --shard-index k`: the publisher's order strided, nothing
dropped or repeated), each its own process writing `<root>/<scene>/shard_<k>/`;
when a scene's shards have all finished, `report.merge_runs` writes the scene's
own directory from them, so the aggregate is unchanged.

```bash
S=runs/gt_oracle_mp3d_2195ep_X_support; mkdir -p $S; echo 14 > $S/jobs
OLLAMA_MODE=native LLM_BASE_URL=http://127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 HABITAT_VRAM_MIB=1400 \
LLM_ANSWERS_RECORD=$PWD/$S/llm_answers \
mp3d/run_benchmark.sh --perception ground-truth --shard-episodes 10 --jobs 14 --jobs-file $S/jobs \
    --llm-servers 3 --output runs/gt_oracle_mp3d_2195ep_X
mp3d/run_benchmark.sh --status runs/gt_oracle_mp3d_2195ep_X     # one screen: done/total, SR/SPL, throughput, ETA, per scene
```

* **The pool** (`shard_pool.py`) starts a shard only while the card's used VRAM
  plus `HABITAT_VRAM_MIB` (and that again for every shard started in the last
  two minutes) stays under `VRAM_CAP` (0.88), 20 s apart. VRAM never pauses or
  stops running work. The number at once is `--jobs`, overridden live by the
  integer in `--jobs-file`; lowering it lets running shards finish, `0` drains
  the pool for a resume under new settings.
* **The LLM servers.** One Ollama server answers about 0.7-0.8 requests/s
  whatever its slot count: measured 2026-11-05 on 120 recorded objnav prompts
  (Ollama 0.21.1, qwen2.5:3b, 8,192 context): 1 server x 4 slots 0.81/s, x 16
  slots 0.71/s, 2 servers x 8 slots 1.32/s. The time goes to a per-request
  server-side cost of about 1.4 s, not to the GPU. So `--llm-servers N` starts N native servers on
  consecutive ports from `LLM_BASE_URL`'s, warms each at the run's context
  window, and deals the shards over them (a started shard keeps the server its
  `run.json` froze). The merge compares the method without the server's
  address; the merged `run.json` lists every address under `llm_base_urls`.
* **Record the LLM answers outside the root** (`LLM_ANSWERS_RECORD`): the
  aggregate reads every subdirectory of the root as a scene, so a store inside
  it makes the summary `complete: false`.
* `--output <root>` resumes: merged scenes and finished shards are kept, started
  shards continue (`--resume`), the rest start.
* **The stall watchdog runs on shards too** (`--stall-minutes N`, default 30;
  since 2026-11-11): a shard whose `live.json` / `progress.json` /
  `episodes.jsonl` / console log have not changed for that long is killed with
  its session and recorded as exit -9 (`shard_pool.run_pool`); its slot goes to
  the next shard, its scene stays unmerged, and `--output <root>` continues it
  from its last finished episode.
* **`--record-log-only`**: every episode's `steps.jsonl` (pose, action, the
  method's snapshot and the command of every decision), `trajectory.csv`, map
  journal and `metrics.json` under `<shard>/recordings/<key>/`, and NO video --
  no dashboard frame is rendered, no FFmpeg is needed. It is what
  `single_storey_failures.py` (which also reads a merged scene's `shard_*/`
  recordings) and `review_recording.py` read; about 33 KB a step, gzip ~9:1. A
  video of any episode is re-made afterwards by replaying it with its recorded
  answers. `--full-records` beside it keeps the policy's whole per-episode
  report (its event logs included) in `episodes.jsonl`, ~100-300 KB a row.
* **`resource_monitor.py`** (`record out.csv --interval 15`, `summarize out.csv`)
  samples GPU utilisation, VRAM, CPU, RAM, load and the live worker count while
  a campaign runs, for the report's utilisation line.
* **`storey_report.py`** (`stairs-off=runs/A stairs-on=runs/B`): SR / SPL /
  DTG overall and on the same-storey against the cross-storey episodes (a goal
  view point within 0.5 m of the start height, entry 043's rule, and the level
  survey's `start_level == goal_level` as a second reading), per category and
  per category by storey, paired against the first arm (bootstrap dSR / dSPL,
  McNemar, flips), and the storey transitions per episode: the floor atlas's
  `floor_arrival` events and completed stair traversals from the full records,
  and the settled-level changes of each recording's trajectory, with how many
  cross-storey episodes ended on the goal's storey.

### A 14B in both roles on the 4090 (measured 2026-11-11)

`qwen2.5:14b-instruct` at 4 request slots and an 8,192-token context takes
15.1 GB (8.1 GB weights, 6.0 GB f16 KV cache, 0.3 GB compute). Unconstrained it
answers about 0.5 requests/s with 1.2k-token prompts and 400-token answers (203
tokens/s aggregate at 4 concurrent, 88 tokens/s single-stream), but **the
client asks Ollama for `format: json`, and the JSON grammar's sampling is CPU
work serialised in Ollama's runner**: with 150-token answers the same server
answers 0.93 requests/s without the format and 0.36 with it (33 against 48
tokens/s per stream plus a fixed cost of some seconds a request); 8 slots give
0.39. In the campaign the rate was ~29 answers/min (0.5/s) with 7 workers and 4
slots, with 12 workers and 4 slots, and with the KV cache in q8_0 -- so the LLM
is saturated by 7-8 renderers, more workers only queue (mean 8.4 requests in
flight at 12, p50 latency 13 s), and the card's compute stays ~40 % busy while
its memory is full. One MP3D ground-truth renderer takes 270-1,180 MiB by scene
(~820 mean), so a 0.90 cap holds 7-8 beside the LLM at Ollama's defaults; the
campaign runs ~110 episodes/h (~16 answers an episode).

### Two models under one prompt, one command: `run_gt_3b_then_14b.sh` (2026-11-19)

The model comparison on the full val split under ground-truth perception:
`qwen2.5:3b-instruct` in both LLM roles, then `qwen2.5:14b-instruct`, both asked
under the 3B's prompt version `facts-qwen2.5-3b-v1` (digest `2d025f01...`) so
that only the model differs. The 14B asked under the 3B's prompt is recorded
(`node_oracle.model.prompt_tuned_on_it: false`), never refused.

```bash
mp3d/run_gt_3b_then_14b.sh --detach     # start, or RESUME after any interruption (the same command)
mp3d/run_gt_3b_then_14b.sh --status     # per stage: smoke, rows, live shard progress and ETA, governor, commits
mp3d/run_gt_3b_then_14b.sh --stop       # stop for good (STOP file + TERM; the launcher stops shards and servers)
```

Per stage: checks (branch `feat/objnav-habitat-mp3d-nadav` containing
`a13214e2`, the registered prompt's digest, both models in the native store --
pulled if absent --, `nvidia-smi`, no stale launcher), a 3-episode smoke (the
first episode of `pLe4wQe7qrG`, `zsNo4HB9uLZ`, `oLBMNvg9in8`: 3 rows, no agent
error, the digest and model in `run.json`, answers recorded from that model),
the sharded campaign (shards of 10; after each launch: every Ollama server
answers, workers alive, the first `run.json`'s `node_oracle.prompt` digest), a
relaunch-to-resume when a launch ends short of a complete split, and finally
`results_summary.py` into `mp3d/results/<campaign>_stage<N>_<model>/` (SR and
SPL overall and per category, episodes, model, prompt version and digest,
source commit, start and end times), committed and pushed -- never forced; a
failed commit or push is logged and the next stage starts. Its own Ollama
servers listen from port 11440, so the resident 11434/11435 are left alone.

Parallelism, from the measurements above (target <= 90 % of GPU, VRAM, CPU and
RAM): the 3B on **2 servers x 5 slots, <= 16 workers** (each server's JSON
sampling is serial, so a second server is worth more than slots); the 14B on
**1 server x 4 slots, <= 7 workers**. The pool's VRAM gate (`VRAM_CAP` 0.88,
850 MiB a renderer) delays starts; `jobs_governor.py` lowers the live jobs file
when a 2-minute mean of CPU, RAM, VRAM or GPU compute passes 90 % and raises it
back after 30 calm minutes.

Logs, under `runs/mp3d_v1_gt_3b_vs_14b/`: `support/orchestrator.log` (every
check and decision), `support/stage<N>_launcher.out`, `stage<N>_commands.txt`
(the exact environment and command of every launch),
`stage<N>_utilization.csv` (`../utilization_log.py`, 15 s),
`stage<N>_governor.csv`; per stage root the launcher's `run.log`,
`preflight.json`, `gpu_plan.json`, `source-commit.txt`, `source-changes.diff`,
`ollama*.log`, per scene `run.json` and full `episodes.jsonl`, per episode
`steps.jsonl` + `trajectory.csv` + map journal (`--record-log-only`); and beside
each root `<root>_llm_answers/` (every LLM request -- system and user prompt,
model, episode -- and its reply).

### The fixed single-storey set (the study of 2026-11-10)

The method's single-storey potential is measured on one fixed list of episodes,
re-run after every change, from a frozen copy of the source:

```bash
mp3d/run_single_storey.sh baseline --jobs 3                      # the current code, every LLM answer recorded
mp3d/run_single_storey.sh flat_p --policy-config mp3d/sets/policy_flat_p.json   # the oracle ablation on the same episodes
# a later arm records into its own store and REPLAYs the earlier ones (two concurrent runs must not share a record dir)
STORE=$PWD/runs/ss_fix_wedge_llm_answers REPLAY=$PWD/runs/single_storey_v1_llm_answers \
    mp3d/run_single_storey.sh fix_wedge --policy-config mp3d/sets/policy_fix_wedge.json --jobs 2
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_report \
    baseline=runs/ss_baseline flat-p=runs/ss_flat_p                # SR, SPL, per-category SR, paired flips
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_failures runs/ss_baseline
```

The per-fix arms of the 2026-11-10 study are the configs under `sets/`:
`policy_fixes_off.json` turns every single-storey fix off (the baseline's
behaviour under today's perception), `policy_fix_{wedge,dining,shape}.json`
turn one on at a time; no config is every fix on. (`policy_fix_closing.json`
-- the raised target's steps after a decided STOP and the frontal standoff --
was deleted on 2026-11-16 with the two rules it switched; the other configs
lost their `target_closing` block the same day, and a config that still names
those settings is refused.) The two labeller rules
(`dining`, `shape`) are off by default since 2026-11-11 -- their configs turn
them on for a comparison, nothing else does.

* **The set** is `sets/single_storey_flat_v1.json`: 180 same-storey episodes
  drawn (seed 17, uniform) from the 1,281 val episodes that start on a storey,
  whose nearest goal view point lies on it within 0.2 m of the start's height,
  whose shortest path never climbs or drops 0.2 m, and whose scene is not
  split-level (x8F5xyUWy9e, X7HyMhZNoso, EU6Fwq7SyZv are out). `single_storey_set.py`
  regenerates it from the shipped level survey `sets/mp3d_val_levels.csv`
  (`hm3d/episode_levels.py` over MP3D val); the `.meta.json` beside it holds the
  rules and the per-scene / per-category counts. Change it only on purpose:
  every comparison made on it depends on its being the same list.
  The [GT-only frontier VLM vs. room LLM comparison](results/gt_rpt_pair_single_storey_v1_seed17/summary.md)
  reports all 180 matched starts, per-episode metrics, and paired uncertainty.
* **`run_single_storey.sh <name>`** rsyncs the tracked source into
  `~/GIT/TheAgency_snapshots/<name>/` once and runs the launcher from there (a
  scene child imports the repository at its own start, so a campaign that runs
  for hours while the tree is edited would mix two methods), with ground-truth
  perception, `--record`, the resident Ollama, and **one LLM answer store for
  every run of the study** (`runs/single_storey_v1_llm_answers`,
  `LLM_ANSWERS_RECORD`): a question an episode has already had answered gets the
  same answer in every later run, so two runs part only where a change makes
  them ask something new. `--resume` continues an interrupted campaign;
  `--policy-config` passes a JSON of `RPTSettings` overrides.
* **`--stall-minutes N`** (campaign, default 30): a scene child whose
  `live.json` / `progress.json` / `episodes.jsonl` / console log stop changing
  for that long is killed with its session and recorded as exit -9 -- a worker
  that lost its main thread keeps its CUDA threads and never exits
  (`LESSONS.md` 2026-11-09); `--output <root>` then resumes the scene.
* **`single_storey_report.py`** tabulates every campaign named against the
  first (SR, SPL, SoftSPL, DTG, steps, STOP / limit, wrong STOPs within 1 m,
  the paired dSR and dSPL with bootstrap intervals, the exact McNemar p, the
  flips) and the per-category and per-scene SR side by side, over the episodes
  all of them ran. **`single_storey_failures.py`** reads each failure's
  recording and the published goals: the STOP's reason, the agent against the
  view points, every lock's first box and memory, a first mechanism tag, with
  the ground-truth sensor's phantom ids (`gt_phantom_*`) counted apart from the
  method's failures; a near-miss STOP is also read against the goal's `.house`
  box (`--scene-dir`): the generator's own distance, how far beside the
  object's width the agent stands and how far off its normal -- `beside`,
  `in_band` or `far`. `--locks` dumps every lock with its visible extent (the
  sliver data); the labeller tally covers every episode.

### The same set with no GPU and no model: a paired check of a change (since 2026-11-16)

A change to the closing, the planner or the map is checked on the same 180
episodes before and after it, with nothing else differing between the arms --
and without the GPU, which is usually taken:

```bash
# arm "before": the code of a frozen copy of the commit before the change (rsync of the tracked tree)
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.cpu_stub_campaign \
    --output runs/cpu_ab/before --tree ~/GIT/TheAgency_snapshots/before
# arm "after": this tree (snapshot it too if it will be edited while the arm runs)
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.cpu_stub_campaign --output runs/cpu_ab/after
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.stuck_report \
    before=runs/cpu_ab/before after=runs/cpu_ab/after          # refused steps, closing counters, identical trajectories
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_report \
    before=runs/cpu_ab/before after=runs/cpu_ab/after          # SR, SPL, per-category, the flips
```

* **`cpu_stub_campaign.py`** runs one process per scene with ground-truth
  perception, the frontier explorer, seed 17 and lean rows, under Mesa's
  llvmpipe (`--software-render`, `/dev/dri` hidden in a user namespace), and
  starts a scene only when its RAM is free (`ram_gate.RamGate`: 2-6 GiB a scene, a
  4.5 GiB reserve) -- an arm takes about 3.5 hours on a 30 GB laptop, two
  arms side by side about 5. `--only` runs a subset, `--shard N` cuts long
  scenes, and what follows `--` goes to every scene's `mp3d.run` (`--
  --policy-config rule_off.json`: an arm with one rule switched off).
* **`../stub_llm.py`** stands in for the language model: every room is
  `living_room` at 0.8 and the nodes of an oracle prompt get an equal share
  of 90 %. It reads the node ids off the prompt and nothing else, so a change
  to the prompt's wording leaves its answers, and a check's arms, as they
  were; the frozen configuration names it (`method.llm_identity["stub"]`).
  **The numbers of such an arm are not results of the method** -- the
  stand-in orders the rooms by travel cost and exploration state alone --
  only the paired differences are.
* **`stuck_report.py`** is the paired table of what the outcome does not
  show (refused forward steps, the closing's releases, relaxed standing
  points, blind verification steps) and the pairing's own check: with a
  deterministic stand-in, an episode in which the change never acted must be
  the identical trajectory in both arms (same steps, same count of every
  action, same path length, same place at the end). On 2026-11-16 all 123
  episodes in which neither removed closing rule had acted were identical.
* **One change per arm.** A change that is a setting is isolated by an arm
  of its own (`-- --policy-config`), run only on the episodes in which the
  arms on either side of it differ, plus controls: where they are identical
  the change did not act, and the arm in between is the same trajectory. A
  later root overrides an earlier one in both reports
  (`after=runs/x/after,runs/x/rerun`), which is how a partial re-run is read.
  The check of 2026-11-16, in two rounds. In the morning, four arms, each
  differing from the one before it in one item of that day's change:

  | per episode (180) | before | no evaluator-shaped rules | + blind-step fix | + 0.20 m retry clearance |
  |---|---|---|---|---|
  | success rate | 78.9 % | 74.4 % | 73.3 % | 71.1 % |
  | SPL | 0.346 | 0.332 | 0.325 | 0.322 |
  | refused forward steps | 7.95 | 7.87 | 6.79 | 5.92 |
  | won / lost against the arm before (exact McNemar p) | -- | 0 / 8 (0.008) | 0 / 2 (0.5) | 2 / 6 (0.29) |
  | trajectories identical to the arm before | -- | 124 | 175 | 138 |

  The last two changes were withdrawn on those numbers (the blind-step fix
  to be handled on its own; the retry clearance stays the body radius), so
  the tree's base is the second column. In the evening the STOP criterion
  was measured on it (`target_closing.stop_within_m`, `stop_measure`; the
  runtime README has the rule), again one change an arm:

  | per episode (180) | STOP on the deciding frame (the base) | + the walk, `planar` 1.0 m | + a decimetre inside the zone, `planar` 0.9 m | + MP3D's measure, `euclidean` 0.9 m (the default) | `euclidean` 1.0 m |
  |---|---|---|---|---|---|
  | success rate | 74.4 % | 77.8 % | 80.6 % | **82.2 %** | 80.6 % |
  | SPL | 0.332 | 0.341 | 0.350 | **0.354** | 0.352 |
  | failed STOPs (of 158) | 24 | 18 | 13 | 10 | 13 |
  | won / lost against the base (exact McNemar p) | -- | 6 / 0 (0.03) | 11 / 0 (0.001) | 14 / 0 (0.0001) | 11 / 0 (0.001) |
  | trajectories identical to the base | -- | 145 | 77 | 64 | 115 |

  For reference, on the same base: the approach as it was at `f257a0ce`
  (steps under a RAISED target only, sized on recorded STOPs) without its
  frontal standoff is 79.4 % / 0.348 -- 9 won, none lost -- and with the
  standoff 78.9 % / 0.346; the criterion wins the same nine and five targets
  on the floor.

  The partial arms (the two `planar` ones and `euclidean` 1.0 m) were run on the
  episodes in which they can differ from the base -- a STOP decided at or
  beyond that distance by that measure, which the full arm's `stop_decision`
  record gives for every criterion at once -- and on controls.

  The stand-in's numbers are not results of the method; the differences are
  what each item moved (CHANGELOG of 2026-11-16).
* Keep the arms. The check of 2026-11-15 was run from a scratch folder and
  its launcher was not in the repository; its second arm could be reused a
  day later only because the rows had been kept and the launcher, rebuilt
  here, reproduced them (33 of 33 trajectories on two scenes).

### Freeze, then evaluate (for a number you intend to publish)

```bash
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.frozen_eval freeze \
    --lock locks/mp3d_v1_val.json --note "developed on Gibson val and HM3D v1 val; MP3D never run before this freeze" \
    -- --episodes-dir ~/datasets/objectnav/mp3d/v1/val --scenes-dir ~/datasets/scene_datasets --detector-url http://127.0.0.1:18093
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.frozen_eval run \
    --lock locks/mp3d_v1_val.json -- --episodes-dir ... --scenes-dir ... --detector-url ... --output runs/mp3d_frozen
```

The lock pins the sources, the built policy, the dataset manifest, the runtime
versions, the protocol and the ordered selection; `run` refuses any drift and
any selection but the complete split.

### Reading the numbers

* **SR / SPL** are what every MP3D paper reports; both are means over all
  2,195 episodes, failures included. **DTG** is habitat-lab's
  `distance_to_goal` (geodesic to the nearest view point); **SoftSPL** is
  supplementary.
* **Two success radii.** Ours is the official 0.1 m. ApexNav's released
  configs use 0.2 m; `report.py` re-scores the *same rows* at 0.2 m (which can
  only add successes) so a like-for-like row exists under their table. Both
  tables and the number of episodes that flip are in `comparison.md`.
* **`complete`** in `campaign_summary.json` and `full_split_completed` in
  `audit.json` are the only things that make a number a benchmark result.

## If something refuses to start

* *"the MP3D val installation is incomplete"* — the launcher printed the
  problems; `assets.py --instructions`.
* *"Only N of M checked starts reproduce the publisher's own geodesic"* — the
  navmesh check (`VALIDATE_STARTS`). On the published release every one of
  the 2,195 starts is reachable and 2,191 reproduce the publisher's number to
  1 mm (measured 2026-11-04, habitat-sim 0.2.4), so a failure here means the
  scenes are not the ones the episodes were generated on, or the protocol's
  `navmesh` was changed.
* *"the detector at ... serves another vocabulary"* — a Gibson or HM3D detector
  is on the MP3D port; set `DETECTOR_PORT`.
* *"Missing .../mp3d.scene_dataset_config.json"* (ground-truth perception) —
  `assets.py --fetch-semantic-config`; the launcher does it itself.
* *"... lack their Matterport3D annotations"* — a scene's `.house` or
  `_semantic.ply` is missing; both ship in `mp3d_habitat.zip`.

## Protocol reference

Every field of `MP3DProtocol` and its source is tabulated in
[`PROTOCOL_AND_TUNING.md`](PROTOCOL_AND_TUNING.md), with the measured navmesh
evidence and the deviations we name rather than apply.

## Files

| File | Role |
| --- | --- |
| `run_benchmark.sh` | the one-button launcher (data → services → preflight → campaign → reports) |
| `protocol.py` | every MP3D number, versioned; `MP3D_V1`, `protocol_for`; `GOAL_ZONES` -- where the generator put a goal's view points (1.0 m from the object's box in 3-D from the floor), from which the method's STOP criterion is derived |
| `dataset.py` | the published release: episodes (sharded, or val_mini's single file), goals, scene assets, semantic assets and config, manifest |
| `env.py` | `MP3DEnv`: the shared `../habitat/env.py::HabitatObjNavEnv` (reset, step, measure, start validation, the oracle hand-off) named for MP3D, with the one MP3D override -- an instance is named by both its `.house` category and its mpcat40 class |
| `run.py` | the MP3D `parser`, `prepare` (preflight) and `ground_truth_issues`; `main` is the shared `../habitat/entrypoint.py::run_main`, and the method's defaults below `--policy-config` the shared `method_defaults(protocol)` |
| `campaign.py` | one process per scene, the aggregate, the stall watchdog (`--stall-minutes`); reuses `../scene_pool.py`, `../smoke_plan.py` and `../campaign_common.py` |
| `sharded_campaign.py` | `--shard-episodes`: every scene as shards on the pool, each scene merged when its shards finish |
| `shard_plan.py` | the shards (interleaved across scenes), their LLM servers, the plan file, the per-scene merge |
| `shard_pool.py` | the VRAM-gated pool with a live worker cap; VRAM only ever delays a start |
| `campaign_status.py` | `--status` of a sharded root: done/total, SR/SPL, shards running, throughput and ETA, per scene |
| `assets.py` | installation check, public episode fetch, scene instructions |
| `report.py`, `sota.py` | `comparison.md` / `audit.json`, the 0.2 m re-scoring, the transcribed paper rows |
| `frozen_eval.py` | freeze a configuration, run the published split under it |
| `single_storey_set.py`, `sets/` | the fixed single-storey evaluation set (180 same-storey episodes in the flat houses) and the level survey it is drawn from |
| `run_single_storey.sh` | a campaign on that set from a frozen source snapshot, one LLM answer store for the whole study |
| `single_storey_report.py`, `single_storey_failures.py` | baseline-versus-steps table (paired); per-failure mechanism from the recordings, the lock / sliver data, the labeller tally |
| `cpu_stub_campaign.py`, `ram_gate.py`, `stuck_report.py`, `../stub_llm.py` | the paired before/after check on that set with no GPU and no model: the CPU launcher, the pool's start gate on free RAM (the first pending scene that fits), the stuck-indicator table with the identical-trajectory count, the stand-in for the language model |
| `smooth_replay.py` | visualization-only smooth replay of a recording |
| `run_gt_3b_then_14b.sh`, `results_summary.py`, `jobs_governor.py`, `results/` | the one-command 3B-then-14B ground-truth comparison (section 3): checks, smoke, sharded campaign, resume; the committable per-stage summary; the live-jobs governor that keeps the machine under 90 % |
| `../ground_truth/mp3d_categories.py` | the ground-truth perception's MP3D table (mpcat40 goal classes, context words) |
| `../tests/test_mp3d*.py` | the adapter without Matterport3D, habitat-sim or a GPU (`test_mp3d_ground_truth.py`: the oracle path) |
