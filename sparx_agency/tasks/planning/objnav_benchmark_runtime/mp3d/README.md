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
| Step budget 500; `l`, `d0`, `dT`, `p`, SR, SPL, SoftSPL as habitat-lab computes them | `env.py`, with the distance from `../hm3d/geodesic.py` (habitat-lab's measure, shared) |
| The published episodes, goals and scene files; the 21-category vocabulary | `dataset.py`; `core/planning/objnav/labels/datasets/mp3d.py` |
| Kinematic self-check bounds for habitat-sim's recomputed navmesh | `protocol.py` → `kinematics()` |

The policy is built by the one shared `policy_factory.build_policy`; the only
things this adapter tells it are the label table and the protocol's body.
Nothing here came from an MP3D score.

## One command (the launcher)

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

### The fixed single-storey set (the study of 2026-11-10)

The method's single-storey potential is measured on one fixed list of episodes,
re-run after every change, from a frozen copy of the source:

```bash
mp3d/run_single_storey.sh baseline --jobs 3                      # the current code, every LLM answer recorded
mp3d/run_single_storey.sh flat_p --policy-config mp3d/sets/policy_flat_p.json   # the oracle ablation on the same episodes
# a later arm records into its own store and REPLAYs the earlier ones (two concurrent runs must not share a record dir)
STORE=$PWD/runs/ss_fix_closing_llm_answers REPLAY=$PWD/runs/single_storey_v1_llm_answers \
    mp3d/run_single_storey.sh fix_closing --policy-config mp3d/sets/policy_fix_closing.json --jobs 2
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_report \
    baseline=runs/ss_baseline flat-p=runs/ss_flat_p                # SR, SPL, per-category SR, paired flips
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_failures runs/ss_baseline
```

The per-fix arms of the 2026-11-10 study are the configs under `sets/`:
`policy_fixes_off.json` turns every single-storey fix off (the baseline's
behaviour under today's perception), `policy_fix_{closing,wedge,dining,shape}.json`
turn one on at a time; no config is every fix on. The two labeller rules
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

### The warm-up study (2026-11-11)

Does the episode's first look pay for itself better as the ten-turn spin in place
or as a look that may walk a few steps first? `RPTSettings.warmup.mode == "vantage"`
(`core/planning/object_search/exploration/vantage_warmup.py`; the runtime README,
"Episode discovery") is the alternative, and this is how it is measured, with
little compute beside a running campaign:

```bash
# the subset: single-storey episodes the ground-truth campaign has already FINISHED, N of them, stratified by scene;
# the campaign's own rows are the spin arm (same commit, same configuration, same model) and cost nothing
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.warmup_study select \
    --campaign runs/gt_oracle_mp3d_2195ep_14b_<stamp> --set mp3d/sets/single_storey_flat_v1.json --n 40 --seed 17 \
    --out mp3d/sets/warmup_study_v1.json
# one arm: a frozen snapshot of the tree, the campaign's resident 14B at :11435, its answer store replayed, the CPU renderer
REPLAY=runs/gt_oracle_mp3d_2195ep_14b_<stamp>_support/llm_answers \
    mp3d/run_warmup_arm.sh p1_b25 --policy-config mp3d/sets/policy_warmup_p1_b25_gt.json --jobs 3
# the table: the spin (the campaign) against every arm, over the subset
"$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.warmup_study report \
    --set mp3d/sets/warmup_study_v1.json spin=runs/gt_oracle_mp3d_2195ep_14b_<stamp> p1_b25=runs/wu_p1_b25 ... \
    --markdown runs/warmup_study.md
```

* **The arms** are the configs `sets/policy_warmup_p<points>_b<budget>_gt.json`: `p0_b15` (a greedy sweep in
  place, never a walk), `p1_b15`, `p1_b25` (one vantage point), `p3_b25`, `p3_b40` (several). `detection_range_m`
  is 5.0 m in every one because the study runs under ground-truth perception, which detects to the depth range;
  the setting's default, 3.5 m, is for a detector.
* **`run_warmup_arm.sh <arm>`** rsyncs the tree into `~/GIT/TheAgency_snapshots/wu_<arm>/` once and runs the
  launcher from there (`--perception ground-truth --episode-ids-file <subset> --record-log-only --full-records`),
  with `HABITAT_RENDER=software` -- Mesa's llvmpipe, no VRAM, about a second an action -- so the campaign's
  renderers and its pool gate are left alone; `REPLAY` serves the campaign's answers to any question it already
  asked in an episode, `STORE` (default `runs/wu_<arm>_llm_answers`) records the new ones. Each arm's new LLM
  calls are the one cost the campaign feels.
* **The local-exploration arm** of the same study (experiment 2) is `sets/policy_explore_viewpoint_gt.json`:
  `{"room_search": {"explore_goals": "viewpoint", "explore_look": {"detection_range_m": 5.0}}}` -- the explore
  visit stands where the camera sees the most of the room's unknown instead of walking to its frontier cells
  (the runtime README, "The room-search loop"); run and tabulated exactly like a warm-up arm.
* **`warmup_study.py report`** prints SR, SPL, SoftSPL, DTG, steps and the paired statistics against the first
  arm (`single_storey_report`), then the warm-up's own measures from the records and recordings: actions it took,
  walks, metres walked, the observed area on its last action and at actions 10 / 20 / 30 / 50, the first action
  the target's class was in the frame (median; share within 30; share never) and the first takeover. Every
  number is over the episodes all the arms ran.

### The same study on a laptop, one arm per command (`look_study.sh`)

`warmup_compare.sh` (experiment 1) and `explore_compare.sh` (experiment 2) are `look_study.sh warmup` and
`look_study.sh explore`: the same fixed subset (`sets/warmup_study_v1.json`), ground-truth perception, one
3B model in both roles at temperature 0 and seed 17, one arm per command. The **`without` arm** (the method
as it stands: the ten-turn spin and the frontier walk) is run once and **records every LLM answer**
(`runs/look_study/llm_answers/without`); every other arm **replays that store** and records only what it asks
anew, so two arms part only where the change under test acts. `--dry-run` prints the plan -- arm, config,
episodes by scene, model and server, stores, renderer, workers, a time estimate per arm -- checks the
interpreter, the data, the Ollama server, the GPU and the disk, and starts nothing; every problem is one line
naming what is missing and what to do. Each arm runs from a frozen snapshot (`~/GIT/TheAgency_snapshots/ls_<arm>`),
so the checkout may be edited meanwhile; `--resume` continues an interrupted arm.

```bash
cd ~/GIT/TheAgency
M=sparx_agency/tasks/planning/objnav_benchmark_runtime/mp3d
$M/warmup_compare.sh without --dry-run                 # the plan and the checks; nothing starts
$M/warmup_compare.sh without --jobs 2                  # the baseline: records the answers (run once, first)
$M/warmup_compare.sh p1_b25                            # arms in priority order; each is its own command
$M/warmup_compare.sh p3_b25
$M/warmup_compare.sh p1_b15
$M/warmup_compare.sh p3_b40
$M/warmup_compare.sh p0_b15
$M/warmup_compare.sh report --explain                  # runs/look_study/warmup_report.md (+ .json)
$M/explore_compare.sh explore_viewpoint                # experiment 2, its own arms, the same `without`
$M/explore_compare.sh explore_viewpoint_transparent
$M/explore_compare.sh report --explain                 # runs/look_study/explore_report.md
```

The report (`warmup_study.py report`) gives SR, SPL, SoftSPL, DTG and steps per arm with the paired statistics
against `without` (dSR and dSPL with bootstrap intervals, the exact McNemar p, the success flips, the episodes
whose SPL got better or worse), the warm-up's measures (actions, walks, metres, the area uncovered at its end and
at actions 10 / 20 / 30 / 50, the first sight of the target, the first takeover), the explore visits' measures,
and with `--explain` one line per changed episode: both outcomes, steps, DTG, terminations, warm-ups, first
sights, LLM answers asked anew, and the action at which the two runs parted with the phase each was in.

Data the laptop needs: the MP3D val episodes (`~/datasets/objectnav/mp3d/v1/val`, fetched by the launcher when
absent), the eight scenes of the subset under `~/datasets/scene_datasets/mp3d/<scene>/` with `.glb`, `.navmesh`,
`.house` and `_semantic.ply` (`mp3d_habitat.zip`; the semantic scene-dataset config is fetched), the habitat-sim
0.2.4 environment (`HAB_PY`) and an Ollama with `qwen2.5:3b-instruct` (pulled by the launcher when absent).
Set `MP3D_SCENES_DIR` / `MP3D_EPISODES_DIR` / `HAB_PY` / `LLM_BASE_URL` when the laptop's layout differs.

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
| `protocol.py` | every MP3D number, versioned; `MP3D_V1`, `protocol_for` |
| `dataset.py` | the published release: episodes (sharded, or val_mini's single file), goals, scene assets, semantic assets and config, manifest |
| `env.py` | the simulation driver behind `ObjNavEnv`: reset, step, measure, start validation |
| `run.py` | preflight and evaluation CLI (the shared `run_evaluation`) |
| `campaign.py` | one process per scene, the aggregate, the stall watchdog (`--stall-minutes`); reuses `hm3d/scene_pool.py` and `hm3d/smoke_plan.py` |
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
| `look_study.sh`, `warmup_compare.sh`, `explore_compare.sh`, `look_study.py` | the warm-up and local-exploration comparisons as one arm per command: the plan and dry run (checks, time estimate), the `without` arm recording the LLM answers the others replay, the report |
| `run_warmup_arm.sh`, `warmup_study.py`, `sets/warmup_study_v1.json`, `sets/policy_warmup_*.json`, `sets/policy_explore_viewpoint*.json` | the warm-up study of 2026-11-11 beside a running campaign: one arm from a frozen snapshot on the CPU renderer; the subset drawn from the campaign's finished episodes; the comparison table with the warm-up's and the explore visits' measures and `--explain`; the arms' configs |
| `smooth_replay.py` | visualization-only smooth replay of a recording |
| `../ground_truth/mp3d_categories.py` | the ground-truth perception's MP3D table (mpcat40 goal classes, context words) |
| `../tests/test_mp3d*.py` | the adapter without Matterport3D, habitat-sim or a GPU (`test_mp3d_ground_truth.py`: the oracle path) |
