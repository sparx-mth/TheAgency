# HM3D protocol, tuning and interpretation

## What the cited papers actually establish

- **ApexNav**, arXiv:2504.14478v3, §V-A / Table I, p.6: HM3D-v1
  (HM3D-Semantics v0.1, 2022 challenge, **2000 episodes / 20 scenes**),
  HM3D-v2 (v0.2, **1000 / 36**) and MP3D (**2195 / 11**, 21 categories), all
  `split: val`. Its three released eval configs are byte-identical apart from
  the data path, and they set **`success.success_distance: 0.2`**,
  `allow_sliding: false`, `forward_step_size: 0.25`, `turn_angle: 30`,
  `radius: 0.18`, `depth 0.0–5.0 normalised`, `max_episode_steps: 500`, and a
  deterministic unshuffled episode order. The paper states the 0.2 m figure
  twice and never compares it to habitat-lab's 0.1 m.
  It discloses which rows it measured: *"As only InstructNav reported HM3Dv2
  results, we re-evaluated several open-source methods under our settings for
  fairness."* So on HM3D-v2, L3MVN, VLFM, VLFM\* and SG-Nav are ApexNav's own
  re-runs and only InstructNav is quoted.
- **SG-Nav**, arXiv:2410.08189v1, §4.1 / Table 1, p.7: MP3D, **HM3D (2022, so
  v1)** and RoboTHOR. Its released HM3D config keeps
  **`SUCCESS_DISTANCE: 0.1`** (it uses 0.2 only for MP3D) and runs habitat-lab
  0.2.1. The paper prints a **0.90 m** camera height, but the config its own
  mapping module reads is **0.88 m**; it also raises `MAX_DEPTH` to 10 m for
  HM3D. `d_s` is never given a number in the paper text. Table 2 adds SoftSPL.
- **OSG ("Open Scene Graphs")**, arXiv:2508.04678v1, §7.1.1, p.11: HM3D-Semantics
  **v0.1**, but *"doubling the number of evaluation episodes sampled from the
  validation set to 400 … sampled evenly across the validation set's 20
  scenes"*. No seed, no sampling code, no episode-id list is published. Its task
  setting is deliberately harder and different: **RGB only**, velocity commands,
  no ground-truth localisation, goal detection by an LLM over GroundingDINO and
  BLIP-2 rather than a STOP action, and **no step cap is stated anywhere in the
  paper**. Its DTG is defined in one sentence — *"Average distance of the agent
  from the goal instance at the end of each episode"* — with no statement of
  geodesic versus Euclidean and no treatment of unreachable goals.

**`TF` and `NM` are not metrics.** They appear only in OSG's *Gibson* table
(Table 2), as property checkmarks glossed in its caption: *training-free* and
*non-metric*. OSG's HM3D table has four columns — Method, SR, SPL, DTG — and no
such flags. Our method is TF; it is not NM.

## The disagreements between these papers

Two papers print different numbers for the same baseline on the same split:

| method, HM3D-v1 | ApexNav Table I | SG-Nav Table 1 |
|---|---|---|
| L3MVN | 50.4 / 23.1 | 48.7 / 23.0 |
| OpenFMNav | 54.9 / 24.4 | 52.5 / 24.1 |
| VLFM | 52.5 / 30.4 | 52.4 / 30.3 |

Both are recorded in [`sota.py`](sota.py) with their sources rather than one
being silently preferred. At least three things differ underneath:

1. **Success radius** — 0.2 m (ApexNav) versus 0.1 m (SG-Nav). Loosening it can
   only raise SR and SPL.
2. **What each row even is** — ApexNav's re-evaluation disclosure covers
   **HM3D-v2 only**. Its HM3D-v1 baseline rows are quotes from the originating
   papers, produced under those authors' settings, so a v1 row is ApexNav's
   citation rather than ApexNav's measurement — which is the likeliest reason
   its L3MVN and OpenFMNav figures differ from SG-Nav's for the same method.
3. **Scenes** — ApexNav's README symlinks `hm3d_v0.2 -> hm3d` over a single
   download, which makes one of its two HM3D columns run against the other
   version's geometry for the four validation scenes HM3DSem v0.2 re-annotated.

The navmesh is **not** one of these differences, despite the habitat-lab
version gap: the recompute lives in habitat-sim, whose
`Simulator._config_pathfinder` rebuilds the mesh at the agent's radius and
height whenever the shipped one disagrees. Both papers run a habitat-sim that
does this, and so do we — which is also why our `l` should reproduce the
publisher's own `info.geodesic_distance`.

None of this is an accusation. It means a single leaderboard-style table across
these papers is not a like-for-like ranking, and our own row must say which
convention it was produced under. That is why `report.py` prints two tables from
one run and why every reported row carries a `notes` field.

## What this implementation does

1. Runs the **complete published validation split** — 2000 episodes / 20 scenes
   for v1, 1000 / 36 for v2 — with no substituted or cherry-picked starts. The
   loader refuses a release whose counts differ from the published ones.
2. Scores under habitat-lab's own rule (STOP, geodesic to the nearest goal view
   point < 0.1 m) and **re-scores the same rows** at ApexNav's 0.2 m, so both
   numbers come from one run and one simulation.
3. Measures `l`, `d0`, `dT` and `p` exactly as habitat-lab's `DistanceToGoal`,
   `SPL` and `SoftSPL` do, on the navmesh the episode is actually navigated on,
   and records that navmesh's settings, navigable area and island count.
4. Reports how closely our `l` reproduces the publisher's stored
   `info.geodesic_distance`. This is a self-test of navmesh provenance, frame
   handling and view-point extraction at once. It is **reported, not asserted**:
   a hard equality check would be untestable against a release whose generator
   used a different definition.
5. Refuses, up front and explicitly, an episode whose goals are unreachable —
   the harness cannot score a non-finite `l`, and habitat-lab's own SPL returns
   NaN for one rather than 0, which would poison a mean instead of failing.
6. Records the source hash, the dataset manifest (every episode file, mesh and
   navmesh hashed), the detector checkpoint and vocabulary, the LLM identity,
   the runtime versions, the seed and the motion tolerances. A resumed run
   cannot quietly change any of them.

## Our own exposure must be disclosed

Nothing in this package has yet been run against HM3D — the scene meshes are not
provisioned — so at the time of writing there is **no HM3D development
contamination to disclose**. That is a fact with a shelf life: the moment a
validation episode is inspected while tuning, it stops being untouched, and no
later flag or rename undoes it.

Keep it that way deliberately:

- Develop and tune on the **train split**, which is installed
  (`~/datasets/objectnav/hm3d/<version>/train`, 80 scenes for v1 and 145 for
  v2), or on `val_mini` (2 scenes, 30 episodes) declared as a development set —
  not on repeated full-validation feedback. HM3D's train split is roughly
  **seven million** episodes, some 48,000 per scene, so a development run must
  be bounded and `run.py` refuses one that is not:

  ```bash
  python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
    --version v2 --split train --episodes-per-scene 2 \
    --episodes-dir "$HOME/datasets/objectnav/hm3d/v2/train" \
    --scenes-dir "$HM3D_SCENES_DIR" --detector-url http://127.0.0.1:18092 \
    --output "$HOME/objnav_benchmark/hm3d_v2/dev"
  ```

  A non-`val` split renames the protocol (`hm3d-objectnav-v2-train/1`), is
  filed under its own split by the harness, and can never be reported as a
  complete run: only `val` has a published count to be complete against. Note
  that the train scene meshes are a separate, much larger download
  (`hm3d_train_v0.2`, 27 GB) than the validation ones.
- Choose parameter ranges from physical dimensions and algorithm contracts
  first (body height and radius, sensor clip range, the 0.25 m / 30° action
  quantisation), then measure on development data. The defaults `run.py`
  supplies are of that kind and none of them came from a validation score:
  `body_height_m` 0.88 and `body_radius_m` 0.18 are the benchmark's own agent,
  `preferred_clearance_m` 0.30 is the A\* clearance preference above that
  radius, and `stop_distance_m` 1.0 is where the dataset generator placed the
  goal view points — the method aims for roughly a metre from the object
  because that is where standing counts. Override them with `--policy-config`
  and the override is recorded in the run manifest.
- Keep an experiment log and report every variant. Do not tune against
  validation and publish only the best number.
- A configuration lock is a **reproducibility commitment, not proof that
  validation was unseen**. It cannot know what was tuned against what
  beforehand. That is what the mandatory `--note` is for.

## Freeze then evaluate

With the Habitat environment active, explicit data paths, and **no `--scenes`,
`--limit` or sharding**:

```bash
export HM3D_EPISODES_DIR="$HOME/datasets/objectnav/hm3d/v2/val"
export HM3D_SCENES_DIR="$HOME/datasets/scene_datasets"

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.frozen_eval \
  freeze --lock "$HOME/objnav_benchmark/hm3d_v2-config-lock.json" \
  --note "Tuned on HM3D train scenes only; HM3D validation never inspected." -- \
  --version v2 --detector-url http://127.0.0.1:18092 --validate-starts

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.frozen_eval \
  run --lock "$HOME/objnav_benchmark/hm3d_v2-config-lock.json" -- \
  --version v2 --detector-url http://127.0.0.1:18092 --validate-starts \
  --output "$HOME/objnav_benchmark/hm3d_v2-frozen-validation"
```

A lock is never overwritten, the run checks the actual prepared execution
configuration (not merely an earlier preflight) before any output is created or
the simulator is reset, and drift in source, policy, converter, data, runtime,
protocol or the ordered selection is refused. The resulting
`evaluation_role.json` records `held_out_claim: false`.

## What a resulting number would and would not mean

It would be our method's score on the complete published HM3D validation split,
under habitat-lab's own success rule, with ground-truth pose and depth and
predicted semantics, on a navmesh whose provenance is recorded and checked
against the publisher's own geodesic distances.

It would **not** be proof of beating any of these papers. Bit-exact
reproduction of their pipelines is not established; their protocols differ from
each other as much as from ours; and a single run on one split, without paired
episodes, does not separate methods that sit a few points apart. The harness's
paired McNemar and permutation tests exist for comparing **our own** variants on
the same episodes, which is the only comparison here that is genuinely paired.
