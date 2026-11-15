# Recorded comparison in 15 distinct Gibson buildings

**Completed:** all 30 runs and videos, zero agent errors. Frontier succeeded
12/15; FALCON 7/15. See [measured results](DISTINCT_BUILDINGS_RESULTS.md).
The gallery is `comparison/index.html` under the campaign directory below;
`video-validation.json` verifies every recording's decoded frame count.

The published Gibson tiny validation split has five buildings, not fifteen.
For this request, use **15 distinct TRAIN buildings** from the already licensed
local Habitat archive. The train `.glb.json.gz` files are PointNav placeholders;
they are not relabelled as ObjectNav episodes.

`generate_development` creates real, deterministic ObjectNav starts using the
reference SemExp training criteria (`5d76902`, `objectgoal_env.py:156–264`,
`arguments.py:89–94`): choose a floor and available target category, sample a
navigable point on that floor, require a free semantic-map cell and reference
start distance strictly between 1.5 and 100 m, then sample yaw. Floor tolerance
is 0.5 m. Additional safeguards reject unreachable FMM cells, bound sampling
attempts and fail rather than substitute another building. GT data is used only
by the generator/evaluator, never by either navigation policy.

## Frozen campaign

- 15 buildings, one start per building, **both explorers: 30 recorded runs**.
- Buildings selected before any policy run by seeded SHA-256 ordering of the
  25 official training-map names. Start/floor/category/yaw seeds are recorded.
- Same completed CPU YOLO-World X-v2, exact Gibson vocabulary, LLM configuration,
  RGB-D/pose profile, 0.25 m forward/30-degree turns and **500-action episode cap**.
- Navigation code/parameters are not tuned during the batch.
- All 30 source/model/data configurations are preflighted and saved before the
  first simulator reset. Execution checks the saved configuration again.
- Sequential fresh simulator processes, alternating explorer order by building;
  no concurrent GPU renderers or GPU model inference.
- Public split is `train-development`, not `val`; no published-validation,
  held-out or SOTA claim. Fifteen single episodes still give wide uncertainty.

## Commands

Use the same dedicated CPU detector setup from [BOUNDED_FALCON.md](BOUNDED_FALCON.md).
Keep unrelated services running; do not change their vocabularies. The model
checkpoint and encoder must already exist—nothing downloads them automatically.

```bash
export HABITAT_PY="$HOME/miniconda3/envs/habitat/bin/python"
export IMAGEIO_FFMPEG_EXE="$HOME/miniconda3/envs/navdp/bin/ffmpeg"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export CAMPAIGN="$HOME/objnav_benchmark/gibson/distinct-15-20260915"

"$HABITAT_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development \
  --train-info "$HOME/datasets/objectnav/gibson/objectnav/gibson/v1.1/train/train_info.pbz2" \
  --archive "$HOME/Downloads/gibson_habitat_trainval.zip" \
  --scenes-dir "$HOME/datasets/gibson/development-15-20260915/scenes" \
  --output "$CAMPAIGN/episodes.json" --buildings 15 --seed 0

"$HABITAT_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distinct_buildings \
  --manifest "$CAMPAIGN/episodes.json" --output "$CAMPAIGN/comparison" \
  --detector-url http://127.0.0.1:18095 --detector-backend yolo_world \
  --seed 0 --video-fps 6 --allow-sim-version-mismatch
```

The manifest is immutable. To resume the exact same comparison, add `--resume`
to the second command; source/data/configuration changes are refused. Use
`--summarize-only` to rebuild the report/gallery from completed recordings.
The original published-validation entrypoints and checks remain unchanged.

## Artifacts

`comparison/index.html` is the paired recording gallery. For each building and
explorer, `recordings/<episode-hash>/` contains:

- `video.mp4`: browser-playable H.264 dashboard with RGB detections, metric depth,
  observed map, executed trail, planned route, reasoning and actual policy phase;
- `trajectory.csv`, `steps.jsonl`, `metrics.json`, `episode.json`, `final.jpg`.

The video uses six decisions per second for playback; this is not real-time
simulation and does not add sensing. FALCON recordings show the actual hierarchy
phase and burst allowance, not the inactive frontier supervisor's state.

At campaign root: immutable `campaign.json`, `locks/`, progress, raw per-run
records, `comparison.json`, `COMPARISON.md`, `coverage.csv`, `statistics.json`
and `recordings.json`. Statistics reuse the harness's confidence intervals and
paired comparison. A FALCON episode that stops before any local plan is retained
and marked unexercised, not discarded to improve its score. Source/model/data
mismatches remain errors. Observed coverage is a proxy, not GT coverage.

Selected buildings for seed 0: Shelbyville, Ranchester, Newfields, Marstons,
Woodbine, Mifflinburg, Beechwood, Allensville, Tolstoy, Leonardo, Coffeen, Merom,
Hanson, Pinesdale and Wainscott. The fixed target mix is in `episodes.json`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests/test_distinct_buildings.py \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests/test_assets.py \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests/test_demo.py -q
```

Tests cover deterministic distinct selection, bounded valid-start generation,
train/val separation, asset drift, both recorded jobs per building and actual
FALCON dashboard rendering. The broader runtime/ObjectNav/harness/detector suite
also runs before the campaign. Results are not used to change navigation policy.


