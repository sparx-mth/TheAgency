# RoboTHOR protocol, tuning and interpretation

## What the cited papers establish

- **SG-Nav**, arXiv:2410.08189v1, §4.1 p.7: RoboTHOR is "1800 validation
  episodes on 15 validation environments with 12 goal object categories".
  Table 1 p.7 reports **SR and SPL only**; Table 2 p.8 repeats RoboTHOR with no
  SoftSPL column even though the other two benchmarks get one. The paper never
  lists the RoboTHOR scenes, never says whether all 1,800 episodes were run,
  never gives a numeric success threshold (its Problem Definition leaves it as
  a symbol `d_s`), and never gives a field of view. Its single "Implementation
  Details" paragraph covers all three benchmarks undifferentiated. Its released
  code covers **MP3D only** — the RoboTHOR figure is not reproducible from it.
- **ApexNav**, arXiv:2504.14478v3: HM3Dv1, HM3Dv2 and MP3D. **No RoboTHOR.**
  Do not use its numbers here.
- **OSG Navigator**, arXiv:2508.04678v1: Gibson and a 400-episode HM3D v0.1
  subset. **No RoboTHOR.** Its Table 2 is nonetheless the source of the **TF**
  and **NM** columns — "TF / NM denote training-free/non-metric approaches" —
  which are method labels, not metrics, and cannot be computed from telemetry.

So of the three papers supplied, **only SG-Nav reports RoboTHOR**, and only
SR and SPL. Everything else in `reported.py` comes from the papers that
actually printed it, not from a downstream comparison table.

## The comparability error the literature propagates

ESC's Table 1 and SG-Nav's Table 1 both place **ProcTHOR 65.2/28.8** and
**ProcTHOR-ZS 55.0/23.7** in a column headed "RoboTHOR", directly beside their
own 1,800-episode validation numbers. Those two figures are **test-split**
numbers, measured on 2,040 episodes.

The proof is internal to ProcTHOR's own Table 2: its EmbCLIP row reads
47.0/0.200, which matches the archived leaderboard's `test_success` 0.4701 and
`test_spl` 0.2004 exactly, while EmbCLIP's *validation* figures are 52.2/26.0.
The Embodied AI workshop retrospective (arXiv:2210.06849 §3.1.3) independently
calls 0.2884 a "test SPL".

This adapter keeps the two groups in separate constants, and the shared
`comparison_table` refuses a reported row whose split differs from the run's.
That refusal is the mechanism, not a convention.

## Where our protocol differs from the official runner

Stated plainly, because each one would otherwise inflate a number:

1. **Ground-truth pose.** The official runner wipes metadata before handing the
   agent an observation: `{object_goal, rgb, depth}` and nothing else. ESC's
   appendix confirms it ran RoboTHOR without GPS. We use the simulator's pose,
   which is the premise of this whole evaluation layer and matches what the
   Habitat-benchmark comparisons get — but on RoboTHOR it is **more information
   than the published baselines had**. Recorded as `pose_source` in every
   manifest. This is the deviation most likely to matter.
2. **Depth is opt-in upstream.** `renderDepthImage` is injected only if the
   agent asks for it; the reference agent does not. We do, so RGB-D is within
   the protocol, but it is not what an RGB-only baseline had.
3. **Any-instance visibility.** Upstream inspects only the first listed
   instance of the goal type. We accept any visible instance, and record both
   verdicts per episode so the report states the size of the difference rather
   than assuming it away.
4. **Build and client.** If `--platform cloud-rendering` is used, the run is on
   a *different Unity build* from every published number — different horizon
   clamp, different depth encoding. The preflight refuses it unless
   `--allow-build-mismatch` is passed, and `reference_build_match` is recorded.

None of these are hidden by a flag name; all four are in `run.json`.

## Our own contamination must be disclosed

Nothing on this branch has been tuned against RoboTHOR validation yet, and that
state should be preserved deliberately:

- RoboTHOR ships **60 training scenes and 108,000 training episodes**
  (`FloorPlan_Train{1..12}_{1..5}`). Develop and tune there, or on synthetic
  scenarios. The loader takes `split=` for exactly this reason.
- Choose parameter ranges from physical dimensions and algorithm contracts
  first — the LoCoBot's 0.175 m radius, the 1.0 m visibility distance, the
  0.25 m / 30° quantisation — then measure on training data.
- Keep an experiment log and report all variants. Do not iterate against
  full-validation feedback and publish only the best number.
- The 12 categories are small objects; the two permissive accept labels
  (`"clock"` → AlarmClock, `"sports ball"` → BasketBall) are a deliberate
  recall-versus-false-STOP trade made *before* seeing any validation score, and
  it should stay that way.

## Freeze then evaluate

With the AI2-THOR environment active and explicit data paths, using the shared
lock helpers (`provenance.freeze_configuration` / `check_frozen_configuration`)
and no scene, limit or sharding:

```bash
python - <<'PY'
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import freeze_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import evaluation_configuration
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor import run as cli
env, policy, config, issues = cli.prepare(cli.parser().parse_args([
    "--detector-url", "http://127.0.0.1:18094"]))
assert not issues, issues
prepared = evaluation_configuration(config, config["selected_episode_ids"],
                                    cli.settings(), policy=policy)
freeze_configuration(prepared, "~/objnav_benchmark/robothor-config-lock.json",
                     development_note="No RoboTHOR validation episode has been "
                                      "inspected or tuned against on this branch.")
env.close()
PY

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run \
  --frozen-lock "$HOME/objnav_benchmark/robothor-config-lock.json" \
  --detector-url http://127.0.0.1:18094 \
  --output "$HOME/objnav_benchmark/robothor-frozen-validation"
```

Lock files are never overwritten, and configuration or data drift is refused
before the output directory is created or the simulator is reset. A lock is a
reproducibility commitment, **not proof that validation was unseen**, and it
creates no held-out or SOTA claim. To back a claim of the complete published
split, also pass the publisher's exact ordered episode ids as
`expected_episode_ids`.

## What a result here would and would not show

A full 1,800-episode run would give SR and SPL directly comparable in
*definition* to SG-Nav's Table 1, subject to the four deviations above — most
importantly ground-truth pose. It would not establish equivalence with any
paper's implementation, and given that ESC's own number does not reproduce and
three rows in the standard table are third-party re-runs, a small margin over a
published figure would not be evidence of anything.
