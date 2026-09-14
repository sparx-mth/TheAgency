# Gibson/ZSON recovery and results — 2026-09-14

## Recovered state

Branch: `feat/objnav-habitat-gibson-nadav`, based on `24dc0393`.
The interrupted work was recovered from the uncommitted source, saved diagnostic
logs and running evaluation artifacts, not from a complete previous-chat transcript.
The inherited small/large detector comparison was allowed to finish unchanged.
No commits, pushes, dataset substitutions or model downloads were performed here.

The algorithm is still scene-graph/LLM room reasoning, RPT* room ordering,
observed-map frontier exploration, weighted A* and the discrete action converter.
**FALCON itself is not running.** No mandatory initial panorama was added.
Here ZSON means the zero-shot ObjectNav task; this is not an implementation or
reproduction claim for the separately named ZSON paper.

## Completed refinements

The previous session's robot-height 2.5D mapping, physical inflation floor,
route commitment, alias/frame deduplication, multi-view target evidence,
bounded oracle repair and published-split/provenance work were retained.

Additional bugs reproduced with synthetic cases and fixed during recovery:

1. Arrival used the requested goal instead of the planner's actual snapped
   endpoint. Route completion now reuses the executor's decision logic and
   monotone progress, including its discrete-step reach limit. A nearby endpoint
   on an unwalked return leg does not count as arrival.
2. Replanning after obstruction could repeatedly reset the no-progress clock.
   A positional watchdog now survives safety-triggered replans and blocked-step
   notifications; producing another path is not evidence of movement.
3. Verification turns could be charged to another visible landmark. The selected
   landmark now owns its turn budget. Rejected and stalled targets are cooled
   down, and old confirmation support is discarded.
4. Pure A*/ObjectNav imported unrelated native OMPL backends. Planner exports and
   OMPL-specific builders now load those bindings only on demand, preserving
   public APIs. The ObjectNav/A* test process now exits normally rather than
   aborting after its passing summary. This does not repair OMPL for callers
   that actually request it.
5. Frozen validation checked an earlier preflight but not the separately prepared
   execution configuration. The execution now checks that configuration again,
   before output or simulator reset, and rejects drift.

An IDE/disk synchronization problem was encountered while editing. Damaged
intermediate views were archived outside the repository and affected modules
were restored from the verified source. Final validation used actual saved files
and fresh Python processes, not the inconsistent previews.

## Validation performed

- **1,633 tests passed, exit code 0**, covering the runtime, shared ObjectNav
  layer/harness, object mapping, topology, A* and common planner geometry.
- The new route-lifecycle tests initially reproduced eight failures; all ten
  cases now pass. Boundary regressions also cover optional imports, legacy
  exports, explicit missing-backend errors and frozen-execution configuration.
- **All 1,000 published validation starts pass preflight**, with no exclusions.
  `Markleeville/000188` remains the disclosed upstream finite-FMM-sentinel case.
- A fresh five-scene recorded regression completed: all five scene subprocesses
  exited 0, no agent errors, and five original videos were produced.
- A separate Corozal smooth replay was generated successfully. Hash checks
  confirmed its original episode results, run configuration, summary,
  trajectory and original video were unchanged.

The IDE may report an unresolved `pytest` import when configured to a model
interpreter without pytest. The tests above were executed with the documented
`.venv/bin/python` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Development results — not a held-out benchmark

Every row below uses the previously inspected first episode in each of the same
five validation scenes. These are complete batches, not cherry-picked episodes.

| Batch | Detector | Successes | Mean SPL | Mean DTG (m) | Mean SoftSPL |
| --- | --- | ---: | ---: | ---: | ---: |
| Inherited refinement comparison | YOLO-World S | 2/5 | 0.1811 | 2.5693 | 0.1900 |
| Inherited refinement comparison | YOLO-World L | 2/5 | 0.1574 | 4.2834 | 0.1574 |
| After recovery fixes | YOLO-World S | 2/5 | 0.1811 | 2.4993 | 0.1917 |

Fresh run details:

| Scene | Success | SPL | Final DTG (m) | Actions | Termination |
| --- | --- | ---: | ---: | ---: | --- |
| Collierville | No | 0.0000 | 2.8676 | 500 | Step limit |
| Corozal | Yes | 0.7900 | 0.0000 | 41 | STOP |
| Darden | No | 0.0000 | 6.3930 | 18 | STOP |
| Markleeville | Yes | 0.1158 | 0.0000 | 415 | STOP |
| Wiconisco | No | 0.0000 | 3.2359 | 500 | Step limit |

**No success-rate or SPL improvement has been demonstrated by this recovery
batch.** The small DTG/SoftSPL change is descriptive, not statistically meaningful
on five examples. The larger detector was not superior on this comparison and
was not made the default. Persistent false-positive target stops and inefficient
exploration remain; the current method does not establish a SOTA result.

The earlier door/room batch, before these inherited refinements, is also retained
at `door-room-fixes-final-20260914-124814-UTC`: 2/5 success and mean SPL 0.2654.
It is another development batch, not evidence that every refinement improves
navigation scores or a source of episodes to splice into a better result.

## Artifacts on this workstation

All paths below are under `~/objnav_benchmark/gibson/demos/`:

- Inherited comparison: `refinement-final-ab-20260914-163552-UTC/{small,large}/`.
- Fresh regression: `resume-lifecycle-20260914-172707-UTC/`.
- Smoothed video:
  `resume-lifecycle-20260914-172707-UTC/Corozal/recordings/ae3ffb216174/smooth_rgb.mp4`.

Each batch preserves its own configuration/source identities and results.
Recordings, live/HTML dashboards, trajectories and per-step reasoning remain in
those directories. Diagnostic/test/preflight/replay-check logs are additionally
under `/tmp/objnav_resume_20260914/` and may disappear after reboot.

The evaluation and replay processes have finished. Existing CPU detector services
on ports 18092 and 18093 were left available rather than stopped or reconfigured.

## Protocol and next research steps

See [PROTOCOL_AND_TUNING.md](PROTOCOL_AND_TUNING.md) for source citations and the
freeze-then-evaluate commands. OSG Navigator reports Gibson tiny validation;
ApexNav and SG-Nav do not report Gibson. Exact OSG evaluator equivalence and a
complete parameter-search protocol are not established by the inspected sources.
There is no basis to accuse the authors of overfitting or to guarantee its absence.

1. Develop/tune on the publisher's **25 training scenes** or synthetic scenarios.
   Use a genuine training episode generator or explicitly separate development
   rig; do not reinterpret dummy PointNav loader rows as ObjectNav episodes.
2. Address false-positive STOPs with a controlled independent visual verifier
   or alternative detector. Measure target precision/recall and runtime on
   training/development data, not just proposal counts or a larger checkpoint.
3. Measure exploration efficiency and repeated room visits separately from
   perception, while retaining the existing RPT* ordering and safety constraints.
4. Freeze a chosen configuration, then run the exact **1,000-episode validation
   split** once for reporting, disclosing the already inspected examples and
   Habitat 0.2.4 versus reference 0.1.5 mismatch. Full validation has not yet been
   executed here. Do not present these five-episode development results as SOTA.

