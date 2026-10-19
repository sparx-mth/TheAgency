# Bounded FALCON / Gibson development results — 2026-09-15

## Verdict

**Implemented and exercised end to end, but not a general ObjectNav improvement.**
The final adaptation succeeds faster and with higher SPL on Corozal, but fails
on Collierville where the preserved frontier baseline succeeds. Keep **frontier
as the default**. Higher aggregate observed-area-per-action is not sufficient:
at the common 48-action horizon FALCON covers less proxy area in both scenes.

These are two **previously inspected development starts**, not a held-out split,
a completed 1,000-episode validation, or a SOTA claim. Every paired attempt is
retained; the report does not combine best episodes from different versions.

## Reproduction and provenance

Final artifacts: `~/objnav_benchmark/gibson/falcon-dev-20260915/paired-validated/`.
The [runbook](BOUNDED_FALCON.md#reproducible-development-run) gives the exact
paired command. Use a new output directory. Original `run.json`, `episodes.jsonl`,
`summary.json`, audit and console logs remain under each explorer/scene folder.

- Actual final Python source SHA-256:
  `5c5b4638202381a002ccd88c544b1ca539bb6fdf067da2a33423a6acc7177f09`.
- Base Gibson `cc57809f`; completed detector delta `5c465f97`, no commit/push.
- Same **YOLO-World X-v2**, CPU, four Torch threads, image size 640, IoU 0.5,
  emission threshold 0.05, downstream object threshold 0.35, exact Gibson
  24-prompt vocabulary. Checkpoint and actual prompt-feature hashes are in every
  run configuration. LLMDet remains selectable but was not substituted here.
- Same CPU Ollama `qwen2.5:3b-instruct`, temperature 0, seed 0, max 768 tokens,
  30 s request timeout, existing bounded retry/schema repair. No paid service,
  model download or model environment installation was used.
- Intel i9-14900HX; RTX 5070 Laptop GPU used only for Habitat rendering. The
  other detector services and simulator worktrees were left unchanged.
- Habitat-sim 0.2.4, numpy 1.26.4, scipy 1.13.1, scikit-image 0.24.0,
  scikit-fmm 2024.5.29, networkx 3.2.1; remaining versions in `run.json`.
- **Unchanged** Gibson/SemExp v1.1 scoring, 500-action episode cap, 0.25 m
  forward/30-degree turns, RGB-D/pose sensor profile and published starts.
  The reference Habitat 0.1.5 mismatch remains explicitly acknowledged.

The entire hierarchy is compared, not an isolated frontier-selection ablation:
reasoning cadence and ground-safety orchestration also differ. Observations have
the same sensor protocol and starts; trajectories naturally produce different
subsequent images. Source/model/configuration equality was checked across the
pair before accepting the report.

## ObjectNav outcomes

“First region entry” is evaluator-only first arrival in the success region, not
multi-view verification. “Actions” includes the terminal STOP. Failure has no
actions-to-success value. No intermediate success signal reaches the policy.

| Explorer | Scene / episode | Success | SPL | Final DTG m | Actions | First region entry |
|---|---|---:|---:|---:|---:|---:|
| Frontier | Collierville/000000 | Yes | 0.1049 | 0.0000 | 198 | 193 |
| FALCON | Collierville/000000 | No | 0.0000 | 5.6101 | 91 | — |
| Frontier | Corozal/000000 | Yes | 0.5721 | 0.0000 | 64 | 61 |
| FALCON | Corozal/000000 | Yes | 0.9204 | 0.0000 | 49 | 43 |

| Aggregate over these two starts | Frontier | FALCON |
|---|---:|---:|
| Successes | 2/2 | 1/2 |
| Mean SPL | 0.3385 | 0.4602 |
| Mean final DTG m | 0.0000 | 2.8050 |
| Total actions | 262 | 140 |
| Total new observed area / total actions, m²/action | 0.2600 | 0.3289 |

The higher mean SPL does **not** erase the success-rate regression. Nor is
shorter runtime on an early failed episode a navigation-efficiency win.

## Coverage curves and motion stability

Coverage means **ever-observed ternary occupancy area**, including observed
occupied cells, not ground-truth accessible floor coverage. Initial area is
excluded from gain. Samples are at decision observations; the terminal image is
not separately fused. Floor revisions have separate masks and can double-count
a physical floor revisited after a reset. All tested final trajectories stayed
in the local planar setting; this is not a multi-floor coverage claim.

`coverage.csv` is the reproducible per-action curve data; `coverage.svg` is the
corresponding two-panel plot. Each curve ends at that policy's terminal decision;
no invented post-STOP sensing is added.

To redraw from any paired output (test environment has matplotlib; no renderer):

```bash
export PAIR="$HOME/objnav_benchmark/gibson/falcon-dev-20260915/paired-validated"
.venv/bin/python - <<'PY'
import csv, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = Path(os.environ["PAIR"])
rows = list(csv.DictReader((root / "coverage.csv").open()))
episodes = sorted({r["episode"] for r in rows})
fig, axes = plt.subplots(1, len(episodes), squeeze=False, figsize=(11, 4))
for ax, episode in zip(axes[0], episodes):
    for backend in ("frontier", "falcon"):
        points = [r for r in rows if r["episode"] == episode and r["explorer"] == backend]
        initial = float(points[0]["observed_m2"])
        ax.plot([int(p["action"]) for p in points],
                [float(p["observed_m2"]) - initial for p in points], label=backend)
    ax.set(title=episode, xlabel="Executed actions", ylabel="New proxy area (square metres)")
    ax.legend()
fig.tight_layout()
fig.savefig(root / "coverage.svg")
PY
```

| Explorer / scene | New area m² | m²/action | 0.5 m cell revisits | Same-direction consecutive turns | Immediate turn reversals | Native collisions | No-progress actions | No-progress decision time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frontier / Collierville | 30.36 | 0.1533 | 28 | 47 | 6 | 3 | 44 | 50.15 |
| FALCON / Collierville | 23.74 | 0.2609 | 4 | 39 | 5 | 0 | 19 | 10.98 |
| Frontier / Corozal | 37.75 | 0.5898 | 5 | 20 | 2 | 0 | 6 | 3.39 |
| FALCON / Corozal | 22.30 | 0.4551 | 0 | 14 | 7 | 0 | 8 | 4.70 |

A revisit is re-entry into a previously occupied 0.5 m pose bin after leaving it;
it is not a ground-truth room revisit. Same-direction turns can be necessary.
A reversal is an immediately opposite turn, not automatically a livelock.
No-progress means <0.05 m translation and <0.01 m² new proxy area. Associated
wall time sums the preceding policy decisions, excluding renderer and external
wait time. Failed-forward notifications were zero in all four runs; native
contacts capture the baseline's partial sliding collisions that proxy misses.

Common-action-horizon **new proxy area**, avoiding unequal stopping horizons:

| Scene | Actions | Frontier m² | FALCON m² |
|---|---:|---:|---:|
| Collierville | 10 | 9.93 | 12.15 |
| Collierville | 25 | 22.89 | 19.82 |
| Collierville | 48 | 26.34 | 22.31 |
| Corozal | 10 | 9.46 | 2.06 |
| Corozal | 25 | 11.68 | 5.21 |
| Corozal | 48 | 26.52 | 22.30 |

Thus **no consistent coverage improvement is demonstrated**, despite a better
aggregate per-action ratio. Early termination and different amounts of transit
confound that ratio.

## Planning latency and pipeline allocation

All values below are measured wall-clock milliseconds on this host. FALCON
planning includes all attempted local CP/SOP/refinement transactions; A* is a
separate component. Those components do different jobs and are not a speedup
comparison. Total policy-decision latency includes mapping, detection, reasoning
and planning; renderer time is excluded.

| Component / explorer / scene | Count | Median ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| A* / Frontier / Collierville | 31 | 4.7 | 41.0 | 60.4 | 63.0 |
| A* / Frontier / Corozal | 13 | 3.0 | 61.4 | 61.5 | 61.5 |
| FALCON planner / Collierville | 12 | 200.8 | 425.4 | 521.0 | 544.9 |
| FALCON planner / Corozal | 4 | 81.6 | 106.2 | 106.9 | 107.1 |
| Total decision / Frontier / Collierville | 198 | 552.5 | 3806.9 | 8777.8 | 9450.3 |
| Total decision / Frontier / Corozal | 64 | 543.0 | 3199.3 | 4524.5 | 5501.3 |
| Total decision / FALCON / Collierville | 91 | 533.1 | 834.8 | 4697.2 | 4874.2 |
| Total decision / FALCON / Corozal | 49 | 544.2 | 636.9 | 3135.9 | 5430.1 |

FALCON component time, in seconds:

| Scene | Mapping | Detector HTTP | Local planning | Room LLM | Room selection incl. cost build | Transit/target A* | Episode wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Collierville | 0.430 | 42.093 | 2.812 | 8.395 | 0.109 | 0.004 | 62.639 |
| Corozal | 0.241 | 23.056 | 0.281 | 4.783 | 0.097 | 0.029 | 34.216 |

Components do not exhaust wall time: graph geometry, evidence processing,
controller and simulator overhead remain. The baseline retains its old integrated
reasoning cadence, so its room/frontier compute is included in total decision
time rather than separately instrumented. Full component records are in
`comparison.json` and `agent_info.policy.exploration_metrics`.

| FALCON scene | Local exploration | Recovery | Target interrupt | Transit | STOP | Total |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 68 | 6 | 0 | 16 | 1 | 91 |
| Corozal | 12 | 12 | 23 | 1 | 1 | 49 |

Baseline raw allocations: Collierville 128 frontier, 8 search/hold, 8 target and
54 transit decisions; Corozal 24 frontier, 7 search/hold, 27 target, 5 transit,
1 select. Its terminal action retains the legacy phase attribution.

FALCON bursts: Collierville **48** charged actions (`action_budget`), then **26**
(`negligible_gain`); no eligible reachable region remained and the method stopped.
Corozal **45** burst actions (`unrecoverable_blockage`), followed by room selection
and another target interrupt; a fresh multi-view-confirmed target produced STOP.
Burst accounting includes active-burst verification/recovery and is not merely
the “local exploration” column. All global counts equal actual environment
steps; no burst exceeded 48. Three unsafe forward proposals were vetoed in
Corozal; none were silently sent as longer/different movement actions.

## Proof of real integration and tests

- Collierville: **12** successful FALCON plans; Corozal: **3** successful plans
  plus one explicitly logged startup `no_useful_reachable_frontiers` attempt.
  Plans record unknown zones, CP order, SOP precedence, refinement and free routes;
  some jointly refine multiple viewpoints. No old-explorer fallback occurred.
- Actual LLM **classification/probability** queries: FALCON 2/2 in Collierville,
  1/1 in Corozal; baseline 17/21 and 0/8. Evidence gates were not lowered.
- Real `rpt_star` solver records exist in both FALCON episodes, followed by A*
  transit. Corozal demonstrates target interruption/verification; Collierville
  demonstrates repeated bounded exploration and room selection.
- One unfinished route was stored across bursts. Actual restored-route execution
  is **unit-tested**, not demonstrated by this pair (zero restoration events).
- **2,242 tests passed**, exit 0, in the documented lightweight environment with
  plugin autoload disabled. Includes budget exhaustion/accounting, room entry and
  renumbering/splits, live frontiers/visibility/unreachability, graph/DP/SOP,
  discrete safety, route retention/resumption, target rejection, floor reset,
  backend provenance, deferred classification and evaluator-only telemetry.
- No new dependencies were installed. Existing scientific-library deprecation
  and optional plotting warnings remain. Core syntax is Python 3.8 compatible;
  ROS/Torch/Habitat were not loaded by the lightweight import check.
- All **1,000 published validation starts** pass dataset/scorer preflight, with
  the existing `Markleeville/000188` finite-sentinel disclosure retained and no
  episode exclusion. This checks inputs, not held-out navigation performance.

## Previous attempts and limitations

All paths here are siblings under `falcon-dev-20260915/`:

| Attempt | FALCON Collierville | FALCON Corozal | What it exposed |
|---|---|---|---|
| `smoke-01` | Fail, 88 actions, DTG 4.0368 m | Not run | Boundary entry goal could finish outside room |
| `paired-01` | Fail, 277 actions, DTG 0.5965 m | Fail, 3 actions | Scope reassignment; false zero-motion view |
| `paired-final` | Fail, 91 actions, DTG 5.6101 m | Fail, 62 actions | Unsafe transit/verification turn-undo behavior |
| `paired-safety-final` | Fail, 91 actions | Success, 49 actions, SPL 0.9204 | Deferred classifier gate was reset every burst |
| `paired-validated` | Fail, 91 actions | Success, 49 actions, SPL 0.9204 | Final source; classifier now actually exercised |

Both frontier starts succeeded in every paired attempt with the same actions
and SPL. Fixes addressed reproduced integration faults, not a held-out parameter
sweep. No best-run splicing was used.

Remaining limitations: conservative unknown/footprint constraints and local
scope/revisit eligibility can terminate prematurely; exact/beam ordering is not
LKH-equivalent; no global multi-floor memory or GT coverage denominator; no
independent visual verifier; planning deadlines are cooperative, not hard
real-time preemption. The supplementary video/images could not be visually reviewed
with the available tools; see the sourced design for research-access limits.
The full frozen validation split has not been run. Further algorithm tuning
belongs on training/synthetic development data before a separate frozen report.



