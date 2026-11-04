# Multi-story ZSON development results — 2026-09-17

## Outcome

**The end-to-end pipeline executed successfully, but navigation performance is
not satisfactory.** The complete frozen batch finished all 15 episodes with no
agent/runtime errors and produced the five preselected recordings. It achieved
**0/15 annotated-target successes**, mean SPL **0.0000**, mean SoftSPL **0.0596**,
and mean final distance-to-goal **12.1099 m**. These results do not support a
claim that multi-story ZSON search quality is solved or improved.

| Environment | Episodes | Successes | Mean SPL | Mean final DTG (m) | Episodes with confirmed floor connections |
|---|---:|---:|---:|---:|---:|
| Ranchester | 3 | 0 | 0.0000 | 11.3499 | 0 |
| Pomaria | 3 | 0 | 0.0000 | 15.2360 | 0 |
| Hanson | 3 | 0 | 0.0000 | 11.6430 | 0 |
| Coffeen | 3 | 0 | 0.0000 | 9.6564 | 2 |
| Woodbine | 3 | 0 | 0.0000 | 12.6641 | 0 |

Eight episodes issued STOP outside the annotated success region; seven exhausted
the 500-action budget. Seven of those STOPs occurred on an unannotated floor;
Coffeen/000000 reached the reference floor but stopped 4.34 m from its annotated
success region. Thus incomplete annotations are a real evaluation limitation,
not a justification for claiming the observed failures were successes.

## What was verified

- **1,695 tests passed**, covering the runtime, ObjectNav core, FALCON, benchmark
  harness, object mapping, topology and A* utilities.
- Actual RGB-D/pose-only stair diagnostics completed **ascent and descent** in
  Ranchester between approximately 0.04 m and 2.64 m. Intermediate stair landings
  did not become new floor graphs. Diagnostics are not scored campaign episodes.
- Coffeen/000000 created a floor connection during the ordinary search policy.
  Coffeen/000002 traversed a connection in both directions and restored floor 0
  on the return visit (`visits=2`, one edge with `traversals=2`).
- The campaign has **exactly five distinct environments × three episodes** and
  one recording per environment, selected before outcomes were known.
- All five H.264 videos passed full FFmpeg decoding. Resolution is 1600×900 at
  six decision frames per second; frame count equals episode actions plus the
  terminal observation. Verification is saved alongside the results.
- Source identity matched the frozen campaign after all runs. Model settings,
  starts, targets and navigation parameters were not changed during the batch.

## Recordings and artifacts

On the evaluation workstation, open:

`~/objnav_benchmark/multistory-20260917/campaign-v2/index.html`

The gallery contains the first episode of every environment, not a selection of
successful attempts. Per-recording durations are:

| Environment | Recorded episode | Frames | Playback duration |
|---|---|---:|---:|
| Ranchester | Ranchester/000000 | 383 | 63.8 s |
| Pomaria | Pomaria/000000 | 172 | 28.7 s |
| Hanson | Hanson/000000 | 501 | 83.5 s |
| Coffeen | Coffeen/000000 | 399 | 66.5 s |
| Woodbine | Woodbine/000000 | 103 | 17.2 s |

The same directory contains `RESULTS.md`, `statistics.json`, `recordings.json`,
`verification.json`, `campaign.json`, frozen `locks/`, and every building's
metrics, evaluator diagnostics, trajectories and decision traces. Full test
output is `../final-tests-v2.log`. Model services ran on CPU; Habitat released
the rendering GPU at the end.

## Protocol and interpretation

This is **`sparx-gibson-multistory-development/2`**, not published SemExp Gibson
validation. Full-building geometry is available, but semantic goal maps cover
only the reference floor. Starts were generated on another connected storey;
scoring requires height-qualified membership in the annotated goal region.
Distances and SPL use 3D travel, and camera LOOK actions are explicitly allowed.
No private goal, distance field or navmesh query reaches the policy.

Version 1 stopped after nine completed episodes when Coffeen produced a native
0.305 m vertical move above its overly restrictive 0.30 m validator bound. The
interrupted attempt remains in `../campaign/`; it is **not mixed** into these
results. A separate, policy-independent audit of 240,000 native moves across
these five original navmeshes measured a 0.51932 m maximum. Version 2 uses a
finite 0.60 m native vertical bound, retains all other motion and path-length
checks, and leaves the policy's 0.24 m observed tread limit unchanged. All fifteen
starts, targets, goal regions and scene assets were verified identical before
the complete rerun. The audit is retained in `../native-step-audit.json`.

## Remaining work

The next improvements are better stair/portal discovery and selection during
ordinary room search, recovery from stalled traversals, and reducing incorrect
target stops. Full-building semantic annotations are required for a defensible
multi-story ObjectNav success-rate evaluation. The validated stair controller
and persistent graph infrastructure are useful foundations, but neither the
component tests nor this small development batch establishes strong ZSON
performance, statistical superiority, or a SOTA result.

See [MULTISTORY.md](MULTISTORY.md) for architecture, configuration and reproduction.

