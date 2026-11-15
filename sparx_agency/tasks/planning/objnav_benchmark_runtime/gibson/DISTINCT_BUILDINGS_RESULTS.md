# 15-building recorded ObjectNav comparison — 2026-09-15

## Completed

**15 distinct Gibson training buildings, 30 complete runs, 30 verified videos,
zero agent errors.** One generated ObjectNav start per building was fixed before
execution; both explorers used that same start, target and seed. Navigation was
not tuned during the batch. This is a training-development comparison, **not**
the five-scene published validation split or a held-out/SOTA claim.

Artifacts: `~/objnav_benchmark/gibson/distinct-15-20260915/comparison/`.
Open **`index.html`** for the paired video gallery. All recordings are H.264,
1600×900, six decisions/second, with RGB detections, metric depth, observed map,
path/trail, phase and reasoning. Every video was fully decoded and has exactly
`actions + 1` frames, including the terminal observation.

- `comparison.json`, `COMPARISON.md`: detailed paired results and diagnostics.
- `statistics.json`: existing harness confidence intervals and paired tests.
- `coverage.csv`, `coverage.svg`: all 15 observed-area-versus-action curves.
- `recordings.json`, `video-validation.json`: video paths and validation results.
- `campaign.json`, `locks/`: pre-execution source/model/data/configuration locks.
- Per run: original scores, configuration, console log, trajectory, per-step
  decisions and video. No episodes were replaced or discarded.

## Main result

| Metric | Frontier | Bounded FALCON |
|---|---:|---:|
| Success | **12/15 (80.0%)** | 7/15 (46.7%) |
| Success 95% Wilson interval | 54.8–93.0% | 24.8–69.9% |
| Mean SPL, failures included | 0.3181 | **0.3707** |
| SPL 95% bootstrap interval | 0.1645–0.4818 | 0.1512–0.6087 |
| Mean final DTG to success region | **0.322 m** | 1.622 m |
| Mean actions | 182.0 | 89.8 |
| Mean actual planar path | 17.374 m | 9.484 m |
| Native collision flags | 409 | **34** |
| Total wall time | 52.17 min | 16.42 min |

**Frontier is substantially more reliable on this sample. Keep it as the
default.** FALCON's higher mean SPL reflects more efficient paths in some
successful cases, but does not compensate for missing five targets found by
frontier. Its fewer actions and shorter wall time also include early failures;
they must not be described as an unconditional efficiency win.

Paired outcomes: seven both succeed, five frontier-only successes, zero
FALCON-only successes, three both fail. Exact McNemar p=0.0625. FALCON's paired
mean SPL difference is +0.0526, with sign-flip permutation p≈0.5056; four SPL
wins, six losses and five ties. With only one start in each of 15 buildings,
uncertainty is wide; these tests do not establish statistical superiority.

## Per-building results

Actions include STOP. DTG is the existing SemExp FMM distance to the 1 m success
region. The unchanged protocol can score success at the 500-action limit without
STOP; this occurred for frontier in Leonardo and Pinesdale.

| Building | Target | Frontier success / SPL / actions | FALCON success / SPL / actions | FALCON final DTG m |
|---|---|---|---|---:|
| Shelbyville | bed | Yes / 1.0000 / 21 | Yes / 1.0000 / 21 | 0.000 |
| Ranchester | potted plant | No / 0.0000 / 142 | No / 0.0000 / 45 | 0.175 |
| Newfields | toilet | Yes / 0.1477 / 417 | No / 0.0000 / 490 | 3.388 |
| Marstons | chair | Yes / 0.3013 / 162 | Yes / 0.6339 / 100 | 0.000 |
| Woodbine | couch | No / 0.0000 / 69 | No / 0.0000 / 5 | 1.825 |
| Mifflinburg | toilet | Yes / 0.3119 / 173 | No / 0.0000 / 21 | 4.425 |
| Beechwood | chair | Yes / 1.0000 / 18 | Yes / 1.0000 / 26 | 0.000 |
| Allensville | toilet | Yes / 0.2052 / 163 | No / 0.0000 / 22 | 3.674 |
| Tolstoy | potted plant | Yes / 0.0751 / 306 | Yes / 0.0638 / 287 | 0.000 |
| Leonardo | chair | Yes / 0.2973 / 500 | Yes / 1.0000 / 96 | 0.000 |
| Coffeen | toilet | Yes / 0.3810 / 38 | Yes / 1.0000 / 18 | 0.000 |
| Merom | toilet | Yes / 0.1922 / 123 | No / 0.0000 / 67 | 3.553 |
| Hanson | chair | Yes / 0.7613 / 24 | Yes / 0.8628 / 31 | 0.000 |
| Pinesdale | potted plant | Yes / 0.0983 / 500 | No / 0.0000 / 30 | 4.044 |
| Wainscott | potted plant | No / 0.0000 / 74 | No / 0.0000 / 88 | 3.243 |

## What the extra buildings reveal

- **Premature FALCON termination remains important.** Allensville and Mifflinburg
  end after local blockage and no eligible region; Pinesdale after negligible
  gain/no eligible region; Merom after bounded transit blockage. Newfields uses
  490 actions across multiple bursts but also ends without a useful eligible
  region. These are navigation failures, not omitted cases or infrastructure
  crashes.
- **Target confirmation is not infallible.** Ranchester, Woodbine and Wainscott
  stop with the policy's “fresh multi-view-confirmed target” reason but fail the
  evaluator. The logs alone cannot distinguish detector false positives from
  approach/range or scorer-alignment issues. Recordings are available for review.
- **Early success-region entry is not verified success.** Some agents enter the
  evaluator's region early and continue moving. `actions_to_success` in the raw
  report is first evaluator-only region entry for finally successful episodes;
  terminal actions are reported separately. No GT signal is fed back to policy.
- FALCON produced executable local CP/SOP plans in **14/15** cases. Hanson's
  candidate-handling run is retained and marked unexercised, not removed; its
  SPL difference is not evidence of a local FALCON coverage-plan benefit.
- Native collisions include partial sliding contacts; fewer collisions partly
  reflect less movement. Total observed-area proxy gain was 668.99 m² versus
  429.88 m², and gain per issued action 0.2451 versus 0.3191 (frontier/FALCON).
  Unequal termination horizons confound the ratio; inspect the saved curves.
  Revisit counts were 169/107, immediate turn reversals 71/76, and no-progress
  actions 982/324. These are diagnostic counts, not independent success metrics.

FALCON burst reasons across the batch: negligible gain 7, verified target 6,
unrecoverable blockage 5, action budget 5, and no useful reachable frontiers 5.
Every local burst was at most 48 charged actions; each global count matched the
actual emitted actions and stayed within 500. Verification and recovery were
not free. There was no switch to the old explorer inside a FALCON run.

## Reproducibility and interpretation

Source SHA-256:
`6ddb9daffac41b44bf42087b0fb8a363381979dff282689ddc213931c86f3617`.
The navigation core and policy modules retained their pre-campaign bytes; changes
were limited to safe training asset/data adapters, campaign/report tooling and
recording labels. The completed CPU YOLO-World X-v2 detector and CPU Qwen2.5
3B LLM configurations were identical across methods. Full details are in each
saved lock. The original 500-action, RGB-D/pose, 0.25 m/30-degree protocol and
Habitat 0.2.4 versus reference 0.1.5 caveat were retained.

The training sampler follows the official SemExp criteria with deterministic
seeds and bounded validity checks. It is not a released validation episode set.
Target mix: bed 1, chair 4, couch 1, potted plant 4, toilet 5; no TV episode.
No claim of complete category or start-pose coverage is supported. Source/model
locks are reproducibility guards, not proof of unseen pretraining data.

Tests before execution: 21 targeted generation/import/recording tests, and
**1,718 passed** in the broader runtime/ObjectNav/harness/detector suite. No
navigation parameter was tuned after results were observed. Existing services
and other simulator worktrees were preserved; no commit or push was made.
See [DISTINCT_BUILDINGS.md](DISTINCT_BUILDINGS.md) for generation, run and resume
commands. Further algorithm changes should be a new, explicitly separate
experiment rather than overwriting this frozen campaign.


