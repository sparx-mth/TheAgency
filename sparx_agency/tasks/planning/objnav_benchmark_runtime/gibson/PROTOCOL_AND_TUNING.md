# Gibson protocol, tuning and interpretation

## What the cited papers establish

- **OSG Navigator**, arXiv:2508.04678v1, p.11 §7.1.1: Gibson tiny
  **validation**, 1,000 episodes evenly distributed over five household scenes.
  Table 2, p.12 reports SR, SPL and DTG. Its simulation controller is standardized
  to an FMM controller based on SemExp (p.12). The paper reports system
  hyperparameters in Appendix Table 10, p.32, and hand-designed household
  schemas (p.12). The inspected text does **not establish a separate, detailed
  hyperparameter-search/tuning protocol for Gibson**.
- **ApexNav**, arXiv:2504.14478v3, p.6 §V-A/Table I: HM3Dv1, HM3Dv2 and
  MP3D, **not Gibson**. Do not use those numbers as Gibson comparisons.
- **SG-Nav**, arXiv:2410.08189v1, p.7 §4.1/Table 1: MP3D, HM3D and
  RoboTHOR, **not Gibson**. Its zero-shot/no-navigation-finetuning statement
  does not by itself prove that parameter choices never used validation.

The public SemExp evaluator and constants (revision `5d76902`) identify 25
training scenes and these five validation scenes: Collierville, Corozal,
Darden, Markleeville and Wiconisco. The released v1.1 files contain 200
validation episodes per scene. Gibson's broader tiny partition also lists a
separate test split, but the OSG result cited here is **validation**, not that
hidden/test split. The downloaded Habitat archive is a train/validation archive.

**There is insufficient evidence to accuse these authors of overfitting.**
Disjoint scenes, no ObjectNav-specific training, cross-dataset experiments and
real-robot transfer offer evidence of generalization. They do not rule out
validation-based hyperparameter selection, prompt selection, detector threshold
selection, or contamination in foundation-model pretraining. Where the papers
do not explain parameter selection, the correct answer is “not established by
the inspected sources,” not an invented guarantee.

Exact OSG evaluator revision, episode hashes and every implementation choice
are still not established from the supplied paper. The current public PONI
checkout (`30682c2`) uses a modified multi-goal evaluator and cannot silently
replace SemExp v1.1. This runtime states its exact SemExp-derived protocol and
its Habitat 0.2.4-vs-reference-0.1.5 difference rather than claiming bit-exact
OSG reproduction.

## What this implementation now does

1. Keeps the **same published 1,000 validation episodes and five scenes**.
   No replacements or cherry-picked starts are introduced.
2. Retains the upstream finite FMM sentinel for in-bounds masked cells,
   including Markleeville/000188. SemExp did not reject that published episode;
   the earlier strict preflight check was a mismatch. Bounds/corruption checks
   remain. `GibsonEnv.sentinel_start_ids` exposes these cases for audit.
3. Separates simulator observations/actions/scoring from optional smoother
   video replay. The standard 0.25 m/30° actions and camera are unchanged.
4. Records source, data, detector checkpoint/configuration, runtime, seed and
   motion tolerances. A resumed run cannot quietly change those quantities.
5. Provides an explicit configuration lock before a full validation run.
   It is a reproducibility guard, **not proof of unseen validation**.

## Our own contamination must be disclosed

The first episode in each validation scene, plus a short raw-view probe in
Collierville, has already been inspected during development. These are
**development smoke examples**, not an untouched test set. No later flag or
renaming can undo that. The five-example averages must not be compared as if
they were the paper's complete validation result.

For further systematic tuning, use the **25 training scenes** and synthetic
unit scenarios. Keep the original validation scene list intact; do not create
a custom holdout and present its score as the published benchmark. The current
rig does not pretend the dummy PointNav loader rows in the train archive are
ObjectNav evaluation episodes: train development requires the publisher's
training episode generator or an explicitly separate development rig.

Choose parameter ranges from physical dimensions and algorithm contracts first
(robot height/radius, sensor clip range, discrete action quantization), then
measure on training/development data. Maintain an experiment log and report all
variants; do not repeatedly tune against full-validation feedback and publish
only the best number.

## Freeze then evaluate

Use the Habitat environment and explicit data/service paths. With the ordinary
`gibson.run` arguments after `--`, and **no scene, limit or sharding**:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.frozen_eval \
  freeze --lock "$HOME/objnav_benchmark/gibson-config-lock.json" -- \
  --episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" \
  --detector-url http://127.0.0.1:18092 --allow-sim-version-mismatch

python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.frozen_eval \
  run --lock "$HOME/objnav_benchmark/gibson-config-lock.json" -- \
  --episodes-dir "$GIBSON_EPISODES_DIR" --scenes-dir "$GIBSON_SCENES_DIR" \
  --detector-url http://127.0.0.1:18092 --allow-sim-version-mismatch \
  --output "$HOME/objnav_benchmark/gibson-frozen-validation"
```

Lock files are not overwritten. Configuration or data drift is refused. The
final role metadata explicitly retains `previous_validation_development: true`
and `held_out_claim: false`. The resulting full-validation score, if run, would
be comparable only with the stated implementation/protocol caveats—not proof
that this method beats all current SOTA.

