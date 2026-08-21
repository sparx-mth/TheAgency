"""Pre-registered test: does the ABSENCE of a collapse signature predict health?

Written 2026-08-21 07:55, before the data existed, for the same reason
``test_p38.py`` was: the hypothesis came from a scan over runs already in hand,
and a scan can only generate a hypothesis.

The circularity is structural here rather than incidental. The five signature
thresholds in ``analyze.collapse_signature`` were chosen by looking at
collapses, so measuring their enrichment on those same collapses cannot confirm
anything. Over the 120 runs that generated it:

    P(collapse | no tag) = 2 %   (1 of 60)
    P(collapse | any tag) = 27 % (16 of 60)
    base rate = 14 %

THE HYPOTHESIS -- one claim, no others, on FRESH runs only:

    P(collapse | no tag) <= 5 %, on runs started after CUTOFF, with n >= 40

Nothing else is scanned on this set. If it passes, the untagged case is a
usable triage signal -- "this run is almost certainly fine" -- which is worth
having even though no tag explains a collapse. If it fails, the signatures are
labels and nothing more, and this file records that they were tested.

Deliberately NOT the hypothesis: that any individual tag predicts collapse.
CIRCLING fires on 32 healthy runs out of 42, and P38 already refuted the
correlation between circling and volume.
"""

import datetime
import glob
import json
import os

#: Runs started after this instant are fresh with respect to the hypothesis.
CUTOFF = datetime.datetime(2026, 8, 21, 8, 0)
#: Refuse to report below this many fresh runs; five leads have died at small n.
MIN_RUNS = 40
#: The pre-registered bar.
MAX_UNTAGGED_COLLAPSE_RATE = 0.05
#: Volume below which a run counts as a collapse.
COLLAPSE_M3 = 1300.0
#: Only the settled configuration is comparable.
MAX_VEL = 0.8
RAYCAST_MAX = 8.0


def _fresh_runs(runs_dir):
    """Settled-config, reliable runs started after ``CUTOFF``."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "2026*Z"))):
        name = os.path.basename(path)
        try:
            started = datetime.datetime.strptime(name[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if started <= CUTOFF:
            continue
        metrics = os.path.join(path, "metrics.json")
        if not os.path.isfile(metrics):
            continue
        with open(metrics) as handle:
            data = json.load(handle)
        config = data.get("config") or {}
        coverage = data.get("coverage") or {}
        if config.get("max_vel") != MAX_VEL:
            continue
        if config.get("raycast_max") != RAYCAST_MAX:
            continue
        if not coverage.get("reliable"):
            continue
        if "collapse_signature" not in data:
            continue          # analysed before the classifier existed
        out.append((name, coverage["final_m3"], data["collapse_signature"]))
    return out


def main(runs_dir):
    """Report the pre-registered result, or refuse if the sample is too small."""
    runs = _fresh_runs(runs_dir)
    untagged = [r for r in runs if not r[2]]
    print("fresh runs since %s: %d (need %d)" % (CUTOFF, len(runs), MIN_RUNS))
    if len(runs) < MIN_RUNS:
        print("NOT YET TESTABLE -- no rate is reported below n=%d, by design."
              % MIN_RUNS)
        return
    if not untagged:
        print("INCONCLUSIVE: no untagged runs in the sample.")
        return
    collapsed = [r for r in untagged if r[1] < COLLAPSE_M3]
    rate = float(len(collapsed)) / len(untagged)
    print("untagged runs: %d, of which collapsed: %d" % (len(untagged),
                                                         len(collapsed)))
    print("P(collapse | no tag) = %.1f %% (bar: <= %.0f %%)"
          % (100.0 * rate, 100.0 * MAX_UNTAGGED_COLLAPSE_RATE))
    print("RESULT: %s" % ("PASSED -- absence of a tag is a usable triage signal"
                          if rate <= MAX_UNTAGGED_COLLAPSE_RATE else
                          "FAILED -- the signatures are labels, nothing more"))
    for name, volume, _ in collapsed:
        print("   collapsed while untagged: %s at %.0f m3" % (name, volume))


if __name__ == "__main__":
    from sparx_agency.tools.falcon_campaign import config as C
    main(str(C.RUNS_DIR))
