"""Add ``collapse_signature`` to runs analysed before the classifier existed.

``analyze.collapse_signature`` reads only ``motion``, ``coverage`` and
``tracking``, all of which are already stored in every ``metrics.json``. So the
tag can be filled in from the stored metrics alone -- no logs, no re-analysis.

That distinction matters: ``analyze()`` refuses to re-run on a pruned run
precisely because its log-derived numbers would come back as zeros. This tool
sidesteps that by never recomputing anything else. It adds one key and leaves
every existing value untouched, so it is safe on pruned runs and idempotent on
repeated invocation.

It cannot affect the pre-registered P41 test, which selects runs by START TIME
after its cutoff; a backfilled historical run is outside that window whether it
carries the key or not.

Usage:
    python -m sparx_agency.tools.falcon_campaign.backfill_signatures [--apply]

Without ``--apply`` it reports what would change and writes nothing.
"""

import json
import sys

from sparx_agency.tools.falcon_campaign import analyze
from sparx_agency.tools.falcon_campaign import config as C


def backfill(runs_dir, apply_changes=False):
    """Fill in the missing tag for every run under ``runs_dir``.

    Args:
        runs_dir: Directory holding the timestamped run folders.
        apply_changes: Write the files. When False, only report.

    Returns:
        tuple: (number filled in, number already present, number unreadable).
    """
    filled = present = broken = 0
    for run in sorted(runs_dir.glob("2026*Z")):
        path = run / "metrics.json"
        if not path.is_file():
            continue
        try:
            metrics = json.loads(path.read_text())
        except (ValueError, OSError):
            broken += 1
            continue
        if "collapse_signature" in metrics:
            present += 1
            continue
        tags = analyze.collapse_signature(metrics)
        filled += 1
        volume = (metrics.get("coverage") or {}).get("final_m3")
        print("%s  %s  %s" % (run.name,
                              ("%7.0f m3" % volume) if volume else "      -  ",
                              "+".join(tags) or "(clean)"))
        if apply_changes:
            metrics["collapse_signature"] = tags
            path.write_text(json.dumps(metrics, indent=2, sort_keys=True,
                                       default=str) + "\n")
    return filled, present, broken


def main():
    """Report or apply the backfill."""
    apply_changes = "--apply" in sys.argv
    filled, present, broken = backfill(C.RUNS_DIR, apply_changes)
    print("\n%s %d run(s); %d already tagged; %d unreadable."
          % ("filled in" if apply_changes else "would fill in",
             filled, present, broken))
    if filled and not apply_changes:
        print("Nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
