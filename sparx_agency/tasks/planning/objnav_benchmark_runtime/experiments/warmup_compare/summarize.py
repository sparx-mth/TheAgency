"""Summarise a warm-up comparison: per-config success, SPL, steps, and paired flips against the baseline."""
import argparse
import json
import math
from pathlib import Path


def load(config_dir):
    rows = {}
    for path in sorted(Path(config_dir).glob("*/episodes.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["episode_id"]] = row
    return rows


def wilson(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sign_test_p(wins, losses):
    """Two-sided exact binomial p-value of ``wins`` against ``losses`` under a fair coin."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def describe(rows):
    n = len(rows)
    wins = sum(1 for r in rows.values() if r["success"])
    low, high = wilson(wins, n)
    spl = sum(r["spl"] or 0.0 for r in rows.values()) / max(1, n)
    steps = sum(r["steps"] for r in rows.values()) / max(1, n)
    locked = [r["agent_info"]["policy"]["target_closing"].get("started_step", -1) for r in rows.values()]
    return dict(n=n, sr=wins / max(1, n), low=low, high=high, spl=spl, steps=steps,
                locked_share=sum(1 for s in locked if s >= 0) / max(1, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="the compare output root: <root>/<config>/<scene>/episodes.jsonl")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()
    configs = {d.name: load(d) for d in sorted(args.root.iterdir()) if d.is_dir()}
    base = configs.get(args.baseline)
    print("%-18s %4s %6s %-13s %6s %6s %7s | %s" % ("config", "n", "SR", "95% CI", "SPL", "steps", "locked", "paired vs " + args.baseline))
    for name, rows in configs.items():
        d = describe(rows)
        paired = ""
        if base is not None and name != args.baseline:
            shared = [e for e in rows if e in base]
            gained = sum(1 for e in shared if rows[e]["success"] and not base[e]["success"])
            lost = sum(1 for e in shared if base[e]["success"] and not rows[e]["success"])
            paired = "%d shared: +%d -%d (p=%.2f)" % (len(shared), gained, lost, sign_test_p(gained, lost))
        print("%-18s %4d %6.3f [%.2f, %.2f] %6.3f %6.1f %6.0f%% | %s" % (
            name, d["n"], d["sr"], d["low"], d["high"], d["spl"], d["steps"], 100 * d["locked_share"], paired))


if __name__ == "__main__":
    main()
