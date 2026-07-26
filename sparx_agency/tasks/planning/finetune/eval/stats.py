"""Paired statistics for a baseline-vs-trained comparison.

Averaging a metric over samples, as ``train/evaluate.py`` does, is the wrong
summary for this question. Every sample here is *paired*: both models answer the
same (frame, goal) pair, so the informative quantity is the per-sample difference
and its distribution -- not two independent means.

What this module reports per metric:

* the paired mean/median delta, oriented so **positive always means safer**;
* how many samples improved, regressed, or tied -- a mean gain built from a few
  big wins alongside many small regressions is not a safety improvement;
* the worst single regression, since a route that is usually better but
  occasionally much worse is not deployable;
* a **Wilcoxon signed-rank test**, which is paired, non-parametric, and does not
  assume the deltas are normal (clearance deltas are strongly skewed);
* a rank-biserial **effect size**, because with a few hundred samples a trivial
  difference reaches significance easily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

from .metrics import HIGHER_IS_BETTER


@dataclass(frozen=True)
class PairedResult:
    """Paired comparison of one metric between two arms.

    Deltas are oriented so a positive value is always an improvement over the
    reference arm, regardless of the metric's natural direction.

    Attributes:
        metric: metric name.
        n: number of usable paired samples.
        ref_mean, arm_mean: raw (unoriented) means, for reporting.
        mean_delta, median_delta: oriented paired differences.
        n_better, n_worse, n_tie: sample counts by direction.
        worst_regression: largest oriented loss (<= 0; 0 if nothing regressed).
        p_value: Wilcoxon signed-rank two-sided p, or NaN if undefined.
        effect_size: rank-biserial correlation in [-1, 1].
    """

    metric: str
    n: int
    ref_mean: float
    arm_mean: float
    mean_delta: float
    median_delta: float
    n_better: int
    n_worse: int
    n_tie: int
    worst_regression: float
    p_value: float
    effect_size: float

    @property
    def significant(self) -> bool:
        """True if the paired difference clears p < 0.05."""
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    @property
    def verdict(self) -> str:
        """Human-readable direction, gated on significance."""
        if not self.significant:
            return "no significant difference"
        return "better" if self.mean_delta > 0 else "WORSE"


def _rank_biserial(delta: np.ndarray) -> float:
    """Rank-biserial effect size from paired differences."""
    nz = delta[delta != 0]
    if nz.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total > 0 else 0.0


def compare_metric(metric: str, ref: Sequence[float],
                   arm: Sequence[float]) -> PairedResult:
    """Compare one metric between a reference arm and a candidate arm.

    Args:
        metric: metric name; must appear in :data:`metrics.HIGHER_IS_BETTER`.
        ref: reference values (typically the untrained baseline).
        arm: candidate values, index-aligned with ``ref``.

    Returns:
        The :class:`PairedResult`. Pairs where either side is NaN are dropped.
    """
    r = np.asarray(ref, dtype=float)
    a = np.asarray(arm, dtype=float)
    ok = np.isfinite(r) & np.isfinite(a)
    r, a = r[ok], a[ok]
    if r.size == 0:
        raise ValueError(f"{metric}: no usable paired samples")

    sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
    delta = sign * (a - r)

    p = float("nan")
    if np.any(delta != 0):
        try:
            p = float(wilcoxon(a, r, zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")

    losses = delta[delta < 0]
    return PairedResult(
        metric=metric, n=int(r.size),
        ref_mean=float(r.mean()), arm_mean=float(a.mean()),
        mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
        n_better=int((delta > 0).sum()), n_worse=int((delta < 0).sum()),
        n_tie=int((delta == 0).sum()),
        worst_regression=float(losses.min()) if losses.size else 0.0,
        p_value=p, effect_size=_rank_biserial(delta),
    )


def compare_all(ref_rows: List[dict], arm_rows: List[dict],
                metrics: Sequence[str]) -> Dict[str, PairedResult]:
    """Run :func:`compare_metric` across several metrics.

    Args:
        ref_rows: per-sample metric dicts for the reference arm.
        arm_rows: per-sample metric dicts for the candidate arm, index-aligned.
        metrics: metric names to compare.

    Returns:
        Mapping of metric name to its :class:`PairedResult`.
    """
    return {m: compare_metric(m, [row[m] for row in ref_rows],
                              [row[m] for row in arm_rows]) for m in metrics}


def collision_rate(rows: List[dict]) -> float:
    """Fraction of trajectories that entered an obstacle."""
    flags = [bool(r["collides"]) for r in rows]
    return float(np.mean(flags)) if flags else float("nan")


def format_table(results: Dict[str, PairedResult], ref_name: str,
                 arm_name: str) -> str:
    """Render paired results as a fixed-width console table."""
    head = (f"{'metric':<20}{ref_name:>12}{arm_name:>12}{'Δ(safer+)':>12}"
            f"{'win/loss':>12}{'p':>10}{'effect':>9}  verdict")
    lines = [head, "-" * len(head)]
    for m, r in results.items():
        lines.append(
            f"{m:<20}{r.ref_mean:>12.3f}{r.arm_mean:>12.3f}{r.mean_delta:>+12.3f}"
            f"{f'{r.n_better}/{r.n_worse}':>12}{r.p_value:>10.2e}"
            f"{r.effect_size:>+9.2f}  {r.verdict}")
    return "\n".join(lines)
