"""Inline-SVG chart builders for the comparison report.

Self-contained SVG (no JS, no CDN) so the report file can be mailed or committed
and still render. Colors arrive as CSS custom properties defined by the report, so
light/dark theming lives in one place.

Three forms, each matched to its job:

* :func:`density_chart` -- overlaid distributions. The job is *shift*: showing that
  the whole clearance distribution moved right, not just its mean.
* :func:`delta_histogram` -- per-sample paired improvement. The job is *polarity*:
  a diverging split at zero separates wins from regressions, which a mean hides.
* :func:`rate_bars` -- collision rate by arm. The job is *magnitude* of a single
  headline number per arm.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

W, H = 560, 300
PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 18, 42


def _x(v: float, lo: float, hi: float) -> float:
    """Data value -> pixel X inside the plot area."""
    if hi <= lo:
        return PAD_L
    return PAD_L + (v - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def _esc(s: str) -> str:
    """Minimal XML text escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _axis(lo: float, hi: float, unit: str, ticks: int = 5) -> str:
    """Bottom axis line plus tick labels."""
    y = H - PAD_B
    out = [f'<line x1="{PAD_L}" y1="{y}" x2="{W - PAD_R}" y2="{y}" '
           f'stroke="var(--grid)" stroke-width="1"/>']
    for v in np.linspace(lo, hi, ticks):
        px = _x(v, lo, hi)
        out.append(f'<line x1="{px:.1f}" y1="{y}" x2="{px:.1f}" y2="{y + 4}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{px:.1f}" y="{y + 18}" text-anchor="middle" '
                   f'font-size="11" fill="var(--text-secondary)">{v:.2f}</text>')
    out.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 6}" '
               f'text-anchor="middle" font-size="11" '
               f'fill="var(--text-secondary)">{_esc(unit)}</text>')
    return "".join(out)


def _density(values: np.ndarray, lo: float, hi: float, bins: int = 40) -> np.ndarray:
    """Histogram densities on a fixed range, normalised to a 0..1 peak."""
    hist, _ = np.histogram(values, bins=bins, range=(lo, hi))
    peak = hist.max()
    return hist / peak if peak else hist.astype(float)


def density_chart(series: Dict[str, Sequence[float]], colors: Dict[str, str],
                  lo: float, hi: float, unit: str,
                  marker: Tuple[float, str] | None = None) -> str:
    """Overlaid filled distributions for two or more arms.

    Args:
        series: arm name -> values.
        colors: arm name -> CSS color expression.
        lo, hi: x-range in data units.
        unit: x-axis caption.
        marker: optional ``(value, label)`` reference line, e.g. the safety threshold.

    Returns:
        A complete ``<svg>`` element.
    """
    plot_h = H - PAD_T - PAD_B
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    for name, vals in series.items():
        d = _density(np.asarray(vals, float), lo, hi)
        step = (hi - lo) / len(d)
        pts = [f"{PAD_L},{H - PAD_B}"]
        for i, v in enumerate(d):
            px = _x(lo + (i + 0.5) * step, lo, hi)
            pts.append(f"{px:.1f},{H - PAD_B - v * plot_h:.1f}")
        pts.append(f"{W - PAD_R},{H - PAD_B}")
        poly = " ".join(pts)
        parts.append(f'<polygon points="{poly}" fill="{colors[name]}" '
                     f'fill-opacity="0.22"/>')
        parts.append(f'<polyline points="{poly}" fill="none" '
                     f'stroke="{colors[name]}" stroke-width="2" '
                     f'stroke-linejoin="round"/>')
    if marker is not None:
        mv, mlabel = marker
        px = _x(mv, lo, hi)
        parts.append(f'<line x1="{px:.1f}" y1="{PAD_T}" x2="{px:.1f}" '
                     f'y2="{H - PAD_B}" stroke="var(--text-secondary)" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{px + 5:.1f}" y="{PAD_T + 12}" font-size="11" '
                     f'fill="var(--text-secondary)">{_esc(mlabel)}</text>')
    parts.append(_axis(lo, hi, unit))
    parts.append("</svg>")
    return "".join(parts)


def delta_histogram(deltas: Sequence[float], unit: str, better: str,
                    worse: str) -> str:
    """Diverging histogram of per-sample paired improvement, split at zero.

    Args:
        deltas: oriented per-sample differences (positive = safer).
        unit: x-axis caption.
        better, worse: CSS colors for the improving and regressing sides.

    Returns:
        A complete ``<svg>`` element.
    """
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    span = float(np.abs(d).max()) if d.size else 1.0
    lo, hi = -span, span
    hist, edges = np.histogram(d, bins=41, range=(lo, hi))
    peak = hist.max() or 1
    plot_h = H - PAD_T - PAD_B

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    for i, count in enumerate(hist):
        if not count:
            continue
        x0, x1 = _x(edges[i], lo, hi), _x(edges[i + 1], lo, hi)
        bh = count / peak * plot_h
        mid = 0.5 * (edges[i] + edges[i + 1])
        parts.append(
            f'<rect x="{x0 + 1:.1f}" y="{H - PAD_B - bh:.1f}" '
            f'width="{max(x1 - x0 - 2, 0.5):.1f}" height="{bh:.1f}" rx="2" '
            f'fill="{better if mid > 0 else worse}"/>')
    zx = _x(0.0, lo, hi)
    parts.append(f'<line x1="{zx:.1f}" y1="{PAD_T}" x2="{zx:.1f}" '
                 f'y2="{H - PAD_B}" stroke="var(--text-primary)" stroke-width="1.5"/>')
    parts.append(f'<text x="{zx - 6:.1f}" y="{PAD_T + 12}" text-anchor="end" '
                 f'font-size="11" fill="var(--text-secondary)">worse</text>')
    parts.append(f'<text x="{zx + 6:.1f}" y="{PAD_T + 12}" font-size="11" '
                 f'fill="var(--text-secondary)">safer</text>')
    parts.append(_axis(lo, hi, unit))
    parts.append("</svg>")
    return "".join(parts)


def rate_bars(rates: List[Tuple[str, float, str]], caption: str) -> str:
    """Horizontal bars for one rate per arm.

    Args:
        rates: ``(label, value_0_to_1, css_color)`` triples.
        caption: axis caption.

    Returns:
        A complete ``<svg>`` element.
    """
    row_h, gap = 34, 14
    height = PAD_T + len(rates) * (row_h + gap) + 34
    label_w = 92
    bar_x = label_w + 8
    bar_w = W - bar_x - 58

    parts = [f'<svg viewBox="0 0 {W} {height}" width="100%" role="img">']
    for i, (label, value, color) in enumerate(rates):
        y = PAD_T + i * (row_h + gap)
        parts.append(f'<text x="{label_w}" y="{y + row_h * 0.66:.0f}" '
                     f'text-anchor="end" font-size="13" '
                     f'fill="var(--text-primary)">{_esc(label)}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{row_h}" '
                     f'rx="4" fill="var(--grid)" fill-opacity="0.35"/>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{max(value * bar_w, 2):.1f}" '
                     f'height="{row_h}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{bar_x + bar_w + 8}" y="{y + row_h * 0.66:.0f}" '
                     f'font-size="13" font-weight="600" '
                     f'fill="var(--text-primary)">{value:.0%}</text>')
    parts.append(f'<text x="{W / 2:.0f}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="11" fill="var(--text-secondary)">{_esc(caption)}</text>')
    parts.append("</svg>")
    return "".join(parts)
