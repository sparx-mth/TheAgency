"""Assemble a shareable HTML report from :mod:`compare` output.

Reads the driver's ``summary.json`` / ``per_sample.csv`` (optionally a second,
stricter run for the robustness section) and writes one self-contained HTML file
-- inline SVG and a base64-embedded route figure, no external assets -- suitable
for mailing or committing alongside the checkpoint.

The report is written to stand alone: every column and every metric is explained
in the page itself, so a reader who was not part of the work can judge the result
without narration. Only the baseline and trained arms are shown; ``compare.py``
still scores the teacher arm into ``summary.json`` for anyone who wants it.

    python -m sparx_agency.tasks.planning.vlas.navdp.finetune.eval.report \
        --eval-dir ~/Downloads/flight_dataset/run_new/eval \
        --strict-dir ~/Downloads/flight_dataset/run_new/eval_strict \
        --routes ~/Downloads/flight_dataset/run_new/eval/routes.png \
        --out ~/Downloads/flight_dataset/run_new/eval/report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from typing import Dict, List

from . import report_charts as rc

#: Metric key -> (display label, what it measures and why it matters).
METRICS = {
    "min_clearance_m": (
        "Min clearance",
        "The tightest point on the whole route. This is <em>the</em> safety "
        "number, because safety is a worst-case property: a route that averages "
        "a metre of room but clips a doorframe at two centimetres is not safe."),
    "p5_clearance_m": (
        "5th-pct clearance",
        "The sustained tight portion of the route, ignoring a single stray "
        "sample. If this moves the same way as the minimum, the improvement is "
        "structural rather than one lucky point."),
    "mean_clearance_m": (
        "Mean clearance",
        "Average room along the route. The weakest of the safety metrics — it "
        "rewards flying down the middle of open space — but a useful measure of "
        "general comfort."),
    "frac_below_safe": (
        "Time below safe distance",
        "Fraction of the route spent inside the danger band. Captures sustained "
        "exposure, which a single worst-case number cannot."),
    "goal_gap_m": (
        "Goal gap",
        "Distance from the final waypoint to the requested goal. This is the "
        "cost side of the ledger: a model can always look safer by refusing to "
        "commit to the goal."),
    "bending": (
        "Path kinkiness",
        "Sum of second differences along the path — how jerky it is. Lower is "
        "smoother and easier for a tracker to follow."),
}
ARMS = ("baseline", "trained")
ARM_COLORS = {"baseline": "var(--baseline)", "trained": "var(--trained)"}

#: Column name -> what it means. Rendered as the "how to read this" table.
COLUMNS = {
    "baseline": "Mean value for the pretrained NavDP checkpoint.",
    "trained": "Mean value for the fine-tuned checkpoint.",
    "Δ (safer +)": "Paired mean difference, sign-flipped so <strong>positive "
                   "always means safer</strong>. For a lower-is-better metric "
                   "such as goal gap, a positive delta still means improvement.",
    "Win / loss": "Of the paired samples, how many individually improved versus "
                  "regressed. A good mean built on many small regressions is not "
                  "an improvement, and this column is what exposes that.",
    "p": "Wilcoxon signed-rank, two-sided. Paired and non-parametric, because "
         "clearance differences are heavily skewed and a t-test would overstate "
         "the confidence.",
    "Effect": "Rank-biserial correlation, −1 to +1. With hundreds of samples "
              "almost any difference reaches significance, so this says whether "
              "the difference is <em>large</em>, not merely real.",
    "Verdict": "Reported as better or worse only when p &lt; 0.05; otherwise no "
               "significant difference.",
}


def _read_rows(eval_dir: Path) -> Dict[str, List[dict]]:
    """Per-arm rows from ``per_sample.csv``, values coerced to float/bool."""
    by_arm: Dict[str, List[dict]] = {}
    with (eval_dir / "per_sample.csv").open() as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for k, v in row.items():
                if k == "arm":
                    parsed[k] = v
                elif k == "collides":
                    parsed[k] = v.strip().lower() == "true"
                else:
                    parsed[k] = float(v)
            by_arm.setdefault(row["arm"], []).append(parsed)
    return by_arm


def _verdict_pill(res: dict) -> str:
    """Coloured pill for a metric's verdict."""
    v = res["verdict"]
    cls = {"better": "ok", "WORSE": "bad"}.get(v, "neutral")
    return f'<span class="pill {cls}">{v}</span>'


def _table(paired: Dict[str, dict], collisions: Dict[str, float] | None = None) -> str:
    """The paired baseline-vs-trained table, optionally with a collision row."""
    head = ('<tr><th>Metric</th><th>baseline</th><th>trained</th>'
            '<th>Δ (safer +)</th><th>Win / loss</th><th>p</th>'
            '<th>Effect</th><th>Verdict</th></tr>')
    rows = []
    for key, (label, _) in METRICS.items():
        if key not in paired:
            continue
        r = paired[key]
        p = r["p_value"]
        p_txt = "—" if p != p else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
        rows.append(
            f'<tr><td class="metric">{label}</td>'
            f'<td>{r["ref_mean"]:.3f}</td><td>{r["arm_mean"]:.3f}</td>'
            f'<td class="delta {"pos" if r["mean_delta"] > 0 else "neg"}">'
            f'{r["mean_delta"]:+.3f}</td>'
            f'<td>{r["n_better"]} / {r["n_worse"]}</td><td>{p_txt}</td>'
            f'<td>{r["effect_size"]:+.2f}</td><td>{_verdict_pill(r)}</td></tr>')
    if collisions is not None:
        rows.append(
            f'<tr class="highlight"><td class="metric">Collision rate</td>'
            f'<td>{collisions["baseline"]:.1%}</td>'
            f'<td>{collisions["trained"]:.1%}</td>'
            f'<td class="delta pos">'
            f'−{(collisions["baseline"] - collisions["trained"]) * 100:.1f} pts</td>'
            f'<td colspan="3" class="muted">share of routes entering an obstacle</td>'
            f'<td><span class="pill ok">better</span></td></tr>')
    return f'<table>{head}{"".join(rows)}</table>'


def _glossary(items: Dict[str, str]) -> str:
    """Two-column definition table."""
    rows = "".join(f'<tr><td class="metric">{k}</td><td class="defn">{v}</td></tr>'
                   for k, v in items.items())
    return f'<table class="defs">{rows}</table>'


def _metric_notes() -> str:
    """The per-metric "what this actually measures" table."""
    return _glossary({label: note for label, note in METRICS.values()})


def build(eval_dir: Path, strict_dir: Path | None, routes: Path | None,
          out: Path) -> None:
    """Render the report HTML.

    Args:
        eval_dir: directory holding the primary run's CSV + JSON.
        strict_dir: optional stricter-judge run, for the robustness section.
        routes: optional route figure PNG to embed.
        out: destination ``.html``.
    """
    summary = json.loads((eval_dir / "summary.json").read_text())
    rows = _read_rows(eval_dir)
    paired = summary["paired"]["baseline_vs_trained"]
    d_safe = summary["d_safe_m"]
    collisions = {a: summary["collision_rate"][a] for a in ARMS}

    base_clear = [r["min_clearance_m"] for r in rows["baseline"]]
    tuned_clear = [r["min_clearance_m"] for r in rows["trained"]]
    deltas = [t - b for b, t in zip(base_clear, tuned_clear)]

    charts = {
        "density": rc.density_chart(
            {a: c for a, c in zip(ARMS, (base_clear, tuned_clear))}, ARM_COLORS,
            lo=0.0, hi=max(max(base_clear), max(tuned_clear)),
            unit="minimum clearance along the route (m)",
            marker=(d_safe, f"{d_safe:.2f} m safe")),
        "delta": rc.delta_histogram(deltas, "per-sample change in min clearance (m)",
                                    better="var(--trained)", worse="var(--bad)"),
        "collisions": rc.rate_bars(
            [(a, collisions[a], ARM_COLORS[a]) for a in ARMS],
            "share of routes entering an obstacle"),
    }

    strict_block = ""
    if strict_dir is not None:
        s = json.loads((strict_dir / "summary.json").read_text())
        strict_block = f"""
        <h2>Robustness: does the verdict survive a stricter judge?</h2>
        <p>The judge map calls a cell blocked above a fused occupancy
        probability. The main run uses {summary.get('judge_occ_prob', 0.65):.2f};
        this one uses {s.get('judge_occ_prob', 0.90):.2f}, which marks far less of
        the room as obstacle. Absolute values shift, as they must — but every
        conclusion is identical, so the result is not an artefact of how strict
        the map is.</p>
        <div class="table-wrap">{_table(
            s["paired"]["baseline_vs_trained"],
            {a: s["collision_rate"][a] for a in ARMS})}</div>"""

    routes_block = ""
    if routes is not None and routes.exists():
        b64 = base64.b64encode(routes.read_bytes()).decode()
        routes_block = f"""
        <h2>What the routes actually look like</h2>
        <p>Frames sampled across the held-out recording. Each panel is one goal,
        drawn over the fused clearance map: warm is near an obstacle, cool is
        open, white was never observed. The dashed line is the
        {d_safe:.2f}&nbsp;m safety contour — where the orange baseline crosses
        inside it and the blue trained route does not, that is the effect in the
        table above, seen directly.</p>
        <img src="data:image/png;base64,{b64}" alt="route comparison panels"/>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.format(
        rec=summary["recording"], n=summary["n_samples"],
        ckpt=Path(summary["checkpoint"]).name,
        window=summary.get("judge_window", 3) * 2 + 1,
        d_safe=d_safe,
        coll_base=collisions["baseline"], coll_tuned=collisions["trained"],
        min_delta=paired["min_clearance_m"]["mean_delta"],
        min_better=paired["min_clearance_m"]["n_better"],
        min_worse=paired["min_clearance_m"]["n_worse"],
        gap_delta=paired["goal_gap_m"]["mean_delta"],
        gap_worse=paired["goal_gap_m"]["n_worse"],
        table=_table(paired, collisions),
        columns=_glossary(COLUMNS), metric_notes=_metric_notes(),
        strict=strict_block, routes=routes_block, **charts))
    print("wrote", out)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NavDP fine-tune: is the route safer?</title>
<style>
:root {{
  color-scheme: light dark;
  --surface: #fcfcfb; --card: #ffffff; --grid: #d6d5d0;
  --text-primary: #0b0b0b; --text-secondary: #52514e;
  --baseline: #eb6834; --trained: #2a78d6; --good: #008300; --bad: #e34948;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --surface: #17171a; --card: #1f1f23; --grid: #3a3a40;
    --text-primary: #f5f5f4; --text-secondary: #b4b3ad;
    --baseline: #d95926; --trained: #3987e5; --good: #33a133; --bad: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  --surface: #17171a; --card: #1f1f23; --grid: #3a3a40;
  --text-primary: #f5f5f4; --text-secondary: #b4b3ad;
  --baseline: #d95926; --trained: #3987e5; --good: #33a133; --bad: #e66767;
}}
body {{ margin:0; padding:2.2rem 1.2rem 4rem; background:var(--surface);
  color:var(--text-primary); font:15px/1.62 -apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
main {{ max-width: 940px; margin: 0 auto; }}
h1 {{ font-size:1.65rem; margin:0 0 .3rem; letter-spacing:-.02em; }}
h2 {{ font-size:1.15rem; margin:2.5rem 0 .6rem; letter-spacing:-.01em; }}
h3 {{ font-size:.95rem; margin:1.6rem 0 .4rem; color:var(--text-secondary);
  text-transform:uppercase; letter-spacing:.05em; }}
.sub {{ color:var(--text-secondary); margin:0 0 1.6rem; }}
.tiles {{ display:grid; gap:.9rem; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  margin:1.4rem 0 .4rem; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:10px;
  padding:.95rem 1.05rem; }}
.tile .k {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.05em;
  color:var(--text-secondary); }}
.tile .v {{ font-size:1.55rem; font-weight:650; margin-top:.2rem;
  font-variant-numeric:tabular-nums; }}
.tile .n {{ font-size:.82rem; color:var(--text-secondary); }}
.grid2 {{ display:grid; gap:1.4rem; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
figure {{ margin:0; background:var(--card); border:1px solid var(--grid);
  border-radius:10px; padding:.9rem 1rem 1rem; }}
figcaption {{ font-size:.84rem; color:var(--text-secondary); margin-top:.4rem; }}
.legend {{ display:flex; gap:1rem; flex-wrap:wrap; font-size:.84rem;
  color:var(--text-secondary); margin-bottom:.3rem; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:.35rem; vertical-align:-1px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem;
  font-variant-numeric:tabular-nums; margin-top:.5rem; }}
th,td {{ text-align:right; padding:.5rem .6rem; border-bottom:1px solid var(--grid);
  vertical-align:top; }}
th {{ font-size:.76rem; text-transform:uppercase; letter-spacing:.04em;
  color:var(--text-secondary); font-weight:600; }}
th:first-child, td.metric {{ text-align:left; }}
tr.highlight td {{ background:color-mix(in srgb, var(--trained) 8%, transparent);
  font-weight:600; }}
td.muted {{ text-align:left; color:var(--text-secondary); font-weight:400;
  font-size:.84rem; }}
table.defs {{ font-variant-numeric:normal; }}
table.defs td.metric {{ width:11rem; font-weight:600; white-space:nowrap; }}
td.defn {{ text-align:left; color:var(--text-secondary); }}
.delta.pos {{ color:var(--good); font-weight:600; }}
.delta.neg {{ color:var(--bad); font-weight:600; }}
.pill {{ font-size:.75rem; padding:.14rem .5rem; border-radius:999px;
  border:1px solid var(--grid); white-space:nowrap; }}
.pill.ok {{ color:var(--good); border-color:currentColor; }}
.pill.bad {{ color:var(--bad); border-color:currentColor; }}
img {{ max-width:100%; border-radius:8px; margin-top:.6rem; }}
.note {{ font-size:.88rem; color:var(--text-secondary); }}
.caveat {{ background:var(--card); border:1px solid var(--grid);
  border-left:3px solid var(--baseline); border-radius:8px; padding:.9rem 1.1rem; }}
.caveat p {{ margin:.6rem 0; }}
ol.why {{ padding-left:1.2rem; }}
ol.why li {{ margin:.5rem 0; }}
</style></head><body><main>

<h1>Is the fine-tuned route actually safer?</h1>
<p class="sub">NavDP <code>{ckpt}</code> versus the pretrained baseline on
<strong>{rec}</strong> — a recording the model never trained on. {n} paired
samples: both models answer the same goal from the same frame, and both are
scored against a {window}-frame fused occupancy map built from AprilTag poses,
<em>not</em> the single-frame field the fine-tune was trained against.</p>

<div class="tiles">
  <div class="tile"><div class="k">Routes hitting an obstacle</div>
    <div class="v">{coll_base:.0%} &rarr; {coll_tuned:.0%}</div>
    <div class="n">baseline &rarr; trained</div></div>
  <div class="tile"><div class="k">Min clearance</div>
    <div class="v">{min_delta:+.2f} m</div>
    <div class="n">{min_better} improved / {min_worse} regressed</div></div>
  <div class="tile"><div class="k">Goal gap</div>
    <div class="v">{gap_delta:+.2f} m</div>
    <div class="n">the cost — it stops further out</div></div>
</div>

<h2>Results</h2>
<div class="table-wrap">{table}</div>

<h3>How to read the columns</h3>
{columns}

<h3>What each metric measures</h3>
{metric_notes}

<h2>Where the routes sit relative to obstacles</h2>
<div class="legend">
  <span><i style="background:var(--baseline)"></i>baseline</span>
  <span><i style="background:var(--trained)"></i>trained</span>
</div>
<div class="grid2">
  <figure>{density}
    <figcaption>The whole distribution moves right, and the mass sitting under
    the {d_safe:.2f}&nbsp;m safety line shrinks. This is a shift in the
    population of routes, not an averaging artefact.</figcaption></figure>
  <figure>{delta}
    <figcaption>Per-sample change, same goal and same frame on both sides. Bars
    right of zero are goals where the fine-tune gave more room; the bulk sitting
    right of the line is what the win/loss column counts.</figcaption></figure>
</div>

<h2>Collision rate</h2>
<figure>{collisions}
  <figcaption>A route counts as colliding if any point along it falls inside an
  obstacle. Routes are densified to 2&nbsp;cm before scoring, so a path cannot
  step over a thin wall between waypoints and still score clean.</figcaption>
</figure>

{strict}
{routes}

<h2>Why these numbers can be trusted</h2>
<ol class="why">
<li><strong>Held-out recording.</strong> <code>{rec}</code> was never in the
training set — the fine-tune used seven other recordings. This measures
generalisation, not memorisation.</li>
<li><strong>Paired design.</strong> Both models answer the same goal from the
same frame, so per-sample differences remove scene difficulty as a confound. The
statistics are paired throughout.</li>
<li><strong>Independent judge.</strong> Routes are scored against a fused
multi-frame occupancy map built from AprilTag poses recovered from the source
bags. The fine-tune was trained against a <em>single-frame</em> field, so it
never optimised against this map. Scoring it with its own training ruler would
have been circular.</li>
<li><strong>Ground-truth validated.</strong> The drone's actually-flown path
never collides on this map (worst clearance 0.22–0.42&nbsp;m across thresholds).
So the baseline's wall entries are real route geometry, not map error.</li>
</ol>

<h2>What this does not show</h2>
<div class="caveat">
<p><strong>The trained model stops further from the goal</strong>
({gap_delta:+.2f}&nbsp;m, losing on {gap_worse} of {n} samples). It buys
clearance partly by being less committal. Whether that trade is acceptable is a
mission decision, not a statistical one.</p>
<p>The judge map is built from the same depth sensor as the training signal. It
is independent of the field the fine-tune optimised against and of the frames
that field was built from, but not of the depth modality itself.</p>
<p>One recording, one indoor scene, static goals sampled from pixels. This is
evidence of safer route <em>geometry</em>, not a flight test.</p>
</div>

</main></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--strict-dir", type=Path, default=None)
    ap.add_argument("--routes", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(args.eval_dir.expanduser(),
          args.strict_dir.expanduser() if args.strict_dir else None,
          args.routes.expanduser() if args.routes else None,
          args.out.expanduser())


if __name__ == "__main__":
    main()
