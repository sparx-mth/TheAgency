"""One self-contained HTML page holding everything a run produced.

    python -m ...world_goal.report --run ~/navdp_world_goal/run1 \
        --dataset ~/navdp_world_goal/dataset

Gathers ``run.json``, the training figures, ``evaluation.json``, the route
panels and -- if a closed-loop comparison has been flown -- ``flights.json``,
and writes ``report.html``. Everything is inlined (SVG as markup, PNG as a data
URI), so the file can be copied anywhere and still renders.

The page is ordered the way the question is actually asked: what was trained,
what it was trained on, what the loss did, whether the flying got better on
geometry the model never saw, and what is still wrong. The caveats section is
not decoration -- a fine-tune that buys clearance by refusing to commit to the
goal is a real failure mode and it should be visible on the same page as the
headline number.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0 auto;
       max-width: 1100px; padding: 2rem 1.2rem 5rem; line-height: 1.55; }
h1 { font-size: 1.7rem; margin-bottom: .2rem; }
h2 { font-size: 1.2rem; margin-top: 2.4rem; border-bottom: 1px solid #8883;
     padding-bottom: .3rem; }
h3 { font-size: 1rem; margin-top: 1.4rem; }
.sub { opacity: .65; font-size: .9rem; margin-top: 0; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; margin: .8rem 0; }
th, td { padding: .35rem .55rem; text-align: right; border-bottom: 1px solid #8882; }
th:first-child, td:first-child { text-align: left; }
thead th { font-weight: 600; opacity: .8; }
.good { color: #1b8a3a; font-weight: 600; }
.bad  { color: #c62828; font-weight: 600; }
.flat { opacity: .6; }
.cards { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1rem 0; }
.card { border: 1px solid #8883; border-radius: 8px; padding: .7rem 1rem; min-width: 150px; }
.card .v { font-size: 1.35rem; font-weight: 600; }
.card .k { font-size: .76rem; opacity: .65; text-transform: uppercase;
           letter-spacing: .04em; }
figure { margin: 1rem 0; } figure img, figure svg { max-width: 100%; height: auto; }
figcaption { font-size: .82rem; opacity: .7; margin-top: .3rem; }
pre { background: #8881; padding: .7rem; border-radius: 6px; overflow-x: auto;
      font-size: .78rem; }
.scroll { overflow-x: auto; }
.note { border-left: 3px solid #2a78d6; padding: .5rem .9rem; background: #2a78d611;
        border-radius: 0 6px 6px 0; margin: 1rem 0; font-size: .9rem; }
"""


def embed_svg(path: Path) -> str:
    """Inline an SVG, stripping its XML prologue so it nests in the page."""
    if not path.exists():
        return ""
    text = path.read_text()
    start = text.find("<svg")
    return text[start:] if start >= 0 else ""


def embed_png(path: Path) -> str:
    """A PNG as a data URI, so the page stays one file."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.stem}">'


def figure_block(run: Path, stem: str, caption: str) -> str:
    body = embed_svg(run / f"{stem}.svg") or embed_png(run / f"{stem}.png")
    if not body:
        return ""
    return f"<figure>{body}<figcaption>{caption}</figcaption></figure>"


def cards(items: List[tuple]) -> str:
    return ('<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{key}</div><div class="v">{value}</div></div>'
        for key, value in items) + "</div>")


def verdict_class(verdict: str) -> str:
    return {"better": "good", "WORSE": "bad"}.get(verdict, "flat")


def paired_table(paired: Dict[str, Dict], ceiling: Optional[Dict[str, Dict]]) -> str:
    """The headline table: baseline, trained, the expert ceiling, and the verdict."""
    head = ("<thead><tr><th>metric</th><th>baseline</th><th>trained</th>"
            "<th>expert</th><th>&Delta; (better +)</th><th>win / loss</th>"
            "<th>p</th><th>effect</th><th>verdict</th></tr></thead>")
    rows = []
    for metric, result in paired.items():
        expert = ceiling.get(metric, {}).get("arm_mean") if ceiling else None
        rows.append(
            f"<tr><td>{metric}</td><td>{result['ref_mean']:.3f}</td>"
            f"<td>{result['arm_mean']:.3f}</td>"
            f"<td>{'-' if expert is None else f'{expert:.3f}'}</td>"
            f"<td>{result['mean_delta']:+.3f}</td>"
            f"<td>{result['n_better']} / {result['n_worse']}</td>"
            f"<td>{result['p_value']:.2e}</td><td>{result['effect_size']:+.2f}</td>"
            f"<td class='{verdict_class(result['verdict'])}'>{result['verdict']}</td></tr>")
    return f'<div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def turn_table(buckets: Dict[str, Dict]) -> str:
    """The same comparison split by how hard the route turns."""
    head = ("<thead><tr><th>turn</th><th>n</th><th>min clearance</th>"
            "<th>&Delta;</th><th>goal gap</th><th>&Delta;</th>"
            "<th>centre offset</th><th>&Delta;</th></tr></thead>")
    rows = []
    for name, entry in buckets.items():
        if "note" in entry:
            rows.append(f"<tr><td>{name}</td><td>{entry['n']}</td>"
                        f"<td colspan='6' class='flat'>{entry['note']}</td></tr>")
            continue
        cells = []
        for key in ("min_clear_m", "goal_gap_m", "centre_offset_m"):
            item = entry[key]
            cells.append(f"<td>{item['baseline']:.3f} &rarr; {item['trained']:.3f}</td>"
                         f"<td class='{verdict_class(item['verdict'])}'>"
                         f"{item['delta']:+.3f}</td>")
        rows.append(f"<tr><td>{name}</td><td>{entry['n']}</td>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def dataset_table(stats: Dict) -> str:
    head = ("<thead><tr><th>split</th><th>samples</th><th>frames</th>"
            "<th>recordings</th><th>sharp turns</th><th>arrivals</th>"
            "<th>worst label clearance</th></tr></thead>")
    rows = []
    for split in ("train", "val", "test"):
        entry = stats.get(split) or {}
        if not entry.get("samples"):
            rows.append(f"<tr><td>{split}</td><td colspan='6' class='bad'>empty</td></tr>")
            continue
        rows.append(
            f"<tr><td>{split}</td><td>{entry['samples']}</td><td>{entry['frames']}</td>"
            f"<td>{entry['recordings']}</td>"
            f"<td>{entry['turn_buckets']['sharp_ge40']}</td>"
            f"<td>{entry['arrival_samples']}</td>"
            f"<td>{entry['label_min_clear_m']['min']:.2f} m</td></tr>")
    return f"<table>{head}<tbody>{''.join(rows)}</tbody></table>"


def flights_table(flights: Dict) -> str:
    """Closed-loop results, when a flight comparison has been run."""
    head = ("<thead><tr><th>arm</th><th>missions</th><th>reached goal</th>"
            "<th>collisions</th><th>min clearance</th><th>path length</th>"
            "<th>time</th></tr></thead>")
    rows = []
    for arm, entry in flights.get("arms", {}).items():
        rows.append(
            f"<tr><td>{arm}</td><td>{entry.get('missions', 0)}</td>"
            f"<td>{entry.get('reached', 0)}</td><td>{entry.get('collisions', 0)}</td>"
            f"<td>{entry.get('min_clear_m', float('nan')):.2f} m</td>"
            f"<td>{entry.get('path_len_m', float('nan')):.1f} m</td>"
            f"<td>{entry.get('duration_s', float('nan')):.0f} s</td></tr>")
    return f"<table>{head}<tbody>{''.join(rows)}</tbody></table>"


def collect_flights(run: Path, flights_dir: Optional[Path]) -> Optional[Dict]:
    """Assemble the closed-loop arms from ``fly_navdp.py``'s per-arm summaries.

    Each arm writes its own ``<arm>/summary.json`` inside the container; this
    gathers whatever has been copied back, so a report can be produced after one
    arm has flown and regenerated after the second.
    """
    if (run / "flights.json").exists():
        return json.loads((run / "flights.json").read_text())
    if flights_dir is None or not flights_dir.exists():
        return None
    arms = {path.parent.name: json.loads(path.read_text())
            for path in sorted(flights_dir.glob("*/summary.json"))}
    return {"arms": arms} if arms else None


def build(run: Path, dataset_dir: Optional[Path],
          flights_dir: Optional[Path] = None) -> str:
    record = json.loads((run / "run.json").read_text())
    evaluation = (json.loads((run / "evaluation.json").read_text())
                  if (run / "evaluation.json").exists() else None)
    index = (json.loads((dataset_dir / "index.json").read_text())
             if dataset_dir and (dataset_dir / "index.json").exists() else None)
    flights = collect_flights(run, flights_dir)

    counts = record.get("param_counts", {})
    summary = record.get("summary", {})
    head = [("trainable", f"{counts.get('trainable', 0) / 1e6:.1f} M"),
            ("frozen", f"{counts.get('frozen', 0) / 1e6:.1f} M"),
            ("steps", summary.get("steps", "-")),
            ("best val loss", f"{summary.get('best_val_total', float('nan')):.4f}")]
    if evaluation:
        rate = evaluation["collision_rate"]
        head += [("test samples", evaluation["samples"]),
                 ("collisions base", f"{100 * rate['baseline']:.1f}%"),
                 ("collisions trained", f"{100 * rate['trained']:.1f}%")]

    parts = [f"<h1>NavDP world-goal fine-tune</h1>",
             f"<p class='sub'>{run.name} &middot; started {record.get('started', '?')} "
             f"&middot; git {record.get('git', '?')} &middot; "
             f"{record.get('environment', {}).get('gpu', 'unknown GPU')}</p>",
             cards(head)]

    parts.append("<h2>What was trained</h2>")
    model_cfg = record.get("config", {}).get("model", {})
    parts.append(
        "<p>The RGB DINOv2 trunk is frozen throughout &mdash; it is the general "
        "visual representation that keeps this from collapsing onto one "
        f"building. Trained: the Q-Former, the 16-layer fusion decoder, the "
        f"point-goal encoder and the action/critic heads "
        f"(<b>{counts.get('trainable', 0) / 1e6:.1f} M</b> of "
        f"{counts.get('total', 0) / 1e6:.1f} M).</p>")
    parts.append(f"<pre>{json.dumps(model_cfg, indent=2)}</pre>")

    if index:
        parts.append("<h2>What it was trained on</h2>")
        parts.append(dataset_table(index.get("stats", {})))
        parts.append("<div class='note'>Splits are three disjoint parts of the "
                     "building, not a random shuffle: the test wing is geometry "
                     "neither training nor checkpoint selection ever saw.<br>"
                     + "<br>".join(index.get("split_plan", [])) + "</div>")
        rejected = index.get("rejected_goals", {})
        if rejected:
            parts.append("<p class='sub'>Candidate goals rejected during label "
                         "generation: " + ", ".join(f"{k}={v}" for k, v in
                                                    list(rejected.items())[:8]) + "</p>")

    parts.append("<h2>Training</h2>")
    parts.append(figure_block(run, "training_curves",
                              "Every term of the objective, train (grey) against "
                              "validation (blue). Separated on purpose: a falling "
                              "total can hide clearance improving while "
                              "goal-reaching rots."))
    parts.append(figure_block(run, "training_nav",
                              "The metres-and-percent view, measured on the "
                              "validation split against the true map."))
    parts.append(figure_block(run, "training_health",
                              "Learning rate, gradient norm, throughput, memory."))

    if evaluation:
        parts.append("<h2>Held-out evaluation</h2>")
        parts.append(f"<p>{evaluation['samples']} samples from the "
                     f"<b>{evaluation['split']}</b> split, every arm answering the "
                     f"same frames with the same diffusion seed. "
                     f"<i>expert</i> is the label itself &mdash; the ceiling "
                     f"imitation can reach, not a competitor.</p>")
        parts.append(paired_table(evaluation["paired_vs_baseline"],
                                  evaluation.get("ceiling_vs_baseline")))
        parts.append("<h3>By how hard the route turns</h3>")
        parts.append("<p class='sub'>Corners are where a navigation policy fails; "
                     "an overall mean can hide getting worse at them.</p>")
        parts.append(turn_table(evaluation.get("by_turn", {})))
        parts.append(figure_block(run, "routes",
                                  "Held-out samples: green dashed is the expert, "
                                  "orange the pretrained baseline, blue the "
                                  "fine-tune. Black arrow is the aircraft, pink "
                                  "star or chevron the goal."))

    if flights:
        parts.append("<h2>Closed-loop flights</h2>")
        parts.append("<p>The same missions flown in PEGASUS with each model in "
                     "the loop. This is the only measurement that includes "
                     "compounding error.</p>")
        parts.append(flights_table(flights))

    parts.append("<h2>What this does not show</h2>")
    parts.append(
        "<ul>"
        "<li>One building. The test wing is unseen geometry, but it is the same "
        "architecture, lighting and renderer &mdash; survey a second scene for a "
        "real generalisation number.</li>"
        "<li>Open-loop. Each sample is scored on one prediction from one frame; "
        "errors that compound over a flight only appear in the closed-loop "
        "section.</li>"
        "<li>Simulated imagery. Nothing here says how the policy behaves on the "
        "real XTEND camera.</li>"
        "<li>Watch <code>goal_gap_m</code> next to the clearance metrics: buying "
        "safety by becoming less committal is a real and easy failure mode.</li>"
        "</ul>")

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>NavDP fine-tune &middot; {run.name}</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--flights", default=None,
                        help="directory of per-arm fly_navdp.py outputs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = Path(args.run).expanduser()
    dataset = Path(args.dataset).expanduser() if args.dataset else None
    flights = Path(args.flights).expanduser() if args.flights else None
    out = Path(args.out).expanduser() if args.out else run / "report.html"
    out.write_text(build(run, dataset, flights))
    print(f"[report] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
