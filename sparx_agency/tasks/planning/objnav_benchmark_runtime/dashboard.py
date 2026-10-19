"""Offline HTML/PNG/CSV demo results, derived from actual shared-harness rows."""
from __future__ import annotations

import csv
from dataclasses import asdict
import html
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows, strict_json, write_atomically)


STYLE = """body{background:#101722;color:#e4edf7;font:16px system-ui;margin:30px auto;max-width:1250px;padding:0 24px}
a{color:#7acfff}h1,h2{color:#a9dcff}.cards{display:flex;gap:18px;flex-wrap:wrap}.card{padding:18px 28px;background:#1c2b3b;border-radius:12px}.value{font-size:30px}img,video{max-width:100%;border:1px solid #405266;border-radius:8px}pre{white-space:pre-wrap;background:#182432;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #3c4b5c}small{color:#acbccb}.warning{background:#453322;padding:14px;border-left:5px solid #e0a953}button{padding:14px;font-size:17px;cursor:pointer}"""


def write_live_page(root):
    """Browser-visible live view; only local artifacts are fetched."""
    root = Path(root)
    text = """<!doctype html><meta charset="utf-8"><title>Gibson single-scene demo</title>
<style>%s</style><h1>Gibson: one-scene algorithm demo</h1>
<p class="warning">Waiting/running is not a measured result. No aggregate Gibson score is claimed.</p>
<p><a href="index.html">Final metrics and recordings</a></p><pre id="status">Waiting for first recorded frame…</pre>
<img id="frame" alt="Live RGB-D, observed map, detections, paths and room reasoning">
<script>setInterval(async()=>{try{let s=await(await fetch('live.json?t='+Date.now())).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);document.getElementById('frame').src='latest.jpg?t='+Date.now()}catch(e){}},1500)</script>""" % STYLE
    write_atomically(root / "live.html", text)


def _plots(root, records, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ("SR", "SPL", "SoftSPL")
    values = (summary.overall.success_rate, summary.overall.spl, summary.overall.soft_spl)
    axes[0].bar(labels, values, color=("#58b687", "#558cdf", "#c092df"))
    axes[0].set(ylim=(0, 1.05), ylabel="Fraction", title="Observed episodes, failures included")
    axes[1].bar([str(i + 1) for i in range(len(records))],
                [r.distance_to_goal_m if r.distance_to_goal_m is not None else 0 for r in records], color="#e3a34f")
    axes[1].set(xlabel="Episode", ylabel="DTG / distance to success (m)", title="Final reference FMM distance")
    fig.tight_layout()
    fig.savefig(root / "metrics.png", dpi=140)
    plt.close(fig)
    for directory in sorted((root / "recordings").glob("*")):
        trajectory = directory / "trajectory.csv"
        if not trajectory.exists():
            continue
        with trajectory.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        steps = [int(r["step"]) for r in rows]
        x, y = [float(r["x"]) for r in rows], [float(r["y"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(x, y, "-", color="#3b9861")
        axes[0].scatter([x[0], x[-1]], [y[0], y[-1]], c=["blue", "red"])
        axes[0].set(xlabel="ENU x (m)", ylabel="ENU y (m)", title="Executed path: blue=start, red=end")
        axes[0].set_aspect("equal", adjustable="datalim")
        dtg = [float(r["distance_to_goal_m"]) if r["distance_to_goal_m"] else float("nan") for r in rows]
        axes[1].plot(steps, dtg, color="#c38b30", label="Evaluator-only DTG")
        axes[1].set(xlabel="Executed action index", ylabel="Distance (m)", title="Progress toward success boundary")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(directory / "trajectory.png", dpi=140)
        plt.close(fig)


def write_dashboard(root):
    """Summarise completed rows only; a single episode has binary SR, not a benchmark SR."""
    root = Path(root)
    records = [row for _, row in read_episode_rows(root / "episodes.jsonl")]
    if not records:
        raise ValueError("No completed episode; metrics cannot be invented")
    summary = summarise(records)
    _plots(root, records, summary)
    overall = summary.overall
    values = {"SR": overall.success_rate, "SPL": overall.spl,
              "DTG (m)": overall.distance_to_goal_m, "SoftSPL": overall.soft_spl}
    cards = "".join('<div class="card">%s<div class="value">%s</div></div>' %
                    (html.escape(key), "unreachable" if value is None else "%.3f" % value)
                    for key, value in values.items())
    without_stop = sum(record.success and not record.stop_called for record in records)
    if without_stop:
        cards += ('<p class="warning"><strong>Protocol success without STOP: %d episode(s).</strong> '
                  'SemExp scores the final position inside the success region even at the step limit. '
                  'This does not establish that the algorithm explicitly declared the target found.</p>'
                  % without_stop)
    rows = []
    with (root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("episode_id", "scene", "target", "success", "spl", "dtg_m", "soft_spl",
                         "steps", "wall_s", "stop_called", "termination", "agent_error"))
        for record in records:
            row = (record.episode_id, record.scene_id, record.target_category, int(record.success), record.spl,
                   record.distance_to_goal_m, record.soft_spl, record.steps, record.wall_s,
                   record.stop_called, record.termination, record.agent_error or "")
            writer.writerow(row)
            rows.append("<tr>" + "".join("<td>%s</td>" % html.escape(str(value)) for value in row) + "</tr>")
    sections = []
    for directory in sorted((root / "recordings").glob("*")):
        info_path = directory / "episode.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        relative = directory.relative_to(root).as_posix()
        title = html.escape("%s — %s" % (info["episode_id"], info["target"]))
        video = '<video controls preload="metadata" src="%s/video.mp4"></video>' % relative if (directory / "video.mp4").exists() else "<p>Recording incomplete; see encoder log.</p>"
        reasons = []
        trace = directory / "steps.jsonl"
        previous = None
        if trace.exists():
            for line in trace.read_text().splitlines():
                item = json.loads(line)
                reasoning = item["method"].get("reasoning", {})
                if reasoning and reasoning != previous:
                    reasons.append("Step %d: %s" % (item["step"], json.dumps(reasoning, indent=2)))
                    previous = reasoning
        sections.append('<h2>%s</h2>%s<p><small>Playback is accelerated: one frame per decision, not wall time.</small></p><img src="%s/trajectory.png" alt="Executed path and DTG curve"><p><a href="%s/steps.jsonl">Decisions/detections/reasoning JSONL</a> · <a href="%s/trajectory.csv">Trajectory CSV</a> · <a href="%s/metrics.json">Episode metrics</a></p><details><summary>Recorded room labels and LLM reasons</summary><pre>%s</pre></details>' %
                        (title, video, relative, relative, relative, relative,
                         html.escape("\n\n".join(reasons) or "No room-reasoning query completed in this episode.")))
    text = '<!doctype html><meta charset="utf-8"><title>Gibson demo results</title><style>%s</style><h1>Gibson one-scene demo results</h1><p class="warning">%d completed episode(s). This is a diagnostic subset, NOT a full Gibson benchmark. One-episode SR is 0 or 1. DTG is SemExp distance to the success region; SoftSPL is supplementary. Published SOTA equivalence is not established.</p><div class="cards">%s</div><p><a href="metrics.csv">All metrics CSV</a> · <a href="run.json">Configuration/provenance</a> · <a href="comparison.md">Qualified paper comparison</a></p><img src="metrics.png" alt="Metric graphs"><table><tr>%s</tr>%s</table>%s' % (
        STYLE, len(records), cards,
        "".join("<th>%s</th>" % h for h in ("Episode", "Scene", "Target", "Success", "SPL", "DTG", "SoftSPL",
                                           "Steps", "Wall s", "STOP", "Termination", "Error")),
        "".join(rows), "".join(sections))
    write_atomically(root / "index.html", text)
    write_atomically(root / "demo_metrics.json", strict_json(asdict(summary), "demo summary", indent=2))
    write_atomically(root / "live.json", strict_json({"completed": True, "episodes": len(records),
                                                    "metrics": values, "report": "index.html"}, "live status"))
    return root / "index.html"

