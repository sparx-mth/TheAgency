"""Complete-batch multi-story metrics and the predeclared recording gallery."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows


def write_multifloor_report(root, data):
    root = Path(root)
    campaign = json.loads((root / "campaign.json").read_text())
    backends = list(dict.fromkeys(job[0] for job in campaign["jobs"]))
    counts = Counter(row["scene"] for row in data["episodes"])
    expected = {"%s/%06d" % (scene, i) for scene, count in counts.items() for i in range(count)}
    statistics, details, videos = {}, [], []
    for backend in backends:
        records = [row for scene in data["scenes"]
                   for _, row in read_episode_rows(root / backend / scene / "episodes.jsonl")]
        if len(records) != len(expected) or {r.episode_id for r in records} != expected:
            raise RuntimeError("Missing, duplicate or substituted campaign episodes for " + backend)
        statistics[backend] = asdict(summarise(records))
        for scene in data["scenes"]:
            run = root / backend / scene
            config = json.loads((run / "run.json").read_text())["config"]
            if config["source_sha256"] != campaign["source_sha256"]:
                raise RuntimeError("Mixed policy sources in campaign results")
            sidecar = [json.loads(line) for line in (run / "evaluation_diagnostics.jsonl").read_text().splitlines()]
            native = {row["episode_id"]: row for row in sidecar}
            for record in records:
                if record.scene_id != scene:
                    continue
                if record.episode_id not in native:
                    raise RuntimeError("Missing evaluator diagnostics: " + record.episode_id)
                policy = record.agent_info.get("policy", {})
                atlas = (policy.get("building") or {}).get("atlas", {})
                details.append({"explorer": backend, "scene": scene, "episode_id": record.episode_id,
                                "target": record.target_category, "success": record.success, "spl": record.spl,
                                "actions": record.steps, "distance_to_goal_m": record.distance_to_goal_m,
                                "observed_floors": len(atlas.get("floors", [])),
                                "observed_connections": len(atlas.get("connections", [])),
                                "agent_error": record.agent_error, **native[record.episode_id]})
            for episode_id in config["recording"]["episode_ids"]:
                key = hashlib.sha256(episode_id.encode()).hexdigest()[:12]
                path = Path(backend) / scene / "recordings" / key / "video.mp4"
                if not (root / path).is_file() or (root / path).stat().st_size == 0:
                    raise RuntimeError("Missing predeclared recording: " + str(path))
                videos.append({"explorer": backend, "scene": scene, "episode_id": episode_id, "path": str(path)})
    if len(videos) != campaign["recordings_expected"]:
        raise RuntimeError("Campaign recording count disagrees with its frozen plan")
    errors = [row for row in details if row["agent_error"]]
    report = {"protocol_id": data["schema"], "complete": True, "operational": not errors,
              "published_benchmark_comparable": False, "held_out_claim": False,
              "semantic_limit": data["generation"]["semantic_limit"], "statistics": statistics,
              "episodes": details, "recordings": videos,
              "cross_floor_episodes": sum(row.get("first_cross_floor_action") is not None for row in details),
              "agent_errors": len(errors)}
    (root / "statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (root / "recordings.json").write_text(json.dumps(videos, indent=2) + "\n")
    lines = ["# Multi-story ZSON development results", "",
             "Frozen starts and policy; all requested episodes are retained, including failures.", "",
             "**Not published Gibson validation.** The native metric is 3D geodesic travel to annotated reference-floor success regions. "
             "Camera inspection uses the existing LOOK actions; the limit is still 500 actions. "
             "Upper-floor target annotations are incomplete, so this is a cross-floor integration test, not a full-building ObjectNav accuracy claim.", "",
             "| Explorer | Environment / episode | Target | Success | SPL | Actions | Floors / links | Cross-floor action |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in details:
        lines.append("| {explorer} | {episode_id} | {target} | {success} | {spl:.4f} | {actions} | {observed_floors} / {observed_connections} | {first_cross_floor_action} |".format(**row))
    lines += ["", "Agent errors: %d. Recorded episodes: %d (selected before evaluation)." % (len(errors), len(videos)),
              "", "Open index.html for the videos; statistics.json includes all metrics and evaluator diagnostics."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    page = ["<!doctype html><meta charset='utf-8'><title>Multi-story ZSON evaluation</title>",
            "<style>body{font:17px sans-serif;background:#181818;color:#eee;margin:30px}a{color:#8cf}video{width:1100px;max-width:100%}section{margin:35px 0}</style>",
            "<h1>Multi-story ZSON: %d buildings, %d episodes</h1>" % (len(counts), len(expected)),
            "<p>All episodes retained. One preselected recording per environment when --record-first is used. "
            "Reference-floor annotations only: development integration test, not published benchmark accuracy.</p>",
            "<p><a href='RESULTS.md'>All episode results</a> · <a href='statistics.json'>Metrics and diagnostics</a> · <a href='campaign.json'>Frozen campaign</a></p>"]
    for video in videos:
        row = next(r for r in details if r["episode_id"] == video["episode_id"] and r["explorer"] == video["explorer"])
        title = "%s — %s — %s — success %s — SPL %.3f" % (video["scene"], video["explorer"], row["target"], row["success"], row["spl"])
        page.append("<section><h2>%s</h2><video controls preload='metadata' src='%s'></video></section>" % (html.escape(title), html.escape(video["path"], quote=True)))
    (root / "index.html").write_text("\n".join(page) + "\n")
    if errors:
        raise RuntimeError("Campaign completed with %d agent errors; see statistics.json" % len(errors))
    return report

