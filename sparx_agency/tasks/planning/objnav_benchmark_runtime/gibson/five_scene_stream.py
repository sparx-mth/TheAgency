"""Live per-episode streaming for the five-scene campaign.

A scene job (``gibson.run``) appends one row to its ``episodes.jsonl`` as each
episode finishes. ``EpisodeTail`` follows that file while the job is running so
the campaign can report an episode the moment it is scored rather than when the
whole scene job ends; ``running_mean_line`` and ``write_benchmark_results``
keep the running aggregate and the campaign-wide ``benchmark_results.json`` /
``benchmark_results.csv`` current after every episode.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark.results_io import read_episode_rows, strict_json, write_atomically

RESULT_COLUMNS = ("index", "scene", "episode_id", "goal", "SR", "SPL", "SoftSPL", "DTG_m", "steps", "runtime_s",
                  "STOP", "termination", "video")


class EpisodeTail:
    """Yields the rows a scene job has appended to ``episodes.jsonl`` since the last poll.

    A row is only considered once the file holds a complete line for it; a
    partially written last line is left for the next poll instead of raising,
    because the writer is still alive.
    """

    def __init__(self, directory: Path):
        self._source = Path(directory) / "episodes.jsonl"
        self._seen = 0

    def poll(self) -> List[EpisodeRecord]:
        if not self._source.exists():
            return []
        try:
            rows = [row for _, row in read_episode_rows(self._source)]
        except ValueError:
            return []  # truncated last line: the job is mid-write
        fresh = rows[self._seen:]
        self._seen = len(rows)
        return fresh


def _dtg(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.3f" % value


def running_mean_line(records: Sequence[EpisodeRecord], total: int) -> str:
    """The running aggregate over every scored episode so far, failures included."""
    overall = summarise(records).overall
    n = len(records)
    return ("RUNNING MEAN      episodes=%d/%d  SR=%.3f  SPL=%.4f  SoftSPL=%.4f  DTG=%s m  steps=%.1f  "
            "runtime=%.1f s" % (n, total, overall.success_rate, overall.spl, overall.soft_spl,
                                _dtg(overall.distance_to_goal_m), sum(r.steps for r in records) / n,
                                sum(r.wall_s for r in records) / n))


def result_row(index: int, record: EpisodeRecord, video: Optional[Path]) -> Dict[str, object]:
    return {"index": index, "scene": record.scene_id, "episode_id": record.episode_id,
            "goal": record.target_category, "SR": int(record.success), "SPL": record.spl,
            "SoftSPL": record.soft_spl, "DTG_m": record.distance_to_goal_m, "steps": record.steps,
            "runtime_s": record.wall_s, "STOP": record.stop_called, "termination": record.termination,
            "video": str(video) if video is not None else None}


def write_benchmark_results(root: Path, rows: Sequence[Dict[str, object]], records: Sequence[EpisodeRecord],
                            total: int, campaign: Dict[str, object]) -> None:
    """Rewrite ``benchmark_results.json`` and ``benchmark_results.csv`` atomically.

    The JSON carries the campaign identity, every per-episode row in finishing
    order and the running ``BenchmarkSummary``; the CSV is the rows alone. Both
    are complete after every episode, so a killed campaign leaves nothing stale.
    """
    root = Path(root)
    summary = asdict(summarise(records)) if records else None
    payload = dict(campaign, episodes_total=total, episodes_scored=len(records), episodes=list(rows),
                   running_summary=summary)
    write_atomically(root / "benchmark_results.json", strict_json(payload, "benchmark results", indent=2))
    with (root / "benchmark_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in RESULT_COLUMNS})
