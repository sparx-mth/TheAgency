"""Terminal reporting for the five-scene campaign: one line per finished episode, one table at the end."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord

COLUMNS = ("scene", "goal", "SR", "SPL", "DTG_m", "runtime_s", "steps", "termination")


def recording_video(directory: Path, record: EpisodeRecord) -> Optional[Path]:
    """The recorder's video for ``record``, or None when it was not written.

    Mirrors ``EpisodeRecorder.begin``: the recording key is the first twelve
    hex digits of the SHA-256 of the episode id.
    """
    key = hashlib.sha256(record.episode_id.encode()).hexdigest()[:12]
    path = Path(directory) / "recordings" / key / "video.mp4"
    return path if path.is_file() else None


def _dtg(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.3f" % value


def status_line(record: EpisodeRecord, video: Optional[Path] = None, job_s: Optional[float] = None) -> str:
    """One explicit line for a scored episode: scene, goal, SR, SPL, DTG, runtime.

    Args:
        record: The scored episode.
        video: Its HUD recording, when written.
        job_s: Wall time of the whole scene job (simulator start-up included),
            as opposed to ``record.wall_s``, the episode's own runtime.
    """
    text = ("EPISODE COMPLETE  scene=%s  goal=%s  SR=%d  SPL=%.4f  DTG=%s m  runtime=%.1f s  "
            "steps=%d  termination=%s" % (record.scene_id, record.target_category, int(record.success),
                                         record.spl, _dtg(record.distance_to_goal_m), record.wall_s,
                                         record.steps, record.termination))
    if job_s is not None:
        text += "  job=%.0f s" % job_s
    return text + "  video=%s" % (video if video is not None else "not written")


def failure_line(scene: str, exit_code: int, log: Path) -> str:
    """The line for a scene job that ended without a scored row."""
    return "EPISODE FAILED    scene=%s  exit=%d  no scored episode; see %s" % (scene, exit_code, log)


def table_rows(records: Sequence[EpisodeRecord]) -> list:
    rows = [(r.scene_id, r.target_category, "%d" % int(r.success), "%.4f" % r.spl,
             _dtg(r.distance_to_goal_m), "%.1f" % r.wall_s, "%d" % r.steps, r.termination)
            for r in records]
    overall = summarise(records).overall
    rows.append(("MEAN (%d ep)" % len(records), "", "%.3f" % overall.success_rate, "%.4f" % overall.spl,
                 _dtg(overall.distance_to_goal_m), "%.1f" % (sum(r.wall_s for r in records) / len(records)),
                 "%.1f" % (sum(r.steps for r in records) / len(records)), ""))
    return rows


def summary_table(records: Sequence[EpisodeRecord]) -> str:
    """A fixed-width table of every completed episode and a mean row.

    Raises:
        ValueError: No completed episode to tabulate.
    """
    if not records:
        raise ValueError("No completed episode to tabulate")
    rows = [COLUMNS] + table_rows(records)
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(COLUMNS))]
    line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(rows[0]))
    rule = "-" * len(line)
    body = [line, rule]
    for row in rows[1:-1]:
        body.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    body.append(rule)
    body.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(rows[-1])))
    return "\n".join(body)
