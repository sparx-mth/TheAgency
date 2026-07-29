"""How far a flight-recording campaign has got, read off the disk it writes to.

Progress is derived from the recording directories themselves, never from the
campaign manifest: :mod:`sim_flight_recording.collect` writes
``campaign_w*.json`` once, when a worker exits, so a manifest-based reading
shows nothing at all for the first hour and then jumps. A recording directory,
by contrast, gains its ``meta.json`` the moment its flight ends, which makes it
an accurate incremental record of completed work.

A directory with images but no ``meta.json`` is the flight currently in the air
(or one abandoned by a killed worker) — counted separately, because it is real
disk and real progress but not yet a usable sample.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Episode:
    """One finished flight, as its ``meta.json`` describes it."""

    name: str
    path: Path
    outcome: str
    outcome_ok: bool
    frames: int
    goal_error_m: float
    estimator_drift_m: float
    finished_at: float

    @classmethod
    def read(cls, directory: Path) -> Optional["Episode"]:
        """Load one recording, or ``None`` if it has not finished writing."""
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        frames = meta.get("frames")
        if frames is None:
            rgb = directory / "rgb"
            frames = sum(1 for _ in rgb.iterdir()) if rgb.is_dir() else 0
        return cls(
            name=directory.name,
            path=directory,
            outcome=str(meta.get("outcome", "unknown")),
            outcome_ok=bool(meta.get("outcome_ok", False)),
            frames=int(frames),
            goal_error_m=float(meta.get("goal_error_m") or 0.0),
            estimator_drift_m=float(meta.get("estimator_drift_m") or 0.0),
            finished_at=meta_path.stat().st_mtime,
        )


@dataclass
class CollectionProgress:
    """Everything the dashboard shows about a running campaign."""

    root: Path
    episodes: List[Episode] = field(default_factory=list)
    in_flight: int = 0
    target_episodes: Optional[int] = None
    started_at: Optional[float] = None

    @property
    def done(self) -> int:
        """Flights that finished and wrote their metadata."""
        return len(self.episodes)

    @property
    def landed(self) -> int:
        """Flights that reached their goal — the ones worth the most."""
        return sum(1 for episode in self.episodes if episode.outcome == "landed")

    @property
    def frames(self) -> int:
        """Total recorded frames across every finished flight."""
        return sum(episode.frames for episode in self.episodes)

    @property
    def outcomes(self) -> Dict[str, int]:
        """How many flights ended each way, commonest first."""
        counts: Dict[str, int] = {}
        for episode in self.episodes:
            counts[episode.outcome] = counts.get(episode.outcome, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    @property
    def fraction(self) -> Optional[float]:
        """Completed share of the target, or ``None`` when open-ended."""
        if not self.target_episodes:
            return None
        return min(1.0, self.done / self.target_episodes)

    def rate_per_hour(self, window: int = 40) -> Optional[float]:
        """Recent throughput in flights per hour.

        Measured over the last ``window`` finished flights rather than the whole
        campaign, so the figure tracks the current worker count instead of being
        dragged down by the boot time of a run that has since scaled up.
        """
        if len(self.episodes) < 2:
            return None
        recent = sorted(self.episodes, key=lambda episode: episode.finished_at)[-window:]
        span = recent[-1].finished_at - recent[0].finished_at
        if span <= 0:
            return None
        return 3600.0 * (len(recent) - 1) / span

    def eta_seconds(self) -> Optional[float]:
        """Seconds until the target is met at the recent rate."""
        rate = self.rate_per_hour()
        if not rate or not self.target_episodes:
            return None
        remaining = self.target_episodes - self.done
        return 3600.0 * remaining / rate if remaining > 0 else 0.0


def scan(root: Path, target_episodes: Optional[int] = None) -> CollectionProgress:
    """Read every recording under ``root``, however deeply it is nested.

    The supervisor gives each worker launch its own directory, so recordings sit
    one level down; a hand-run ``collect.py`` writes them at the top. Recursing
    on the presence of an ``rgb/`` subdirectory handles both without being told
    which layout it is looking at.
    """
    progress = CollectionProgress(root=root, target_episodes=target_episodes)
    if not root.is_dir():
        return progress

    earliest: Optional[float] = None
    for candidate in sorted(root.rglob("rgb")):
        if not candidate.is_dir():
            continue
        directory = candidate.parent
        episode = Episode.read(directory)
        if episode is None:
            progress.in_flight += 1
        else:
            progress.episodes.append(episode)
        try:
            created = directory.stat().st_mtime
        except OSError:
            continue
        earliest = created if earliest is None else min(earliest, created)

    progress.started_at = earliest
    return progress


def worker_logs(root: Path) -> List[Path]:
    """Every worker log under ``root``, newest last."""
    return sorted(root.rglob("worker*.log"), key=lambda path: path.stat().st_mtime)


def live_worker_count(root: Path, stale_after_s: float = 120.0) -> int:
    """Workers whose log was written to recently enough to still be alive.

    A worker that has crashed leaves its log behind, so presence alone says
    nothing. Isaac Sim logs continuously while flying, which makes recency a
    reliable liveness test and needs no access to the container's process table.
    """
    now = time.time()
    alive = 0
    for log in worker_logs(root):
        try:
            if now - log.stat().st_mtime < stale_after_s:
                alive += 1
        except OSError:
            continue
    return alive
