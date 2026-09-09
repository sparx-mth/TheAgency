"""Assemble a recorded run into a timeline of :class:`NavFrame` for replay.

A run folder holds several independent lanes (see :mod:`.schema`): the BEV maps
(:mod:`.bev_source`), the route layers (:mod:`.route_source`), the planner's
events (:mod:`.event_source`) and -- on the Sphera exploration stack -- the
reference being chased, the tracker's verdict, the map-quality counters and,
under ``ros2/``, what the drone was actually told and what it actually did.
This module joins them onto one spine and builds frames from it.

**The spine** is the certainty CSV when the flight wrote one (XTEND), otherwise
the recorder's own ``telemetry.jsonl`` -- which on Sphera, where no CSV exists
at all, is the timeline in its own right (:mod:`.timeline`). Every other lane is
resolved by an *as-of* join: the newest row at or before the frame, exactly as
the drone saw it, latched topics and all.

**The join across recorders.** The ``ros2/`` lanes are written by a different
process, in a different container, on a different ROS clock; only the host
``wall`` clock is shared. Those lanes are therefore joined by wall time and the
ROS1 lanes by ROS time, with the frame's own wall stamp taken from the ROS1
recorder's median ``wall - t`` offset (:attr:`NavSession.clock`) whenever the
spine does not carry one. That offset, its spread and any join warning are
exposed on the session, so a bad join is reportable instead of a silently
shifted panel.

Frames are built lazily (:meth:`NavSession.build`) so a long flight's hundreds
of BEV snapshots are never all in memory at once.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from sparx_agency.tasks.planning.nav_debug import records, schema, timeline
from sparx_agency.tasks.planning.nav_debug.bev_source import BevSource
from sparx_agency.tasks.planning.nav_debug.event_source import (
    EventSource, classify_event,
)
from sparx_agency.tasks.planning.nav_debug.frame import NavFrame
from sparx_agency.tasks.planning.nav_debug.route_source import RouteSource
from sparx_agency.tasks.planning.nav_debug.state_source import StateSource
from sparx_agency.tasks.planning.nav_debug.sources import (
    ClockOffset, Stream, as_of_index, read_jsonl, to_float,
)
from sparx_agency.tasks.planning.nav_debug.why import why

__all__ = ["NavSession", "classify_event"]

_TRAIL_LEN = 48        # localization-trail length (frames)
_HIST_LEN = 64         # history-strip length (frames)
_LANE_MAX_AGE_S = 1.0  # a per-tick lane older than this is stale, not latched
_MAP_MAX_AGE_S = 5.0   # the map lane updates slowly; latch it for longer
#: Past this the ros2/ lanes belong to another flight, not to this one.
_JOIN_GAP_WARN_S = 60.0

# ROS1 lanes join on ROS time, ROS2 lanes on the host wall clock.
_ROS1_LANES = ("reference", "control", "mapping")
_ROS2_LANES = ("actuator", "truth", "altitude", "axis_trace")


class NavSession:
    """A loaded run: the frame spine plus an as-of index per recorded lane."""

    def __init__(self, run_dir: str, csv_path: Optional[str] = None,
                 ros2_dir: Optional[str] = None) -> None:
        """Load a run.

        Args:
            run_dir: The ROS1 recorder's run folder.
            csv_path: Certainty CSV to use as the spine. Defaults to the
                manifest's, then to one found beside the run; ``None`` on Sphera,
                where no CSV is written and telemetry.jsonl is the spine.
            ros2_dir: Where the ROS2 recorder's lanes live, if not
                ``<run_dir>/ros2``. The two recorders run in containers with no
                shared mount -- ``it`` cannot see the FALCON log directory -- so
                on Sphera the ROS2 half usually lands under its own workspace
                bind and is either collected into the run folder afterwards or
                pointed at here.
        """
        self.run_dir = run_dir
        self.ros2_dir = ros2_dir or os.path.join(run_dir, schema.ROS2_DIR)
        self.manifest = self._load_manifest(run_dir)
        self.warnings = []                   # type: List[str]

        self.csv_path = self._resolve_csv(run_dir, csv_path)
        self.rows, self.timeline_source = timeline.load(run_dir, self.csv_path)
        if not self.rows:
            raise ValueError(
                "no frames to replay in %r (need a certainty CSV or telemetry.jsonl)"
                % run_dir)
        self._stamps = [r["t"] for r in self.rows]
        if self.timeline_source == timeline.CERTAINTY_CSV:
            timeline.backfill_cmd_vz(self.rows, timeline.telemetry_rows(run_dir))

        self.bev_source = BevSource(run_dir, self.manifest)
        self.route_source = RouteSource(run_dir)
        self.event_source = EventSource(run_dir)
        self.lanes = self._open_lanes(run_dir, self.ros2_dir)
        self.clock, self.ros2_clock = self._estimate_clocks()
        self._check_join()
        self.state_source = StateSource(self.rows)
        self._err_series, self._speed_series = self._build_series()

    def __len__(self) -> int:
        return len(self.rows)

    # ── loading ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load_manifest(run_dir: str) -> dict:
        """The run manifest, or ``{}`` when it is absent or unreadable."""
        path = os.path.join(run_dir, schema.MANIFEST_FILE)
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    manifest = json.load(fh)
                return manifest if isinstance(manifest, dict) else {}
            except (OSError, IOError, ValueError):
                pass
        return {}

    def _resolve_csv(self, run_dir: str, csv_path: Optional[str]) -> Optional[str]:
        """The certainty CSV to use as the spine, or None (the Sphera case).

        The manifest's path was written inside the FALCON container, so it is
        checked and then ignored if that mount is not visible from here.
        """
        if csv_path:
            return csv_path
        recorded = self.manifest.get("certainty_csv")
        if recorded and os.path.isfile(recorded):
            return recorded
        cands = (glob.glob(os.path.join(run_dir, "certainty_*.csv"))
                 + glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")),
                                          "certainty_*.csv")))
        return max(cands, key=os.path.getmtime) if cands else None

    @staticmethod
    def _open_lanes(run_dir: str, ros2: str) -> Dict[str, Stream]:
        """Open every optional jsonl lane; a lane never recorded is empty."""
        paths = (
            ("reference", os.path.join(run_dir, schema.REFERENCE_FILE)),
            ("control", os.path.join(run_dir, schema.CONTROL_FILE)),
            ("mapping", os.path.join(run_dir, schema.MAPPING_FILE)),
            ("actuator", os.path.join(ros2, schema.ACTUATOR_FILE)),
            ("truth", os.path.join(ros2, schema.TRUTH_FILE)),
            ("altitude", os.path.join(ros2, schema.ALTITUDE_FILE)),
        )
        lanes = OrderedDict((name, Stream.from_file(path, name))
                            for name, path in paths)
        lanes["axis_trace"] = Stream(
            records.group_axis_rows(
                read_jsonl(os.path.join(ros2, schema.AXIS_TRACE_FILE))),
            "axis_trace")
        return lanes

    def _estimate_clocks(self) -> Tuple[ClockOffset, ClockOffset]:
        """Median ``wall - t`` per recorder, over every row it wrote."""
        ros1 = itertools.chain(self.rows, self.event_source.rows,
                               *[self.lanes[n].rows for n in _ROS1_LANES])
        ros2 = itertools.chain(*[self.lanes[n].rows for n in _ROS2_LANES])
        return ClockOffset.estimate(ros1), ClockOffset.estimate(ros2)

    def _check_join(self) -> None:
        """Record why the cross-recorder join might be wrong, if it might be."""
        if not any(len(self.lanes[n]) for n in _ROS2_LANES):
            # A single-recorder run. Say so on a stack that expects two, because
            # every ROS2-fed panel then reads "not recorded" and there is nothing
            # on screen to distinguish that from "the drone was told nothing".
            if self.lanes["control"].rows or self.lanes["reference"].rows:
                self.warnings.append(
                    "no ros2/ lane in this run: the actuator, ground-truth and "
                    "altitude panels have no data. Record it with "
                    "robots/ROBOTICAN/run_nav_debug_recorder.sh")
            return
        self._check_overlap()
        self._check_clocks()

    def _check_overlap(self) -> None:
        """Warn when the ROS2 lanes were recorded during a different flight.

        Pointing ``--ros2`` at the wrong stamp loads thousands of rows that the
        as-of join then rejects one at a time, and the result is indistinguishable
        from having recorded nothing at all -- a blank panel either way.
        """
        span = self._wall_of(0), self._wall_of(len(self.rows) - 1)
        for name in _ROS2_LANES:
            walls = [to_float(r.get("wall")) for r in self.lanes[name].rows
                     if to_float(r.get("wall")) is not None]
            if not walls:
                continue
            gap = max(span[0] - max(walls), min(walls) - span[1])
            if gap > _JOIN_GAP_WARN_S:
                self.warnings.append(
                    "ros2/%s does not overlap this run (%.0f s apart): it is "
                    "almost certainly a different flight" % (name, gap))

    def _check_clocks(self) -> None:
        """Warn when either recorder's wall clock cannot be trusted as a join key."""
        if not self.clock.known:
            self.warnings.append(
                "the ROS1 recording carries no wall clock; ros2/ lanes are joined "
                "as if ROS time were wall time")
        if not self.ros2_clock.known:
            self.warnings.append("ros2/ lanes carry no wall clock; join unreliable")
        for label, clock in (("ROS1", self.clock), ("ROS2", self.ros2_clock)):
            if clock.suspect:
                self.warnings.append("%s %s" % (label, clock.describe()))

    def join_report(self) -> str:
        """A short account of the timeline, both clocks and every lane's size."""
        lines = ["timeline: %s (%d frames)" % (self.timeline_source, len(self.rows)),
                 "ROS1 %s" % self.clock.describe(),
                 "ROS2 %s" % self.ros2_clock.describe(),
                 "lanes: bev=%d routes=%d events=%d " % (
                     len(self.bev_source), len(self.route_source),
                     len(self.event_source))
                 + " ".join("%s=%d" % (n, len(s)) for n, s in self.lanes.items())]
        return "\n".join(lines + ["warning: " + w for w in self.warnings])

    def _build_series(self) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Per-frame tracking error and achieved speed, for the history strips.

        Read straight off the raw lane rows rather than through the dataclasses:
        these two run over the whole timeline, not just the visible frame.

        The speed series is the *measured* one, from whichever source
        :mod:`.state_source` resolved. It used to fall back to the commanded
        velocity when the ground-truth lane was missing, which made the strip
        plot the command against itself and read as perfect tracking on exactly
        the runs where nothing had been measured at all.
        """
        control, truth = self.lanes["control"], self.lanes["truth"]
        err, speed = [], []
        for i, row in enumerate(self.rows):
            control_row = control.at(row["t"], _LANE_MAX_AGE_S)
            tracked = records.section(control_row, "tracking")
            err.append(to_float(tracked.get("position_error_m")) if tracked else None)
            state = self.state_source.at(
                i, control_row,
                truth.at_wall(self._wall_of(i), _LANE_MAX_AGE_S))
            speed.append(None if state is None else state.speed)
        return err, speed

    # ── as-of joins ────────────────────────────────────────────────────────────
    @staticmethod
    def _as_of(stamps: List[float], t: float) -> Optional[int]:
        """Index of the latest stamp <= ``t`` (None if all are later)."""
        return as_of_index(stamps, t)

    def index_at(self, t: float) -> Optional[int]:
        """The frame that was current at time ``t``, for seeking by timestamp."""
        return as_of_index(self._stamps, t)

    def _wall_of(self, i: int) -> float:
        """Host wall clock of frame ``i``: recorded if present, else estimated."""
        wall = self.rows[i].get("wall")
        return wall if wall is not None else self.clock.to_wall(self.rows[i]["t"])

    def _lanes_at(self, i: int, t: float, wall: float) -> dict:
        """Every jsonl lane at this instant: ROS1 by ``t``, ROS2 by ``wall``.

        ``i`` is the frame index, which the measured-state lane needs: its
        last-resort source is the pose spine itself, not a lane row.
        """
        control = self.lanes["control"].at(t, _LANE_MAX_AGE_S)
        truth_row = self.lanes["truth"].at_wall(wall, _LANE_MAX_AGE_S)
        axis_row = self.lanes["axis_trace"].at_wall(wall, _LANE_MAX_AGE_S)
        return {
            "reference": records.reference(
                self.lanes["reference"].at(t, _LANE_MAX_AGE_S)),
            "velocity_target": records.velocity_target(control),
            "velocity_requested": records.velocity_requested(control),
            "velocity_received": records.velocity_received(axis_row),
            "state": self.state_source.at(i, control, truth_row),
            "tracking": records.tracking(control),
            "terms": records.control_terms(control),
            "map_stats": records.map_stats(
                self.lanes["mapping"].at(t, _MAP_MAX_AGE_S)),
            "actuator": records.actuator(
                self.lanes["actuator"].at_wall(wall, _LANE_MAX_AGE_S)),
            "altitude": records.altitude(
                self.lanes["altitude"].at_wall(wall, _LANE_MAX_AGE_S)),
            "truth": records.truth(truth_row),
            "axes": records.axes(axis_row),
        }

    # ── frame assembly ─────────────────────────────────────────────────────────
    def build(self, i: int) -> NavFrame:
        """Assemble the :class:`NavFrame` for frame ``i`` (0-based)."""
        n = len(self.rows)
        if not 0 <= i < n:
            raise IndexError("frame %d out of range [0, %d)" % (i, n))
        r = self.rows[i]
        t = r["t"]
        our = None
        if r.get("vx") is not None or r.get("wz") is not None:
            our = (r.get("vx") or 0.0, r.get("vy") or 0.0,
                   r.get("vz") or 0.0, r.get("wz") or 0.0)

        lanes = self._lanes_at(i, t, self._wall_of(i))
        bev, conf = self.bev_source.at(t)
        return NavFrame(
            stamp=t, x=r["x"], y=r["y"], yaw=r["yaw"], z=r.get("z"),
            trail=self._trail(i), our_cmd=our,
            drone_cmd=_drone_cmd(r, lanes["actuator"]),
            quality=r.get("quality"), drift=r.get("drift"),
            target=self._target(r), advanced=self._advanced(i),
            bev=bev, bev_conf=conf, routes=self.route_source.at(t),
            replan=self.event_source.at(t),
            why=why(r, our, lanes), **self._history(i), **lanes)

    @staticmethod
    def _target(r: dict):
        """The active waypoint as ``(index, count, x, y)``; None if not recorded."""
        if r.get("wp_idx") is None or r.get("tx") is None:
            return None
        return (int(r["wp_idx"]), int(r.get("num_wp") or 0), r["tx"], r["ty"])

    def _advanced(self, i: int) -> bool:
        """True on the tick the active waypoint index grew."""
        if i <= 0:
            return False
        now, before = self.rows[i].get("wp_idx"), self.rows[i - 1].get("wp_idx")
        return bool(now is not None and before is not None and now > before)

    def _trail(self, i: int) -> List[Tuple[float, float]]:
        """The recent pose trail ending just before frame ``i``."""
        lo = max(0, i - _TRAIL_LEN)
        return [(self.rows[k]["x"], self.rows[k]["y"]) for k in range(lo, i)]

    def _history(self, i: int) -> Dict[str, List[float]]:
        """The four trailing strips ending at frame ``i`` (oldest first)."""
        window = range(max(0, i - _HIST_LEN), i + 1)
        return {
            "cmd_history": [self.rows[k].get("vx") or 0.0 for k in window],
            "conf_history": [self.rows[k]["conf"] for k in window
                             if self.rows[k].get("conf") is not None],
            "err_history": [self._err_series[k] for k in window
                            if self._err_series[k] is not None],
            "speed_history": [self._speed_series[k] for k in window
                              if self._speed_series[k] is not None],
        }


def _drone_cmd(row: dict, actuator) -> Optional[Tuple[int, int, int, int]]:
    """The counts the drone was actually sent, as ``(fwd, lat, vert, yaw)``.

    Two stacks fill this from two places, and only the first was ever wired up:

    * **XTEND** writes all four axes into the certainty CSV, which the spine
      carries as ``drone``;
    * **Sphera** has no CSV. The counts exist only as the ``ManualControl`` the
      Rooster command unit publishes, which lives in the ROS2 half of the
      recording -- so on Sphera this lane read ``no cmd_nav`` on every frame of
      every run, whether or not the counts had been recorded.

    ``manual`` is ``[x, y, z, r]`` in the FCU's own axis order, which is exactly
    ``(forward, lateral, vertical, yaw)``; the renderer un-negates the two axes
    the twist adapter inverted so both gauge stacks read the same direction.
    """
    recorded = row.get("drone")
    if recorded is not None:
        return recorded
    manual = getattr(actuator, "manual", None)
    if not manual or len(manual) < 4:
        return None
    return tuple(int(round(to_float(v) or 0.0)) for v in manual[:4])
