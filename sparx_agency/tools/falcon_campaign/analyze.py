"""Post-flight analysis: turn one run directory into the next code change.

The campaign loop (MISSION.md section 2) is only as good as its ANALYZE step.
An agent resuming with no memory of the flight cannot watch the drone, so this
module is the only thing standing between a 10-minute flight and the next
decision: it reduces telemetry plus container logs to the metrics MISSION.md
section 6 asks for, then commits to a ranked, quantified list of what to fix.

``metrics.json`` is for machines (trending across runs); ``findings.md`` is for
the agent choosing the next change. Every input is optional -- a flight that
half-crashed still produced the most interesting logs -- so a missing source
becomes a reported data gap and an uncomputable metric stays ``null``, never a
plausible-looking zero.

Log lines are matched by CONTENT, never by log filename: the follower, the
planner FSM and the altitude loop all land in whichever roslaunch/tee log the
harness happened to name, and keying on filenames silently produced empty
tracking metrics. Pure stdlib, so this runs on the 3.12 host venv and inside
the Noetic container.
"""
from __future__ import annotations

import calendar
import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path

try:
    from . import config
except ImportError:  # executed as a plain script, e.g. inside a container
    import config

#: MISSION.md's smoothness definition: a stop is a contiguous run below this
#: speed lasting longer than MIN_STOP_S.
STOP_SPEED_MPS, MIN_STOP_S = 0.05, 0.3

#: Coverage change below this counts as no change, m^3.
COVERAGE_EPS_M3 = 0.5

#: Smallest commanded speed a per-tick achieved/commanded RATIO is computed at.
#:
#: Ratios of small numbers are not evidence. Gating at STOP_SPEED_MPS let a
#: 0.05 m/s demand divide into a 0.8 m/s coast and report 16x, which pushed
#: "axis calibration off, p90 5.14" to the top of a run's ranked findings while
#: the mean achieved speed (0.378) and mean demand (0.380) were in fact within
#: 1%. The platform cannot hold anything below ~0.15 m/s anyway.
RATIO_MIN_COMMANDED_MPS = 0.15

#: The FALCON follower gates translation until heading error drops below this.
ALIGN_GATE_DEG = 85.0

#: The measured plant curve (robots/ROBOTICAN/rooster_axis_curve): counts ->
#: steady-state m/s, one curve for both horizontal axes and every regime. This
#: is the PLANT's truth, not any controller's model, so reading logs back
#: through it is valid for both A/B arms (2026-08-31; it retired the
#: dead-band/moving-pair constants that previously lived here and had to be
#: hand-synced with the adapter).
from sparx_agency.robots.ROBOTICAN.rooster_axis_curve import (
    ROOSTER_HORIZONTAL_CURVE,
)

#: Counts below the curve's first measured level: the speed read-back there is
#: an extrapolation toward zero, not a measurement.
SUB_RESOLUTION_COUNTS = 250.0

STALE_AGE_S = 1.0        # a stream older than this is not flowing
ALT_BAND_M = 0.15        # ranger error that counts as "on target"
ALT_CONVERGED_S = 5.0    # time inside the band before we call it converged

_KV = re.compile(r"(\w+)=(\S+)")
#: Both followers log the same heartbeat shape; config.EXPLORATION_FOLLOWER
#: decides which one flies, so the analyzer must read either.
_HB = re.compile(r"(?:falcon_exploration_follower|rooster_bspline_follower)"
                 r" hb\s+(.*)")
_ALT = re.compile(r"altitude hold: ranger=(-?[\d.]+)m target=(-?[\d.]+)m "
                  r"error=([+-][\d.]+)m")
_LOG_T = re.compile(r"\[\s*(?:INFO|WARN|WARNING|ERROR|DEBUG)[^\]]*\]"
                    r"\s*\[(\d+\.\d+)")

#: Log substrings whose recurrence is itself a health signal. Every needle is
#: verified against the emitting source, not guessed.
_EVENTS = (("velocity_feedback_stale", "velocity feedback stale"),
           ("no_fresh_attitude", "no fresh attitude"),
           ("pose_dropped_tilted", "tilted"),
           ("pinned_escapes", "PINNED"),
           ("altitude_nudges", "altitude target nudged"),
           ("withheld_translation", "withholding translation"),
           ("process_died", "process has died"))


# -- numeric helpers ------------------------------------------------------
def _num(value):
    """``value`` as a finite float, or ``None`` if it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def _through_origin_gain(pairs):
    """Least-squares slope of achieved against commanded, forced through zero.

    The honest single number for "does the platform deliver what was asked".
    Unlike a mean of per-tick ratios it cannot be dominated by ticks where the
    demand was nearly zero, and unlike a mean of speeds it is not fooled by a
    run that spent most of its time near one operating point.

    Args:
        pairs: ``(commanded_mps, achieved_mps)`` samples.

    Returns:
        The slope, or None if there is nothing to fit. 1.0 is perfect.
    """
    num = sum(w * g for w, g in pairs)
    den = sum(w * w for w, g in pairs)
    return (num / den) if den > 0.0 else None


def _stats(values):
    """Summarise samples; every field is ``None`` when there are none."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return dict(n=0, mean=None, median=None, p90=None, min=None, max=None)
    k = (len(vals) - 1) * 0.9
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    p90 = vals[lo] if lo == hi else vals[lo] * (hi - k) + vals[hi] * (k - lo)
    return dict(n=len(vals), mean=statistics.fmean(vals),
                median=statistics.median(vals), p90=p90,
                min=vals[0], max=vals[-1])


def _f(value, spec="%.2f"):
    """Format a possibly-``None`` number for the human-readable report."""
    return "?" if value is None else spec % value


# -- inputs ---------------------------------------------------------------
def _load_jsonl(path):
    """Read JSONL dicts; returns ``(records, unreadable_line_count)``.

    A killed recorder leaves a half-written last line, so unreadable lines are
    tolerated -- but counted, and reported as a data gap rather than dropped.
    """
    records, bad = [], 0
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                bad += line.strip() != ""
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    return records, bad


def _flatten(record):
    """Flatten one recorder row into a lookup dict.

    The recorder nests per stream (``{"truth": {...}, "cmd_nav": {...}}``), so
    each leaf is exposed both bare (first stream wins) and as ``stream.key``.
    """
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub, val in value.items():
                flat.setdefault(sub, val)
                flat[key + "." + sub] = val
        else:
            flat.setdefault(key, value)
    return flat


def _pick(flat, *names):
    """First numeric value among ``names`` present in a flattened record."""
    for name in names:
        out = _num(flat.get(name))
        if out is not None:
            return out
    return None


def _normalize(records):
    """Project raw recorder rows onto the fields the metrics need.

    Pose/velocity/yaw are read from the localization frame first and Sphera's
    raw truth second. The two differ by a rotation of pi about z, which every
    metric here is blind to because all of them use magnitudes.
    """
    samples = []
    for record in records:
        f = _flatten(record)
        samples.append(dict(
            t=_pick(f, "t"),
            x=_pick(f, "localization.x", "truth.x", "x"),
            y=_pick(f, "localization.y", "truth.y", "y"),
            vx=_pick(f, "velocity.vx", "truth.vx", "vx"),
            vy=_pick(f, "velocity.vy", "truth.vy", "vy"),
            yaw=_pick(f, "localization.yaw", "truth.yaw", "attitude.yaw",
                      "yaw"),
            ax=_pick(f, "cmd_nav.x"), ay=_pick(f, "cmd_nav.y"),
            roll=_pick(f, "truth.roll", "attitude.roll"),
            pitch=_pick(f, "truth.pitch", "attitude.pitch"),
            ranger=_pick(f, "state.ranger", "ranger"),
            battery=_pick(f, "state.battery", "battery"),
            airborne=f.get("airborne") is True,
            # "<stream>.age": None means that stream never published at all.
            ages=dict((k[:-4], _num(v)) for k, v in f.items()
                      if k.endswith(".age"))))
    return samples


def _read_log_lines(run_dir):
    """Every line of every ``logs/*.log``, with the file count."""
    log_dir = run_dir / "logs"
    # rosout* duplicates node output that is already here, and rotates
    # mid-flight, so including it counted FSM lines twice by a margin that
    # varied per run. Runs collected before 2026-08-20 still have it on disk.
    files = sorted(p for p in log_dir.glob("*.log")
                   if not p.name.startswith("rosout")) if log_dir.is_dir() else []
    lines = []
    for path in files:
        lines.extend(path.read_text(errors="replace").splitlines())
    return lines, len(files)


def _log_time(line):
    """ROS log timestamp in seconds, or ``None`` if the line carries none."""
    found = _LOG_T.search(line)
    return float(found.group(1)) if found else None


def _rebase(times):
    """Shift timestamps so the first known one is zero."""
    base = next((t for t in times if t is not None), None)
    return [None if (t is None or base is None) else t - base for t in times]


# -- metrics --------------------------------------------------------------
def _airborne_window(samples):
    """Trim to the airborne stretch: sitting on the ground is not a stop."""
    flags = [s["airborne"] for s in samples]
    if True not in flags:
        return samples, False
    first = flags.index(True)
    last = len(flags) - 1 - flags[::-1].index(True)
    return samples[first:last + 1], True


def _times(samples):
    """Run-relative timestamps, falling back to a nominal 10 Hz clock."""
    times = _rebase([s["t"] for s in samples])
    if any(t is None for t in times):
        return [0.1 * i for i in range(len(samples))]
    return times


def _fill_velocity(samples):
    """Backfill vx/vy by finite difference where only pose was recorded.

    Called once, before any metric: otherwise a recorder that lost the
    velocity stream nulls out every speed-derived number. Mutates in place.
    """
    times = _times(samples)
    for idx in range(1, len(samples)):
        a, b = samples[idx - 1], samples[idx]
        if b["vx"] is None and None not in (a["x"], a["y"], b["x"], b["y"]):
            dt = max(1e-3, times[idx] - times[idx - 1])
            b["vx"], b["vy"] = (b["x"] - a["x"]) / dt, (b["y"] - a["y"]) / dt
    return samples


def _timeline(samples):
    """Return (times, dts, speeds) for the run."""
    times = _times(samples)
    speeds = [None if s["vx"] is None or s["vy"] is None
              else math.hypot(s["vx"], s["vy"]) for s in samples]
    dts = [0.0] + [max(0.0, b - a) for a, b in zip(times, times[1:])]
    return times, dts, speeds


def _stop_runs(times, speeds):
    """(start, end) of every below-threshold run longer than the gate."""
    runs, start = [], None
    for idx, speed in enumerate(speeds):
        slow = speed is not None and speed < STOP_SPEED_MPS
        if slow and start is None:
            start = idx
        elif not slow and start is not None:
            runs.append((times[start], times[idx - 1]))
            start = None
    if start is not None:
        runs.append((times[start], times[-1]))
    return [r for r in runs if r[1] - r[0] > MIN_STOP_S]


def motion_metrics(samples, airborne_only):
    """Distance, speed, and the campaign's primary stop metrics."""
    times, dts, speeds = _timeline(samples)
    distance = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                   for a, b in zip(samples, samples[1:])
                   if None not in (a["x"], a["y"], b["x"], b["y"]))
    distance = distance if len(samples) > 1 else None
    total = sum(dts) or None
    slow = sum(dt for dt, sp in zip(dts, speeds)
               if sp is not None and sp < STOP_SPEED_MPS)
    stops = _stop_runs(times, speeds)
    spans = [end - start for start, end in stops]
    gaps = [stops[i][0] - stops[i - 1][1] for i in range(1, len(stops))]
    return dict(airborne_only=airborne_only, duration_s=total,
                distance_m=distance, speed=_stats(speeds),
                frac_time_below_stop_speed=None if not total else slow / total,
                stop_count=len(stops) if samples else None,
                stop_duration=_stats(spans), stop_total_s=sum(spans),
                stops_per_min=(None if not total
                               else len(stops) / (total / 60.0)),
                mean_s_between_stops=statistics.fmean(gaps) if gaps else None,
                per_minute=_per_minute(samples),
                **_turning_effort(samples))


def _per_minute(samples):
    """Distance, spatial span and turning, one row per minute of flight.

    This is the single most-used diagnostic in the campaign: it is what
    separates PARKED from CIRCLING from TRAVELLING, and it identified the
    unreachable-viewpoint lock, the yaw limit cycle, the latched pinned hold and
    the purely-vertical stall — each time by being hand-rolled from truth.jsonl
    after the fact. Recording it with every run means the next collapse is
    already diagnosed when it is noticed.

    Args:
        samples: Normalised pose samples, in time order.

    Returns:
        A list of dicts with ``t`` (window start, s), ``distance_m``,
        ``span_m`` (the widest extent of the window, so a circling aircraft
        reads small) and ``deg_per_m``.
    """
    if not samples:
        return []
    base = samples[0]["t"]
    windows, previous = {}, None
    for sample in samples:
        if None in (sample["x"], sample["y"]):
            continue
        key = int((sample["t"] - base) // 60) * 60
        window = windows.setdefault(key, dict(d=0.0, turn=0.0, xs=[], ys=[]))
        if previous is not None:
            step = math.hypot(sample["x"] - previous["x"], sample["y"] - previous["y"])
            if step < 1.0:                 # a jump that size is a pose glitch
                window["d"] += step
            if None not in (sample.get("yaw"), previous.get("yaw")):
                delta = (sample["yaw"] - previous["yaw"] + math.pi) % (2.0 * math.pi) - math.pi
                window["turn"] += abs(delta)
        window["xs"].append(sample["x"])
        window["ys"].append(sample["y"])
        previous = sample
    rows = []
    for key in sorted(windows):
        w = windows[key]
        span = max(max(w["xs"]) - min(w["xs"]), max(w["ys"]) - min(w["ys"]))
        rows.append(dict(t=key, distance_m=round(w["d"], 1), span_m=round(span, 1),
                         deg_per_m=(round(math.degrees(w["turn"]) / w["d"], 0)
                                    if w["d"] > 0.1 else None)))
    return rows


def _turning_effort(samples):
    """Degrees of yaw spent per metre travelled, whole flight and last third.

    The signature of a yaw limit cycle, and the only metric that separates it
    from healthy flight: an aircraft spinning in place is busy by every rate
    measure this module reports. Measured 2026-08-19, exploring runs 41-70
    deg/m while a stalled one runs 200-270 -- the two do not overlap.

    Args:
        samples: Normalised pose samples, in time order.

    Returns:
        ``turning_deg_per_m`` over the flight and ``turning_deg_per_m_late``
        over its last third, or None where there is too little travel to judge.
    """
    def ratio(rows):
        turn = dist = 0.0
        for a, b in zip(rows, rows[1:]):
            if None in (a["x"], a["y"], b["x"], b["y"], a.get("yaw"), b.get("yaw")):
                continue
            delta = (b["yaw"] - a["yaw"] + math.pi) % (2.0 * math.pi) - math.pi
            turn += abs(delta)
            dist += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        return round(math.degrees(turn) / dist, 1) if dist > 1.0 else None

    return dict(turning_deg_per_m=ratio(samples),
                turning_deg_per_m_late=ratio(samples[2 * len(samples) // 3:]))


def tracking_metrics(lines):
    """Follower heartbeats: position error, heading error, gate, holds."""
    pos, hdg, holding, diverged, ticks, escapes = [], [], 0, 0, 0, 0
    dz = []
    for line in lines:
        found = _HB.search(line)
        if not found:
            continue
        ticks += 1
        fields = dict(_KV.findall(found.group(1)))
        value = _num(fields.get("pos_err", "-").rstrip("m"))
        if value is not None:
            pos.append(value)
        value = _num(fields.get("hdg_err", "-").replace("deg", ""))
        if value is not None:
            hdg.append(abs(value))
        value = _num(fields.get("dz", "-").rstrip("m"))
        if value is not None:
            dz.append(value)
        holding += fields.get("holding") == "True"
        diverged += fields.get("diverged") == "True"
        escapes = max(escapes, int(_num(fields.get("escapes")) or 0))
    gated = sum(1 for h in hdg if h > ALIGN_GATE_DEG)
    return dict(heartbeats=ticks, pos_err_m=_stats(pos),
                # SIGNED, unlike pos_err: a bias means the aircraft is flying a
                # different height from the one being planned for it, which an
                # absolute error hides completely.
                ref_minus_pose_z_m=_stats(dz),
                hdg_err_deg=_stats(hdg), holding_ticks=holding,
                diverged_ticks=diverged, escapes=escapes,
                frac_past_align_gate=(None if not hdg
                                      else gated / float(len(hdg))),
                frac_holding=None if not ticks else holding / float(ticks))


def _body_speed(sample, axis):
    """Body-frame speed along ``axis`` ('x' forward, 'y' lateral)."""
    if None in (sample["vx"], sample["vy"], sample["yaw"]):
        return None
    cos_y, sin_y = math.cos(sample["yaw"]), math.sin(sample["yaw"])
    return (sample["vx"] * cos_y + sample["vy"] * sin_y if axis == "x"
            else -sample["vx"] * sin_y + sample["vy"] * cos_y)


def actuation_metrics(samples):
    """Commanded axis counts vs achieved body speed, per horizontal axis."""
    out = {}
    for axis, key in (("x", "ax"), ("y", "ay")):
        counts, ratios, want_all, got_all, dead_ticks = [], [], [], [], 0
        pairs = []                     # (commanded, achieved), for the slope
        for sample in samples:
            value = sample[key]
            if value is None:
                continue
            counts.append(abs(value))
            got = _body_speed(sample, axis)
            if 0.0 < abs(value) < SUB_RESOLUTION_COUNTS:
                dead_ticks += 1  # below the curve's first measured level
            want = ROOSTER_HORIZONTAL_CURVE.speed_at(abs(value))
            if want > STOP_SPEED_MPS:
                want_all.append(want)
                if got is not None:
                    got_all.append(abs(got))
                    if want >= RATIO_MIN_COMMANDED_MPS:
                        ratios.append(abs(got) / want)
                    pairs.append((want, abs(got)))
        out[axis] = dict(ticks=len(counts), dead_band_ticks=dead_ticks,
                         axis_counts=_stats([c for c in counts if c > 0]),
                         commanded_mps=_stats(want_all),
                         achieved_mps=_stats(got_all),
                         achieved_over_commanded=_stats(ratios),
                         gain=_through_origin_gain(pairs),
                         frac_dead_band=(None if not counts
                                         else dead_ticks / float(len(counts))))
    return out


def altitude_metrics(samples, lines):
    """Ranger tracking, preferring the hold loop's own log over telemetry."""
    logged = [(_ALT.search(ln), _log_time(ln)) for ln in lines]
    logged = [(m, t) for m, t in logged if m]
    errors = [abs(float(m.group(3))) for m, _ in logged]
    rangers = ([float(m.group(1)) for m, _ in logged] if logged else
               [s["ranger"] for s in samples if s["ranger"] is not None])
    source = ("rooster_command_unit log" if logged
              else ("telemetry" if rangers else None))
    times, best, run_start = _rebase([t for _, t in logged]), 0.0, None
    for idx, error in enumerate(errors):
        stamp = times[idx] if times[idx] is not None else idx * 0.1
        run_start = None if error >= ALT_BAND_M else (
            stamp if run_start is None else run_start)
        if run_start is not None:
            best = max(best, stamp - run_start)
    # The hold loop's own logged target, not the configured one: the twist
    # adapter nudges the setpoint in flight, so quoting config here made the
    # report say "0.23 m off a 1.20 m target" while the loop was in fact holding
    # 0.23 m off a target that had drifted to ~1.35 m. The error is real either
    # way; the target it is measured against must be the live one.
    logged_targets = [float(m.group(2)) for m, _ in logged]
    return dict(source=source,
                target_m=(logged_targets[-1] if logged_targets
                          else config.TARGET_RANGER_M),
                target_m_configured=config.TARGET_RANGER_M,
                target_m_drift=(_stats(logged_targets) if logged_targets else None),
                ranger_m=_stats(rangers), abs_error_m=_stats(errors),
                longest_in_band_s=best if errors else None,
                converged=(best >= ALT_CONVERGED_S) if errors else None)


def health_metrics(samples, lines):
    """Stream liveness from ``age``, attitude, battery, and log events."""
    ages, never = {}, set()
    for sample in samples:
        for name, value in sample["ages"].items():
            if value is None:
                never.add(name)
            else:
                ages.setdefault(name, []).append(value)
    stalled = [dict(stream=name, max_age_s=max(vals),
                    stale_samples=sum(1 for v in vals if v > STALE_AGE_S))
               for name, vals in sorted(ages.items())
               if max(vals) > STALE_AGE_S]
    battery = [s["battery"] for s in samples if s["battery"] is not None]
    tilt = [max(abs(s["roll"]), abs(s["pitch"])) for s in samples
            if None not in (s["roll"], s["pitch"])]
    return dict(streams_stalled=stalled,
                streams_never_published=sorted(never - set(ages)),
                stream_max_age_s=dict((k, max(v)) for k, v in ages.items()),
                battery_frac_start=battery[0] if battery else None,
                battery_frac_end=battery[-1] if battery else None,
                max_tilt_rad=_stats(tilt),
                log_events=dict((key, sum(1 for ln in lines if needle in ln))
                                for key, needle in _EVENTS))


def _requested_window(run_dir):
    """The flight window this run was asked to fly, seconds, or None."""
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return getattr(config, "FLIGHT_SECONDS", None)
    return summary.get("duration_requested_s") or getattr(
        config, "FLIGHT_SECONDS", None)


def coverage_metrics(run_dir, flight_s=None):
    """Explored volume over time -- the one metric that IS the mission.

    Everything else this module reports is a proxy: an aircraft can fly
    beautifully smooth circles forever and map nothing. A plateau here is the
    honest signal that exploration has stalled, and it is the number to compare
    across runs when judging whether a change actually helped.

    Args:
        run_dir: The run folder, expected to hold ``coverage.jsonl``.

    Returns:
        Final volume, the fraction of the box it represents, the mean rate, and
        how long the trace spent flat at the end.
    """
    path = run_dir / "coverage.jsonl"
    rows, gaps = [], 0
    window_s = _requested_window(run_dir)
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("coverage_m3") is None:
                # A gap is not missing data, it is data. FALCON stops publishing
                # coverage while the FSM sits in FINISH -- i.e. exactly when
                # nothing is being explored -- so dropping these rows deleted
                # the stalls and then measured the rate over what was left. One
                # run scored 258 m3/min that way, from 185 s of a 437 s flight.
                # Carry the last known volume forward: a gap is a flat interval.
                gaps += 1
                if rows:
                    rows.append(dict(wall=row.get("wall"),
                                     coverage_m3=rows[-1]["coverage_m3"],
                                     filled=True))
                continue
            rows.append(row)
    rows = [r for r in rows if r.get("wall") is not None]
    if len(rows) < 2:
        return dict(samples=len(rows), gaps=gaps, final_m3=None, frac_of_box=None,
                    rate_m3_per_min=None, plateau_s=None, frontier_points=None,
                    span_s=None, reliable=False)

    # Samples keep arriving after the flight window closes, and a tail flown on
    # a flat battery is not exploration -- it dilutes the rate and inflates the
    # stall. Judge the flight the campaign actually asked for.
    if window_s:
        kept = [r for r in rows if r["wall"] - rows[0]["wall"] <= window_s]
        if len(kept) >= 2:
            rows = kept

    span = rows[-1]["wall"] - rows[0]["wall"]
    gained = rows[-1]["coverage_m3"] - rows[0]["coverage_m3"]
    # Trailing plateau: how long the last sample's value has already held.
    plateau_s = 0.0
    final = rows[-1]["coverage_m3"]
    for row in reversed(rows[:-1]):
        if abs(final - row["coverage_m3"]) > COVERAGE_EPS_M3:
            break
        plateau_s = rows[-1]["wall"] - row["wall"]
    return dict(
        samples=len(rows), gaps=gaps, span_s=span,
        # A rate is only comparable across runs when the trace actually spans
        # the flight. Below this the number is reported but must not be ranked
        # or compared -- see the sampler's own comment.
        reliable=(span >= 0.7 * flight_s) if flight_s else None,
        final_m3=final,
        frac_of_box=(final / config.EXPLORABLE_VOLUME_M3
                     if getattr(config, 'EXPLORABLE_VOLUME_M3', 0) else None),
        gained_m3=gained,
        rate_m3_per_min=(gained / span * 60.0) if span > 0 else None,
        plateau_s=plateau_s,
        frontier_points=rows[-1].get("frontier_points"),
        **_stall_metrics(rows))


#: A coverage gain slower than this counts as "not exploring", m3/min.
#: Productive stretches measure 110-480 and a stalled planner dribbles 0-26,
#: so the gap is an order of magnitude wide and the threshold sits in it.
STALL_RATE_M3_PER_MIN = 30.0


def _stall_metrics(rows):
    """How much of the flight produced no coverage, and when it stopped.

    ``plateau_s`` only sees a flat TAIL, and one run ended with a single large
    jump that reset it to zero while the middle of the flight held flat for
    285 s. The mission loses the same time either way, so measure the longest
    barren stretch wherever it falls, plus how long the productive part lasted.

    Args:
        rows: Coverage samples, oldest first, each with ``wall`` and
            ``coverage_m3``.

    Returns:
        ``longest_stall_s`` (longest stretch under
        :data:`STALL_RATE_M3_PER_MIN`), ``stall_frac`` of the sampled span,
        and ``t95_s`` (seconds to reach 95 % of the run's final volume).
    """
    longest = run_s = 0.0
    for prev, cur in zip(rows, rows[1:]):
        dt = cur["wall"] - prev["wall"]
        if dt <= 0:
            continue
        rate = (cur["coverage_m3"] - prev["coverage_m3"]) / dt * 60.0
        run_s = run_s + dt if rate < STALL_RATE_M3_PER_MIN else 0.0
        longest = max(longest, run_s)
    span = rows[-1]["wall"] - rows[0]["wall"]
    target = 0.95 * rows[-1]["coverage_m3"]
    t95 = next((r["wall"] - rows[0]["wall"] for r in rows
                if r["coverage_m3"] >= target), span)
    return dict(longest_stall_s=round(longest, 1),
                stall_frac=round(longest / span, 3) if span > 0 else None,
                t95_s=round(t95, 1))


#: Parameters that define "which configuration was this run flown under".
#: Read from the roslaunch parameter dump, which the supervisor prunes after 30
#: runs -- so they are copied into metrics.json while the log still exists.
#: Without this, a run silently drops out of any within-configuration comparison
#: the moment its logs are pruned, which is exactly when the sample is finally
#: large enough to be worth analysing.
_CONFIG_PARAMS = (
    ("max_vel", r"max_linear_velocity: ([\d.]+)"),
    ("raycast_max", r"raycast_max: ([\d.]+)"),
    ("cluster_min", r"cluster_min: ([\d.]+)"),
    ("bspline_distance", r"pos/distance: ([\d.]+)"),
    ("safe_distance", r"bspline_opt/safe_distance: ([\d.]+)"),
    ("course_slew_deg_s", r"course_slew_deg_s: ([\d.]+)"),
    ("tilt_limit_deg", r"tilt_limit_deg: ([\d.]+)"),
    ("tracker_pos_kp", r"tracker_pos_kp: ([\d.]+)"),
)


def config_metrics(run_dir):
    """The flight configuration, lifted out of the launch log before it is pruned.

    Args:
        run_dir: The run folder.

    Returns:
        A dict of parameter name to value (float), empty if the log is gone.
    """
    log = run_dir / "logs" / "falcon_roslaunch.log"
    if not log.is_file():
        return {}
    found, patterns = {}, list(_CONFIG_PARAMS)
    with log.open(errors="ignore") as handle:
        for line in handle:
            for name, pattern in list(patterns):
                hit = re.search(pattern, line)
                if hit:
                    found[name] = float(hit.group(1))
                    patterns = [p for p in patterns if p[0] != name]
            if not patterns:
                break
    return found


def clearance_metrics(run_dir):
    """Distance from the aircraft AND its reference to the nearest obstacle.

    Half the mission's goal is "minimum collisions", and until this existed
    nothing measured proximity at all -- only the aftermath, as PINNED events.
    Recording both numbers is what separates a plan that goes too close from a
    plan that is merely followed badly; measured 2026-08-20, it was the plan.

    Args:
        run_dir: The run folder, expected to hold ``clearance.jsonl``.

    Returns:
        Stats for each, plus the fraction of samples inside the planner's
        inflation radius.
    """
    path = run_dir / "clearance.jsonl"
    air, ref, err, cross = [], [], [], []
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("nearest_m") is not None:
                air.append(row["nearest_m"])
            if row.get("ref_nearest_m") is not None:
                ref.append(row["ref_nearest_m"])
            if row.get("pos_err_m") is not None:
                err.append(row["pos_err_m"])
            if row.get("cross_m") is not None:
                cross.append(abs(row["cross_m"]))
    inflation = getattr(config, "OBSTACLES_INFLATION", 0.4)
    frac = lambda v: (None if not v
                      else sum(1 for x in v if x < inflation) / float(len(v)))
    return dict(samples=len(air),
                aircraft_m=_stats(air), reference_m=_stats(ref),
                xy_pos_err_m=_stats(err),
                # Only this half of the error can drive the aircraft into a wall
                # its reference is clear of; along-track lag cannot.
                abs_cross_track_m=_stats(cross),
                aircraft_frac_inside_inflation=frac(air),
                reference_frac_inside_inflation=frac(ref))


def exploration_metrics(lines):
    """FSM transitions, replan verdicts, and whether exploration finished."""
    transitions, replans, plan_fail, finish_t = [], {}, 0, None
    reopened, traj_server_exited = 0, False
    base = next((t for t in (_log_time(ln) for ln in lines)
                 if t is not None), None)
    for line in lines:
        low = line.lower()
        stamp = _log_time(line)
        when = None if None in (stamp, base) else stamp - base
        plan_fail += "plan fail" in low
        reopened += "re-opening exploration" in low
        traj_server_exited |= "traj server shutdown" in low
        if "[fsm]" in low:
            transitions.append(
                dict(t=when, text=line[low.index("[fsm]"):].strip()[:160]))
        if finish_t is None and ("finish exploration" in low
                                 or "exploration finished" in low):
            # -1.0 marks "finished, but the line carried no timestamp".
            finish_t = when if when is not None else -1.0
        if "replan" in low:
            key = re.sub(r"[-+]?\d*\.?\d+", "N", line.strip())[-90:]
            replans[key] = replans.get(key, 0) + 1
    result = dict(fsm_transitions=len(transitions), fsm_lines=transitions[:40],
                finished=finish_t is not None, finish_t_s=finish_t,
                reopened=reopened, traj_server_exited=traj_server_exited,
                plan_fail=plan_fail, replan_verdicts=dict(
                    sorted(replans.items(), key=lambda kv: -kv[1])[:12]))
    result.update(_unreachable_viewpoint(lines, base))
    return result


def _unreachable_viewpoint(lines, base):
    """How long the planner stayed locked on a viewpoint A* could not reach.

    Measured 2026-08-19: for 21 315 consecutive planning iterations across five
    minutes the exploration manager chose the SAME viewpoint, A* failed on it
    every time (default profile then coarse), and coverage did not move while
    the aircraft flew 20 m per minute inside a one-metre box. Nothing upstream
    retires a destination for being unreachable -- the coverage tour simply
    offers it again -- so this is the shape a stalled run takes, and it is
    invisible in every other metric here.

    Args:
        lines: Log lines from the run.
        base: Timestamp of the first stamped line, or None.

    Returns:
        ``no_path_fails``, the most-repeated target as ``locked_target``, its
        repeat count, and ``locked_s``: the wall time spanned by the longest
        unbroken run of that same target.
    """
    fails = 0
    targets, times = [], []
    for line in lines:
        if "No path to next viewpoint" in line:
            fails += 1
            continue
        hit = re.search(r"Next pos: (-?[\d.]+), (-?[\d.]+), (-?[\d.]+)", line)
        # Each of these is logged twice, once by glog without a parseable
        # timestamp; keeping only the stamped copy leaves the series intact.
        stamp = _log_time(line) if hit else None
        if hit and stamp is not None:
            targets.append(tuple(round(float(g), 1) for g in hit.groups()))
            times.append(stamp)
    longest, longest_key, count = 0.0, None, 0
    start = run_len = 0
    for i, key in enumerate(targets):
        if i and key == targets[i - 1]:
            run_len += 1
        else:
            start, run_len = i, 1
        if run_len > count:
            count = run_len
            longest_key = key
            first, last = times[start], times[i]
            longest = (last - first) if None not in (first, last) else 0.0
    return dict(no_path_fails=fails,
                locked_target=list(longest_key) if longest_key else None,
                locked_repeats=count, locked_s=round(longest, 1))


# -- ranking and rendering ------------------------------------------------
def _spread(stats, spec="%.2f"):
    """Mean/p90/max of a :func:`_stats` dict as one phrase."""
    return "mean %s, p90 %s, max %s" % (_f(stats["mean"], spec),
                                        _f(stats["p90"], spec),
                                        _f(stats["max"], spec))


def _pct(fraction):
    """A 0-1 fraction as an integer-percent string.

    ``None`` renders as ``?``, not as zero: "0% of the box" is a claim that the
    aircraft mapped nothing, while the truth is that nothing was measured, and
    those two lead to opposite decisions.
    """
    return "?" if fraction is None else _f(100 * fraction, "%.0f")


#: Thresholds for ``collapse_signature``. Derived from the collapses they
#: label, so see that function's note on circularity before trusting them.
PARKED_MAX_M = 2.0
CIRCLING_MAX_SPAN_M = 4.0
CIRCLING_MIN_DIST_M = 5.0
DIVERGED_POS_ERR_M = 5.0
NEVER_STARTED_MAX_M = 60.0
WANDERING_MIN_STALL_S = 120.0
#: Final volume below which a run is called a collapse.
COLLAPSE_M3 = 1300.0


def collapse_signature(metrics):
    """Which known failure shapes this run matches, as a list of tags.

    COLLAPSE_SIGNATURE_TAGS -- a label, NOT a predictor.
    
    Five shapes a bad flight takes, so a run carries its diagnosis instead of
    needing archaeology. Measured over 120 settled runs (2026-08-21):
    
        tag             P(collapse|tag)   lift over the 14% base rate
        NEVER-STARTED       100% (n=1)        7.1x
        DIVERGED             50% (n=2)        3.5x
        PARKED               33% (n=15)       2.4x
        WANDERING            29% (n=7)        2.0x
        CIRCLING             24% (n=42)       1.7x
        any tag              27% (n=60)       1.9x
        NO tag                2% (n=60)       0.1x
    
    Read that table the right way round. A tag is weak evidence: 32 of the 42
    CIRCLING runs finished healthy, so "it collapsed and it was circling" is not
    an explanation -- which is exactly why P38 found no correlation between
    circling minutes and final volume. The informative cell is the last one: an
    untagged run collapsed once in sixty.
    
    And treat even that as provisional. These thresholds were chosen by looking at
    collapses, so their enrichment on those same collapses is partly circular --
    the scan-generates-a-hypothesis problem in structural form. The claim is
    pre-registered in test_p41.py and must be confirmed on fresh runs.

    Args:
        metrics: The metrics dict, needing ``motion``, ``coverage`` and
            ``tracking``.

    Returns:
        list[str]: Matching tags, empty when the flight matches none.
    """
    cov = metrics.get("coverage") or {}
    mo = metrics.get("motion") or {}
    # The trailing row is a partial minute; judging it as a whole one reports a
    # short final sample as a parked minute.
    rows = (mo.get("per_minute") or [])[:-1]
    parked = [r for r in rows if (r.get("distance_m") or 0) < PARKED_MAX_M]
    circling = [r for r in rows
                if (r.get("span_m") or 99.0) < CIRCLING_MAX_SPAN_M
                and (r.get("distance_m") or 0) >= CIRCLING_MIN_DIST_M]
    tags = []
    if parked:
        tags.append("PARKED")
    if circling:
        tags.append("CIRCLING")
    if (((metrics.get("tracking") or {}).get("pos_err_m") or {}).get("mean")
            or 0.0) > DIVERGED_POS_ERR_M:
        tags.append("DIVERGED")
    if (mo.get("distance_m") or 0.0) < NEVER_STARTED_MAX_M:
        tags.append("NEVER-STARTED")
    # FALCON's own planner aborting mid-flight. Measured over 253 runs: with a
    # dead process the median is 1261 and 52 % collapse; without, 1628 and 12 %
    # (Fisher p=1.1e-06). By far the strongest signal in the campaign, and the
    # only tag with an obvious mechanism -- nothing plans new frontiers, so the
    # aircraft flies out its remaining minutes over ground it already mapped.
    if ((metrics.get("health") or {}).get("log_events") or {}).get("process_died"):
        tags.append("PLANNER-DIED")
    # Wandering is the residual: motion healthy in every minute, yet the map
    # stopped growing. Only meaningful once parked and circling are excluded --
    # otherwise it would absorb both.
    if (not parked and not circling
            and (cov.get("longest_stall_s") or 0.0) >= WANDERING_MIN_STALL_S):
        tags.append("WANDERING")
    return tags


def _rank(m):
    """Score candidate next fixes; the highest score is the most urgent."""
    mo, tr, al, ex, he = (m["motion"], m["tracking"], m["altitude"],
                          m["exploration"], m["health"])
    cov = m.get("coverage") or {}
    out = []

    def add(score, text):
        """Record a candidate fix unless its evidence is empty/zero."""
        if score:
            out.append((float(score), text))

    # Ranked above every other finding because it is not a symptom, it is the
    # cause: FALCON's exploration_node aborts and nothing plans after that.
    # Observed signature is a glog CHECK on a negative voxel index, i.e. the
    # aircraft reached a position outside the configured map box.
    died = ((he.get("log_events") or {}).get("process_died") or 0)
    if died:
        add(200.0,
            "PLANNER DIED: FALCON's exploration_node aborted mid-flight (%d "
            "roslaunch 'process has died' events). Everything below is a "
            "CONSEQUENCE -- the aircraft kept flying with no new frontiers. "
            "Root cause seen so far is a glog CHECK on a negative voxel index "
            "(out of map bounds); read the roslaunch log in this run's logs/ "
            "for the abort line. NOTE the live liveness check CANNOT see this: "
            "rosnode list reads the master's registration and ROS1 never "
            "deregisters a crashed node." % died)

    # A parked stretch is the campaign's most expensive single failure and the
    # cheapest to spot: whole minutes with the aircraft going nowhere. Ranked
    # high because a run that parks loses roughly half its coverage, and the
    # per-minute table beside it says which kind of nowhere it was.
    parked = [row for row in (mo.get("per_minute") or [])[:-1]
              if row["distance_m"] < 2.0]
    if parked:
        add(60.0 + len(parked),
            "PARKED for %d whole minute(s) (from t=%s): distance %s m with span "
            "%s m -- not circling, not travelling. Check the commanded axis "
            "first: zero means nothing was asked of it (a reference it cannot "
            "reach, e.g. purely vertical), non-zero means it is wedged."
            % (len(parked), ", ".join(str(r["t"]) for r in parked),
               "/".join("%.1f" % r["distance_m"] for r in parked),
               "/".join("%.1f" % r["span_m"] for r in parked)))

    # The other half of "went nowhere": still flying, but round in circles. It
    # costs as much as parking and looks healthy in every rate metric —
    # distance, speed and stop counts all read normal while the aircraft orbits
    # a few metres of already-mapped floor.
    circling = [row for row in (mo.get("per_minute") or [])[:-1]
                if row["span_m"] < 4.0 and row["distance_m"] >= 5.0]
    if circling:
        add(55.0 + len(circling),
            "CIRCLING for %d whole minute(s) (from t=%s): flew %s m inside a "
            "%s m box at %s deg/m. Distance and speed look normal; the SPAN is "
            "what says it went nowhere. Check turning per metre against the "
            "41-70 healthy band, then whether the reference is orbiting it."
            % (len(circling), ", ".join(str(r["t"]) for r in circling),
               "/".join("%.0f" % r["distance_m"] for r in circling),
               "/".join("%.1f" % r["span_m"] for r in circling),
               "/".join(str(r["deg_per_m"]) for r in circling)))

    # A reference the aircraft is not merely lagging but is nowhere near. The
    # ordinary spread is 0.83 m median, 1.75 m p90 across 115 settled runs, and
    # tracking error in THAT range was tested and closed as a collapse predictor
    # (+0.20, n=15, pre-registered). This flag is for the other regime: two runs
    # have sat above 5 m mean -- 17.6 m on 064835Z, median 23 m, i.e. more than
    # half the flight spent ~23 m from the reference -- which is not lag, it is
    # a reference somewhere else entirely. Too rare to correlate (2 of 115), so
    # this LABELS it rather than claiming it causes anything; without the label
    # the next occurrence costs the same archaeology this one did.
    pe_mean = ((tr.get("pos_err_m") or {}).get("mean")) or 0.0
    if pe_mean > 5.0:
        pe = tr["pos_err_m"]
        add(50.0 + pe_mean,
            "REFERENCE DIVERGENCE: pos_err mean %.1f m (median %.1f, p90 %.1f, "
            "max %.1f) against a 0.83 m corpus median -- the aircraft was not "
            "tracking this reference, it was chasing one somewhere else. Read "
            "the per-minute span table for a long transit minute, and check "
            "whether the divergence PRECEDED the coverage stall or followed it."
            % (pe_mean, pe.get("median") or -1, pe.get("p90") or -1,
               pe.get("max") or -1))

    add(mo["stops_per_min"],
        "P1 stop/go stutter: %s stops in %ss (%s/min); stop duration %s s, "
        "%ss stopped in total; %s%% of flight below %s m/s; %ss moving "
        "between stops." % (
            mo["stop_count"], _f(mo["duration_s"], "%.0f"),
            _f(mo["stops_per_min"], "%.1f"), _spread(mo["stop_duration"]),
            _f(mo["stop_total_s"], "%.0f"),
            _pct(mo["frac_time_below_stop_speed"]), STOP_SPEED_MPS,
            _f(mo["mean_s_between_stops"], "%.1f")))
    for axis, d in sorted(m["actuation"].items()):
        ratio = d["achieved_over_commanded"]
        add(6.0 * (d["frac_dead_band"] or 0),
            "Axis %s sub-resolution: %s of %s ticks (%s%%) below %d counts -- "
            "commanded slower than the first measured calibration level." % (
                axis, d["dead_band_ticks"], d["ticks"],
                _pct(d["frac_dead_band"]), int(SUB_RESOLUTION_COUNTS)))
        gain = d.get("gain")
        gap = abs((gain if gain is not None else 1.0) - 1.0)
        add(3.0 * gap if gap > 0.25 else 0,
            "Axis %s calibration off: delivers %sx what is commanded "
            "(through-origin gain over %d ticks; per-tick ratio %s) -- "
            "flies %s than commanded (%s vs %s m/s mean)." % (
                axis, _f(gain), ratio["n"], _spread(ratio),
                "slower" if (gain or 1) < 1 else "faster",
                _f(d["commanded_mps"]["mean"]), _f(d["achieved_mps"]["mean"])))
    p90 = tr["pos_err_m"]["p90"] or 0
    add(2.0 * p90 if p90 > 0.8 else 0,
        "Tracking error: pos_err %s m over %d heartbeats." % (
            _spread(tr["pos_err_m"]), tr["heartbeats"]))
    gate = tr["frac_past_align_gate"] or 0
    add(4.0 * gate if gate > 0.15 else 0,
        "Yaw align gate blocks translation on %s%% of ticks (|hdg_err| > %d "
        "deg; %s) -- a direct source of stop/go." % (
            _pct(gate), int(ALIGN_GATE_DEG),
            _spread(tr["hdg_err_deg"], "%.0f")))
    add(2.0 * (tr["frac_holding"] or 0),
        "Follower held station on %d ticks (%s%%), diverged on %d." % (
            tr["holding_ticks"], _pct(tr["frac_holding"]),
            tr["diverged_ticks"]))
    add(1.0 + tr["escapes"] / 10.0 if tr["escapes"] else 0,
        "%d PINNED escapes: commanded but not moving." % tr["escapes"])
    add(3.0 if al["converged"] is False else 0,
        "Altitude never converged: |error| %s m against target %s m; longest "
        "in-band run %ss; ranger %s-%s m." % (
            _spread(al["abs_error_m"]), _f(al["target_m"]),
            _f(al["longest_in_band_s"], "%.1f"), _f(al["ranger_m"]["min"]),
            _f(al["ranger_m"]["max"])))
    add(min(4.0, ex["plan_fail"] / 50.0),
        "Planner logged 'plan fail' %d times across %d FSM lines." % (
            ex["plan_fail"], ex["fsm_transitions"]))
    add(1.5 if (ex["fsm_transitions"] and not ex["finished"]) else 0,
        "Exploration never reached FINISH in %ss (%d FSM transitions)." % (
            _f(mo["duration_s"], "%.0f"), ex["fsm_transitions"]))
    # Coverage is the mission. A long plateau outranks everything except a dead
    # traj_server, because a smooth, well-tracked flight that maps nothing is a
    # failure that every other metric in this file will happily call a success.
    reliable = cov.get("reliable")
    add(3.0 if reliable is False and cov.get("samples") else 0,
        "Coverage trace covers only %ss of a %ss flight (%s samples, %s gaps) -- "
        "its rate of %s m3/min is NOT comparable with other runs. "
        "/voxel_mapping/map_coverage goes quiet while the FSM sits in FINISH, so "
        "a run that re-opens repeatedly samples sparsely."
        % (_f(cov.get("span_s"), "%.0f"), _f(mo.get("duration_s"), "%.0f"),
           cov.get("samples"), cov.get("gaps"),
           _f(cov.get("rate_m3_per_min"), "%.1f")))
    plateau = (cov.get("plateau_s") or 0.0) if reliable else 0.0
    add(min(8.0, plateau / 60.0) if plateau >= 90.0 else 0,
        "Coverage PLATEAUED for the last %ss at %s m3 (%s of the box) with %s "
        "frontier points still reported -- the aircraft is flying but no longer "
        "mapping." % (_f(plateau, "%.0f"), _f(cov.get("final_m3"), "%.0f"),
                      _pct(cov.get("frac_of_box")), cov.get("frontier_points")))
    rate = cov.get("rate_m3_per_min") if reliable else None
    add(2.0 if (rate is not None and rate < 5.0 and plateau < 90.0) else 0,
        "Coverage rate is only %s m3/min (%s m3 total, %s of the box) -- the "
        "flight is smooth but slow to explore."
        % (_f(rate, "%.1f"), _f(cov.get("final_m3"), "%.0f"),
           _pct(cov.get("frac_of_box"))))
    add(9.0 if ex.get("traj_server_exited") else 0,
        "traj_server SHUT DOWN mid-flight: every later tick has no reference, so "
        "the follower holds station for the rest of the run. Check that "
        "/traj_server/exit_on_finish is false and that the image carries "
        "fix_falcon_finish_reopen.sh.")
    add(4.0 if (ex["finished"] and not ex.get("reopened")) else 0,
        "Exploration finished at t=%ss and never re-opened -- the rest of the "
        "flight was station-keeping. Expected 'Re-opening exploration' in the "
        "FSM log." % _f(ex.get("finish_t_s"), "%.0f"))
    for stream in he["streams_stalled"]:
        add(3.5, "Stream '%s' went stale mid-flight: max age %ss over %d "
                 "samples." % (stream["stream"],
                               _f(stream["max_age_s"], "%.1f"),
                               stream["stale_samples"]))
    for stream in he["streams_never_published"]:
        add(4.0, "Stream '%s' never published: dead for the whole flight."
            % stream)
    for key, count in sorted(he["log_events"].items(), key=lambda kv: -kv[1]):
        add(min(2.5, count / 100.0), "%d '%s' log events." % (count, key))
    out.sort(key=lambda item: -item[0])
    return [text for _, text in out]


def _triage_line(metrics):
    """One line saying whether this run needs reading at all.

    Wording is bounded by what P41 actually established on fresh data (0 of 25
    untagged runs collapsed, against 7 of 16 tagged, Fisher p=0.0005): the
    ABSENCE of a signature predicts health, while a tag does NOT explain a
    collapse -- 56 % of tags sit on healthy flights. So an untagged run is
    called almost certainly fine, and a tagged one is only called worth
    reading. Neither is called a diagnosis.
    """
    tags = metrics.get("collapse_signature")
    if tags is None:
        return "**Triage: not available** -- this run predates the signature classifier."
    volume = (metrics.get("coverage") or {}).get("final_m3")
    collapsed = volume is not None and volume < COLLAPSE_M3
    if not tags:
        return ("**Triage: CLEAN** -- no failure signature. Untagged runs collapsed "
                "0 times in 25 on fresh data, so this one is almost certainly fine%s."
                % (" -- but it DID land low, which makes it worth a look"
                   if collapsed else ""))
    return ("**Triage: %s** -- worth reading. A tag is not a diagnosis: 56 %% of "
            "tagged runs finish healthy, so read the per-minute table before "
            "concluding anything." % "+".join(tags))


def _render(run_dir, metrics):
    """Build findings.md: ranked actions first, then the supporting numbers."""
    mo = metrics["motion"]
    lines = ["# Findings -- %s" % run_dir.name, "",
             "Drone `%s`, map `%s`, %ss of telemetry (%d samples), %s m flown."
             % (config.DRONE_ID, config.MAP_NAME, _f(mo["duration_s"], "%.0f"),
                metrics["samples_analyzed"], _f(mo["distance_m"], "%.1f")),
             "",
             "**Coverage: %s m3 (%s of the box), %s m3/min.** This is the "
             "mission; everything below is a proxy for it."
             % (_f((metrics.get("coverage") or {}).get("final_m3"), "%.0f"),
                _pct((metrics.get("coverage") or {}).get("frac_of_box")),
                _f((metrics.get("coverage") or {}).get("rate_m3_per_min"), "%.1f")),
             "", _triage_line(metrics),
             "", "## Fix next (ranked)", ""]
    lines += ["%d. %s" % (i, t)
              for i, t in enumerate(metrics["ranked_findings"], 1)] or [
        "_Nothing scored -- read the data gaps below before trusting that._"]
    lines += ["", "## Metrics", ""]
    for name in ("motion", "tracking", "actuation", "altitude", "health",
                 "exploration"):
        lines += ["### " + name, "```json",
                  json.dumps(metrics[name], indent=2, sort_keys=True,
                             default=str), "```", ""]
    return "\n".join(lines + ["## Data gaps", ""] +
                     (["- " + g for g in metrics["data_gaps"]] or
                      ["- none"])) + "\n"


# -- entry points ---------------------------------------------------------
def _data_gaps(metrics, samples, bad_lines, log_files):
    """Human-readable list of inputs that were missing or unusable."""
    axes = any(s["ax"] is not None or s["ay"] is not None for s in samples)
    checks = (
        (not samples, "truth.jsonl missing or empty: motion, actuation and "
                      "health metrics are null."),
        (bool(samples) and not axes,
         "telemetry carries no cmd_nav axes: actuation metrics are null."),
        (bool(bad_lines),
         "%d unreadable truth.jsonl line(s) skipped." % bad_lines),
        (not metrics["motion"]["airborne_only"],
         "no airborne flag in telemetry: motion metrics include any time "
         "spent on the ground."),
        (not log_files, "no logs/*.log: tracking, altitude and exploration "
                        "metrics are null."),
        (bool(log_files) and not metrics["tracking"]["heartbeats"],
         "no follower heartbeat lines in %d log file(s): tracking metrics "
         "are null." % log_files),
        (bool(log_files)
         and metrics["altitude"]["source"] != "rooster_command_unit log",
         "no altitude-hold log lines: altitude falls back to the telemetry "
         "ranger, with no target error."),
        (bool(log_files) and not metrics["exploration"]["fsm_transitions"],
         "no [FSM] lines: exploration progress is unknown."))
    return [text for failed, text in checks if failed]


def planner_death(run_dir):
    """When FALCON's planner aborted this run, and why, if it did.

    P42 makes this the campaign's most valuable fact, and pruning used to
    delete it -- so it is lifted into ``metrics.json`` where it survives.
    Two files, verified 2026-08-22 against two preserved runs:

    - the death TIMESTAMP is in ``logs/roslaunch-*.log`` ("process has died"),
    - the REASON, when there is one, is a glog line in
      ``logs/falcon_roslaunch.log`` (roslaunch's stdout capture).

    The reason is often absent. Three of five known cases died ``exit code -6``
    with no glog output anywhere, while two carried
    ``Check failed: addr < map_data_->data.size()`` -- a negative voxel index
    out of ``voxel_mapping::MapBase<>::getVoxel()``. So there are at least two
    abort modes and ``reason`` being None is a real observation, not a gap.

    Args:
        run_dir: The run folder.

    Returns:
        dict with ``died`` (bool), ``wall`` (ISO timestamp or None) and
        ``reason`` (the glog line or None).
    """
    out = {"died": False, "wall": None, "reason": None}
    for path in sorted(run_dir.glob("logs/roslaunch-*.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        match = re.search(r"\[([\d-]+ [\d:]+),\d+\]?:? \[exploration_node[^\]]*\] "
                          r"process has died", text)
        if not match:
            match = re.search(r"([\d-]{10} [\d:]{8}),\d+: \[exploration_node[^\]]*\] "
                              r"process has died", text)
        if match:
            out["died"] = True
            out["wall"] = match.group(1)
            break
    reason_path = run_dir / "logs" / "falcon_roslaunch.log"
    if out["died"] and reason_path.is_file():
        try:
            for line in reason_path.read_text(errors="replace").splitlines():
                if line.startswith("F") and "Check failed" in line:
                    out["reason"] = line.strip()[:300]
                    break
        except OSError:
            pass
    return out


def _epoch_of(wall):
    """Epoch seconds for a roslaunch UTC wall-clock string, or None."""
    if not wall:
        return None
    try:
        moment = datetime.datetime.strptime(wall, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return calendar.timegm(moment.timetuple())


def _save_planner_death_excerpt(run_dir, death, before_s=120.0, after_s=20.0):
    """Copy the log window AROUND a planner abort out of ``logs/``.

    Selected by TIMESTAMP, not by line position. Two earlier versions sliced
    lines around an anchor and both failed, because ``falcon_roslaunch.log`` is
    the interleaved stdout of a dozen nodes with independent buffering, so line
    order is not time order. Measured on `182508Z` (died 18:30:35): the tail
    version put 407 of 413 lines after the abort, and the anchored version
    still put 432 of 441 after it. Filtering on the epoch stamp each line
    carries is the only thing that actually selects the run-up.

    The supervisor also exempts planner-death runs from pruning, but that edit
    is inert until the supervisor restarts (see LESSONS.md). This runs
    in-process every cycle, and pruning deletes ``logs/`` and nothing else, so
    an excerpt written beside it survives either way.

    Args:
        run_dir: The run folder.
        death: The ``planner_death`` block.
        before_s: Seconds of run-up to keep.
        after_s: Seconds after the abort, enough to show the aftermath begin.
    """
    if not death.get("died"):
        return
    source = run_dir / "logs" / "falcon_roslaunch.log"
    if not source.is_file():
        return
    try:
        lines = source.read_text(errors="replace").splitlines()
    except OSError:
        return
    target = _epoch_of(death.get("wall"))
    kept, stamped = [], 0
    for line in lines:
        match = re.search(r"\[(\d{10}\.\d+)\]", line)
        if not match:
            continue
        stamped += 1
        when = float(match.group(1))
        if target is None:
            continue
        if target - before_s <= when <= target + after_s:
            kept.append((when, line))
    if target is None or not kept:
        # No usable timestamps: fall back to the tail rather than nothing, and
        # say so in the header so nobody reads it as run-up.
        kept = [(0.0, l) for l in lines[-200:]]
        note = "FALLBACK: no epoch stamps matched; this is the TAIL, all aftermath"
    else:
        kept.sort()
        note = ("%d lines within -%.0fs/+%.0fs of the abort, out of %d stamped"
                % (len(kept), before_s, after_s, stamped))
    header = ["# Planner abort context, kept outside logs/ so pruning cannot",
              "# remove it. died at: %s" % death.get("wall"),
              "# reason: %s" % (death.get("reason") or
                                "NONE (exit -6, no glog output)"),
              "# %s" % note,
              "# Selected by TIMESTAMP: this log interleaves many nodes, so line",
              "# order is not time order and positional slicing does not work.", ""]
    try:
        (run_dir / "planner_death_context.log").write_text(
            "\n".join(header + [l for _, l in kept]) + "\n")
    except OSError:
        pass


def analyze(run_dir):
    """Analyse one run directory, writing ``metrics.json`` and ``findings.md``.

    Args:
        run_dir: A run directory produced by the campaign harness.

    Returns:
        dict: The metrics, including the ranked findings list.

    Raises:
        NotADirectoryError: If ``run_dir`` does not exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError("no such run directory: %s" % run_dir)
    # A pruned run keeps its metrics.json but not its logs. Re-analysing one
    # would silently overwrite good numbers with zeros for everything log-derived
    # -- tracking, exploration, half the health signals -- and nothing downstream
    # could tell the difference afterwards.
    log_dir = run_dir / "logs"
    has_logs = log_dir.is_dir() and any(log_dir.glob("*.log"))
    if (run_dir / "metrics.json").exists() and not has_logs:
        raise RuntimeError(
            "%s has metrics but no logs (pruned); refusing to overwrite them "
            "with a log-less analysis" % run_dir.name)
    records, bad_lines = _load_jsonl(run_dir / "truth.jsonl")
    samples = _normalize(records)
    flying, airborne_only = _airborne_window(_fill_velocity(samples))
    lines, log_files = _read_log_lines(run_dir)
    motion = motion_metrics(flying, airborne_only)
    metrics = dict(
        run=run_dir.name, samples_recorded=len(samples),
        samples_analyzed=len(flying), log_files=log_files,
        motion=motion,
        actuation=actuation_metrics(flying),
        tracking=tracking_metrics(lines),
        altitude=altitude_metrics(flying, lines),
        coverage=coverage_metrics(run_dir, motion.get('duration_s')),
        health=health_metrics(samples, lines),
        exploration=exploration_metrics(lines),
        clearance=clearance_metrics(run_dir),
        config=config_metrics(run_dir))
    metrics["planner_death"] = planner_death(run_dir)
    _save_planner_death_excerpt(run_dir, metrics["planner_death"])
    metrics["collapse_signature"] = collapse_signature(metrics)
    metrics["data_gaps"] = _data_gaps(metrics, samples, bad_lines, log_files)
    metrics["ranked_findings"] = _rank(metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "findings.md").write_text(_render(run_dir, metrics))
    return metrics


def newest_run():
    """The latest run directory under ``config.RUNS_DIR``, by its timestamp name.

    Not by mtime: the supervisor prunes logs from old runs, and deleting a
    subdirectory bumps the parent's mtime, so an hours-old run that was just
    pruned sorts as the newest one. Run directories are named for their start
    time, so lexicographic order IS chronological order and cannot be disturbed
    by anything touching the files afterwards.
    """
    # Only flight runs: the folder also holds axiscal_* sweeps, and "a" sorts
    # after "2", so an unfiltered name-max picks a calibration folder.
    stamped = re.compile(r"^\d{8}_\d{6}Z$")
    runs = ([p for p in config.RUNS_DIR.glob("*")
             if p.is_dir() and stamped.match(p.name)]
            if config.RUNS_DIR.is_dir() else [])
    if not runs:
        raise NotADirectoryError(
            "no run directories under %s" % config.RUNS_DIR)
    return max(runs, key=lambda p: p.name)


def main(argv=None):
    """CLI: ``analyze [run_dir]``, defaulting to the newest run."""
    argv = sys.argv[1:] if argv is None else argv
    run_dir = Path(argv[0]) if argv else newest_run()
    metrics = analyze(run_dir)
    print("%s: %d samples, %s stops -> %s" % (
        run_dir.name, metrics["samples_analyzed"],
        metrics["motion"]["stop_count"], run_dir / "findings.md"))
    for index, text in enumerate(metrics["ranked_findings"][:5], 1):
        print("  %d. %s" % (index, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
