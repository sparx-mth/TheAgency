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

#: The FALCON follower gates translation until heading error drops below this.
ALIGN_GATE_DEG = 85.0

#: Measured ManualControl response (rooster_twist_control_adapter defaults):
#: per body axis, the dead-band counts and the m/s produced at 1000 counts.
AXIS_CURVE = {"x": (620.0, 1.25), "y": (700.0, 1.02)}

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
    files = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
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
                mean_s_between_stops=statistics.fmean(gaps) if gaps else None)


def tracking_metrics(lines):
    """Follower heartbeats: position error, heading error, gate, holds."""
    pos, hdg, holding, diverged, ticks, escapes = [], [], 0, 0, 0, 0
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
        holding += fields.get("holding") == "True"
        diverged += fields.get("diverged") == "True"
        escapes = max(escapes, int(_num(fields.get("escapes")) or 0))
    gated = sum(1 for h in hdg if h > ALIGN_GATE_DEG)
    return dict(heartbeats=ticks, pos_err_m=_stats(pos),
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
        dead, v_full = AXIS_CURVE[axis]
        counts, ratios, want_all, got_all, dead_ticks = [], [], [], [], 0
        for sample in samples:
            value = sample[key]
            if value is None:
                continue
            counts.append(abs(value))
            if 0.0 < abs(value) < dead:
                dead_ticks += 1  # motion demanded, none physically produced
            want = max(0.0, (abs(value) - dead) / (1000.0 - dead)) * v_full
            got = _body_speed(sample, axis)
            if want > STOP_SPEED_MPS:
                want_all.append(want)
                if got is not None:
                    got_all.append(abs(got))
                    ratios.append(abs(got) / want)
        out[axis] = dict(ticks=len(counts), dead_band_ticks=dead_ticks,
                         axis_counts=_stats([c for c in counts if c > 0]),
                         commanded_mps=_stats(want_all),
                         achieved_mps=_stats(got_all),
                         achieved_over_commanded=_stats(ratios),
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
    return dict(source=source, target_m=config.TARGET_RANGER_M,
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


def exploration_metrics(lines):
    """FSM transitions, replan verdicts, and whether exploration finished."""
    transitions, replans, plan_fail, finish_t = [], {}, 0, None
    base = next((t for t in (_log_time(ln) for ln in lines)
                 if t is not None), None)
    for line in lines:
        low = line.lower()
        stamp = _log_time(line)
        when = None if None in (stamp, base) else stamp - base
        plan_fail += "plan fail" in low
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
    return dict(fsm_transitions=len(transitions), fsm_lines=transitions[:40],
                finished=finish_t is not None, finish_t_s=finish_t,
                plan_fail=plan_fail, replan_verdicts=dict(
                    sorted(replans.items(), key=lambda kv: -kv[1])[:12]))


# -- ranking and rendering ------------------------------------------------
def _spread(stats, spec="%.2f"):
    """Mean/p90/max of a :func:`_stats` dict as one phrase."""
    return "mean %s, p90 %s, max %s" % (_f(stats["mean"], spec),
                                        _f(stats["p90"], spec),
                                        _f(stats["max"], spec))


def _pct(fraction):
    """A 0-1 fraction as an integer-percent string; ``None`` reads as zero."""
    return _f(100 * (fraction or 0), "%.0f")


def _rank(m):
    """Score candidate next fixes; the highest score is the most urgent."""
    mo, tr, al, ex, he = (m["motion"], m["tracking"], m["altitude"],
                          m["exploration"], m["health"])
    out = []

    def add(score, text):
        """Record a candidate fix unless its evidence is empty/zero."""
        if score:
            out.append((float(score), text))

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
            "Axis %s dead band: %s of %s ticks (%s%%) below %d counts -- "
            "motion demanded, none produced." % (
                axis, d["dead_band_ticks"], d["ticks"],
                _pct(d["frac_dead_band"]), int(AXIS_CURVE[axis][0])))
        gap = abs((ratio["mean"] or 1.0) - 1.0)
        add(3.0 * gap if gap > 0.3 else 0,
            "Axis %s calibration off: achieved/commanded %s over %d ticks -- "
            "flies %s than commanded (%s vs %s m/s mean)." % (
                axis, _spread(ratio), ratio["n"],
                "slower" if (ratio["mean"] or 1) < 1 else "faster",
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


def _render(run_dir, metrics):
    """Build findings.md: ranked actions first, then the supporting numbers."""
    mo = metrics["motion"]
    lines = ["# Findings -- %s" % run_dir.name, "",
             "Drone `%s`, map `%s`, %ss of telemetry (%d samples), %s m flown."
             % (config.DRONE_ID, config.MAP_NAME, _f(mo["duration_s"], "%.0f"),
                metrics["samples_analyzed"], _f(mo["distance_m"], "%.1f")),
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
    records, bad_lines = _load_jsonl(run_dir / "truth.jsonl")
    samples = _normalize(records)
    flying, airborne_only = _airborne_window(_fill_velocity(samples))
    lines, log_files = _read_log_lines(run_dir)
    metrics = dict(
        run=run_dir.name, samples_recorded=len(samples),
        samples_analyzed=len(flying), log_files=log_files,
        motion=motion_metrics(flying, airborne_only),
        actuation=actuation_metrics(flying),
        tracking=tracking_metrics(lines),
        altitude=altitude_metrics(flying, lines),
        health=health_metrics(samples, lines),
        exploration=exploration_metrics(lines))
    metrics["data_gaps"] = _data_gaps(metrics, samples, bad_lines, log_files)
    metrics["ranked_findings"] = _rank(metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "findings.md").write_text(_render(run_dir, metrics))
    return metrics


def newest_run():
    """The most recently modified run directory under ``config.RUNS_DIR``."""
    runs = ([p for p in config.RUNS_DIR.glob("*") if p.is_dir()]
            if config.RUNS_DIR.is_dir() else [])
    if not runs:
        raise NotADirectoryError(
            "no run directories under %s" % config.RUNS_DIR)
    return max(runs, key=lambda p: p.stat().st_mtime)


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
