#!/usr/bin/env python3
"""Fit the Rooster's ManualControl axis response from a calibration flight.

The flight half lives in :mod:`rooster_axis_calibration`; this is the half that
turns its two files into numbers somebody can paste into a config. It is kept
separate because it must run without docker, without ROS and without a drone --
a fit that can only be re-run by flying again cannot be improved after the fact,
and every number in MISSION.md P5 was paid for in flight time.

Two rules decide what is even readable here:

* **Only the last two seconds of each hold count.** The aircraft accelerates for
  most of a 3-5 s step, so averaging the whole hold measures the transient, not
  the gain -- that is one of the two reasons the 2026-08-18 curve under-delivered
  by 3x in flight.
* **Velocity comes from ``/R1/velocity_truth``, never from the truth message's
  own ``velocity`` field**, which is declared, documented in m/s, and all-zero in
  this Sphera build (LESSONS.md 2026-08-18). Body frame is recovered with the
  yaw the whole stack consumes, from ``/R1/localization``.

Pure stdlib on purpose: it is imported by the flight tool inside whatever
interpreter happens to be available. Python 3.8-compatible.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics

#: Only samples this close to the end of a hold are treated as settled.
SETTLE_WINDOW_S = 2.0

#: A feedback stream older than this is not evidence of anything.
MAX_FEEDBACK_AGE_S = 1.0

#: Below these the axis is inside its dead band and carries no gain information.
MOTION_EPS_MPS = 0.05
MOTION_EPS_RADPS = 0.05

FULL_SCALE_AXIS = 1000.0

#: Which body-frame response each axis is supposed to produce.
AXIS_FIELD = {"x": "fwd", "y": "lat", "r": "yaw_rate"}

TRUTH_FILE = "truth.jsonl"
SEGMENTS_FILE = "segments.jsonl"


def load_jsonl(path):
    """Read a JSONL file, skipping any torn trailing line.

    A recorder killed by a dead battery leaves half a line behind, and that is
    exactly the run this fit most needs to read.

    Args:
        path: File to read.

    Returns:
        List of decoded objects.

    Raises:
        FileNotFoundError: If the file does not exist -- an empty result would
            look identical to a flight that produced no data.
    """
    rows = []
    with open(str(path)) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _body_frame(row):
    """Body-frame ``(forward, lateral, yaw_rate)`` from one recorder sample.

    Returns:
        The triple, or ``None`` when either feedback stream is stale or missing
        -- a stale stream must never be averaged in as a real zero.
    """
    velocity = row.get("velocity") or {}
    localization = row.get("localization") or {}
    for stream in (velocity, localization):
        age = stream.get("age")
        if age is None or age > MAX_FEEDBACK_AGE_S:
            return None
    vx, vy = velocity.get("vx"), velocity.get("vy")
    yaw, yaw_rate = localization.get("yaw"), velocity.get("yaw_rate")
    if vx is None or vy is None or yaw is None or yaw_rate is None:
        return None
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return (vx * cos_yaw + vy * sin_yaw, -vx * sin_yaw + vy * cos_yaw, yaw_rate)


def segment_response(rows, segment):
    """Mean settled response of one segment, or ``None`` if it has no samples.

    Args:
        rows: Every recorder sample of the run.
        segment: One record from ``segments.jsonl``.

    Returns:
        Dict with ``n``, ``fwd``, ``lat``, ``yaw_rate`` and ``speed``.
    """
    end = segment.get("t_end")
    if end is None:
        return None
    start = end - SETTLE_WINDOW_S
    forward, lateral, rates = [], [], []
    for row in rows:
        wall = row.get("wall")
        if wall is None or wall < start or wall > end:
            continue
        body = _body_frame(row)
        if body is None:
            continue
        forward.append(body[0])
        lateral.append(body[1])
        rates.append(body[2])
    if not forward:
        return None
    fwd, lat = statistics.mean(forward), statistics.mean(lateral)
    return {"n": len(forward), "fwd": fwd, "lat": lat,
            "yaw_rate": statistics.mean(rates), "speed": math.hypot(fwd, lat)}


def single_axis(axes):
    """Return ``(key, magnitude)`` for a one-axis command, else ``None``.

    ``key`` is the signed axis name the whole report is indexed by: ``x+``,
    ``y-``, ``r+`` and so on, because the two signs are separate actuators
    until measurement says otherwise.
    """
    live = [(name, value) for name, value in axes.items() if abs(value) > 0.0]
    if len(live) != 1:
        return None
    name, value = live[0]
    return name + ("+" if value > 0 else "-"), abs(value)


def _eps(key):
    """Motion threshold for an axis key, in that axis's own units."""
    return MOTION_EPS_RADPS if key.startswith("r") else MOTION_EPS_MPS


def _linfit(points):
    """Least-squares ``(slope, intercept)`` of y on x, or ``None``."""
    xs = [x for x, _ in points]
    if len(points) < 2 or len(set(xs)) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean([y for _, y in points])
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    return slope, mean_y - slope * mean_x


def _curve(points):
    """Summarise one axis fit as slope, dead-band intercept and full scale.

    ``deadzone`` is the axis value where the fitted line crosses zero speed --
    the number ``velocity_to_axis`` needs -- and ``full_scale`` is the speed the
    same line predicts at axis 1000.
    """
    fit = _linfit(points)
    if fit is None:
        return None
    slope, intercept = fit
    summary = {"slope": slope, "intercept": intercept, "n": len(points),
               "deadzone": None, "full_scale": None}
    if slope > 0.0:
        summary["deadzone"] = round(-intercept / slope, 1)
        summary["full_scale"] = round(slope * FULL_SCALE_AXIS + intercept, 4)
    return summary


def _measured(rows, segments):
    """Attach a settled response to every usable segment."""
    out = []
    for segment in segments:
        label = segment.get("label", "")
        if segment.get("aborted") or label.endswith("/preload"):
            continue
        response = segment_response(rows, segment)
        if response is None:
            continue
        entry = dict(segment)
        entry.update(response)
        out.append(entry)
    return out


def breakaway(measured):
    """Lowest block-(i) axis value that produced motion from a standstill."""
    lowest = {}
    for entry in measured:
        if entry.get("block") != "i":
            continue
        pair = single_axis(entry["axes"])
        if pair is None:
            continue
        key, magnitude = pair
        if abs(entry[AXIS_FIELD[key[0]]]) < _eps(key):
            continue
        if key not in lowest or magnitude < lowest[key]:
            lowest[key] = magnitude
    return lowest


def single_axis_curves(measured):
    """Per-signed-axis gain fitted over every isolated single-axis hold."""
    points = {}
    for entry in measured:
        if entry.get("block") not in ("i", "iii"):
            continue
        pair = single_axis(entry["axes"])
        if pair is None:
            continue
        key, magnitude = pair
        value = abs(entry[AXIS_FIELD[key[0]]])
        if value < _eps(key):
            continue                      # inside the dead band: no gain here
        points.setdefault(key, []).append((magnitude, value))
    return dict((key, _curve(pts)) for key, pts in points.items())


def moving_curves(measured):
    """Forward gain while already moving, one fit per block-(ii) approach."""
    points = {}
    for entry in measured:
        if entry.get("block") != "ii":
            continue
        approach = entry["label"].split("/")[1]
        value = abs(entry["fwd"])
        if value < MOTION_EPS_MPS:
            continue
        points.setdefault(approach, []).append((abs(entry["axes"]["x"]), value))
    return dict((key, _curve(pts)) for key, pts in points.items())


def hysteresis(measured):
    """Speed gap between the two block-(ii) approaches at the same axis value.

    Positive means the aircraft is faster when it arrived from the 850 pre-load
    than from the 650 one -- i.e. the standing-vs-moving gap the follower has to
    pay for every time it stops.
    """
    by_approach = {}
    for entry in measured:
        if entry.get("block") != "ii":
            continue
        parts = entry["label"].split("/")
        by_approach.setdefault(parts[1], {})[parts[2]] = entry["fwd"]
    down, up = by_approach.get("down", {}), by_approach.get("up", {})
    gaps = dict((value, round(down[value] - up[value], 4))
                for value in sorted(set(down) & set(up), key=float))
    mean_gap = round(statistics.mean(list(gaps.values())), 4) if gaps else None
    return {"per_axis_value": gaps, "mean_gap_mps": mean_gap}


def _predict(curves, key, magnitude):
    """Speed a single-axis curve predicts for one axis value."""
    curve = curves.get(key)
    if not curve or curve.get("slope") is None or curve["slope"] <= 0.0:
        return None
    return max(0.0, curve["slope"] * magnitude + curve["intercept"])


def combined(measured, curves):
    """Achieved-vs-predicted speed for every multi-axis block-(iii) hold.

    A ratio above 1 means the combination goes faster than the single-axis
    curves say it should, so each axis must be commanded LESS when they are
    used together -- which is exactly the cross-axis compensation MISSION.md P5
    says does not exist yet.
    """
    rows, groups = [], {}
    for entry in measured:
        if entry.get("block") != "iii" or single_axis(entry["axes"]):
            continue
        predicted, usable = {}, True
        for name in ("x", "y"):
            value = entry["axes"].get(name, 0.0)
            if abs(value) == 0.0:
                predicted[name] = 0.0
                continue
            speed = _predict(curves, name + ("+" if value > 0 else "-"), abs(value))
            if speed is None:
                usable = False
                break
            predicted[name] = speed
        if not usable:
            continue
        expected = math.hypot(predicted["x"], predicted["y"])
        achieved = math.hypot(entry["fwd"], entry["lat"])
        ratio = round(achieved / expected, 3) if expected > 0.0 else None
        rows.append({"label": entry["label"], "axes": entry["axes"],
                     "achieved_mps": round(achieved, 4),
                     "predicted_mps": round(expected, 4), "ratio": ratio,
                     "yaw_rate": round(entry["yaw_rate"], 4)})
        if ratio is not None:
            groups.setdefault(entry["label"].rsplit("/", 1)[0], []).append(ratio)
    return {"segments": rows,
            "mean_ratio": dict((key, round(statistics.mean(vals), 3))
                               for key, vals in groups.items())}


def _mean_of_signs(curves, prefix, field):
    """Average one fitted quantity across an axis's two signs."""
    values = [curves[key][field] for key in (prefix + "+", prefix + "-")
              if curves.get(key) and curves[key].get(field) is not None]
    return round(statistics.mean(values), 4) if values else None


def recommend(result):
    """The adapter parameter values this flight actually measured."""
    curves, moving = result["single_axis"], result["moving"]
    values = {}
    for prefix in ("x", "y", "r"):
        values[prefix + "_deadzone"] = _mean_of_signs(curves, prefix, "deadzone")
        values[prefix + "_v_full"] = _mean_of_signs(curves, prefix, "full_scale")
    down = moving.get("down") or {}
    values["x_v_full_moving"] = down.get("full_scale")
    ratios = list(result["combined"]["mean_ratio"].values())
    values["cross_axis_ratio"] = round(statistics.mean(ratios), 3) if ratios else None
    return values


def _fmt(value, digits=2):
    """Format a number for the report, or say plainly that it is missing."""
    return "not measured" if value is None else ("%%.%df" % digits) % value


def _markdown(result):
    """Render the human-readable half of the report."""
    values, curves = result["recommended"], result["single_axis"]
    lines = ["# Rooster ManualControl axis calibration", "",
             "Run: `%s`  --  %d segments flown, %d with settled samples."
             % (result["run_dir"], result["segments_flown"],
                result["segments_measured"]), "",
             "## Standing-start breakaway (lowest axis value that moved at all)", ""]
    for key in sorted(result["breakaway"]):
        lines.append("- `%s`: **%d counts**" % (key, int(result["breakaway"][key])))
    lines += ["", "## Fitted single-axis curves", "",
              "| axis | slope (per count) | dead band (counts) | value at 1000 | points |",
              "|---|---|---|---|---|"]
    for key in sorted(curves):
        curve = curves[key] or {}
        lines.append("| %s | %s | %s | %s | %s |" % (
            key, _fmt(curve.get("slope"), 5), _fmt(curve.get("deadzone"), 0),
            _fmt(curve.get("full_scale"), 3), curve.get("n", 0)))
    lines += ["", "## Standing vs moving (forward axis)", "",
              "Mean speed gap between the two approaches: **%s m/s**."
              % _fmt(result["hysteresis"]["mean_gap_mps"], 3), ""]
    preload = {"down": "850", "up": "650"}
    for key in sorted(result["moving"]):
        curve = result["moving"][key] or {}
        lines.append("- approached from the %s pre-load: dead band %s counts, "
                     "%s m/s at 1000" % (preload.get(key, key),
                                         _fmt(curve.get("deadzone"), 0),
                                         _fmt(curve.get("full_scale"), 3)))
    lines += ["", "## Combined axes", ""]
    for key, ratio in sorted(result["combined"]["mean_ratio"].items()):
        lines.append("- `%s`: achieved / single-axis prediction = **%.2f**" % (key, ratio))
    lines += ["",
              "A ratio above 1.0 means each axis must be commanded LESS when",
              "combined; below 1.0 means more. The per-axis dead-band offset is",
              "currently added once per axis, so a diagonal pays it twice.", "",
              "## Put these into `rooster_twist_control_adapter.py`", ""]
    for name in ("x_deadzone", "x_v_full", "x_v_full_moving",
                 "y_deadzone", "y_v_full", "r_deadzone", "r_v_full"):
        # Dead bands are axis counts; everything else is a rate.
        digits = 0 if name.endswith("_deadzone") else 3
        lines.append("- `%s = %s`" % (name, _fmt(values.get(name), digits)))
    lines += ["",
              "`r_deadzone` / `r_v_full` have no parameter in the adapter today --",
              "yaw still goes through `wz / max_yaw_rate * 1000`, with no dead band",
              "at all. `r_v_full` is rad/s at axis 1000, so `max_yaw_rate` should",
              "become that value once the dead band is applied alongside it.", ""]
    return "\n".join(lines)


def fit(run_dir):
    """Fit every curve this flight can support and write both reports.

    Args:
        run_dir: A calibration run directory holding ``truth.jsonl`` and
            ``segments.jsonl``.

    Returns:
        The same dict written to ``calibration.json``.

    Raises:
        ValueError: If the run has no segments or no settled samples at all --
            an empty report is worse than a loud failure, because it reads as a
            measurement.
    """
    run_dir = pathlib.Path(run_dir)
    rows = load_jsonl(run_dir / TRUTH_FILE)
    segments = load_jsonl(run_dir / SEGMENTS_FILE)
    if not segments:
        raise ValueError("%s holds no segments -- nothing was flown" % run_dir)
    measured = _measured(rows, segments)
    if not measured:
        raise ValueError(
            "%d segments flown but none had settled telemetry in their last "
            "%.0f s -- check the recorder's stream ages in truth.jsonl"
            % (len(segments), SETTLE_WINDOW_S))

    result = {"run_dir": str(run_dir), "segments_flown": len(segments),
              "segments_measured": len(measured),
              "breakaway": breakaway(measured),
              "single_axis": single_axis_curves(measured),
              "moving": moving_curves(measured),
              "hysteresis": hysteresis(measured)}
    result["combined"] = combined(measured, result["single_axis"])
    result["recommended"] = recommend(result)
    result["per_segment"] = [
        {"label": entry["label"], "axes": entry["axes"], "n": entry["n"],
         "fwd": round(entry["fwd"], 4), "lat": round(entry["lat"], 4),
         "yaw_rate": round(entry["yaw_rate"], 4)} for entry in measured]

    with open(str(run_dir / "calibration.json"), "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with open(str(run_dir / "calibration.md"), "w") as handle:
        handle.write(_markdown(result) + "\n")
    return result
