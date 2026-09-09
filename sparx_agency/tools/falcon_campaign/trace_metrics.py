"""Metrics from ``flight_trace.jsonl`` -- tracking against the plan, and mapping rate.

``LOOP_MISSION.md`` asks two questions of every flight, and neither was answerable
before this file existed. ``metrics.json``'s tracking block came from a 0.5 Hz
log line (~490 samples of one scalar); ``coverage.jsonl`` sampled one volume
every ~18 s. The follower has been publishing its complete per-tick internals on
``/nav_debug/control_trace`` the whole time and nothing collected them.

What this adds, per flight:

* **Where the error is.** Position error split into ALONG-track (lateness) and
  CROSS-track (off the route), separately for the whole flight and for the part
  where the reference is actually moving -- the two answer different questions
  and mixing them hid that cross-track p90 is ~1.9 m against a 0.55 m planner
  margin.
* **Whether the controller had authority left.** The share of ticks on which the
  position-correction term is railed at its own limit. A saturated correction
  cannot pull harder however far behind the aircraft is, so a gain change on a
  saturated loop does nothing.
* **What the reference was doing.** ``traj_server`` republishes a frozen endpoint
  with fresh stamps, so "reference is fresh" does not mean "reference is moving";
  every distribution here is available conditioned on it moving.
* **Mapping rate.** Volume and (once the image carries ``sparx_map_stats.sh``)
  known/free/occupied voxels, 2D footprint and 1 m^3 cells, at 2 Hz instead of
  once per 18 s -- plus the NEW-voxel rate over time, which is the discovery
  decay curve the campaign is for.

Every value is a plain summary of samples the flight already produced; nothing
here reads a controller's own estimate of anything (see
``analyze._fill_ground_velocity`` for why that matters).
"""
from __future__ import annotations

import bisect
import json
import math


#: The horizontal position PID's own output ceiling, m/s
#: (``reference_tracker_3d.params._default_horizontal_pid``). The trace reports
#: WHICH envelope limits bound a tick but not the PID's cap, so this is read from
#: the tracker's tuning; keep the two in step if that default ever changes.
HORIZONTAL_OUT_LIMIT = 1.0


def _stats(values):
    """Summarise samples, or ``None`` when there are none."""
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    def pct(p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]
    return dict(n=len(vals), mean=sum(vals) / len(vals), median=pct(0.5),
                p90=pct(0.9), p99=pct(0.99), min=vals[0], max=vals[-1])


def read_trace(path):
    """Rows of ``flight_trace.jsonl``, grouped by ``kind``. Bad lines are skipped."""
    rows = {}
    try:
        handle = open(str(path))
    except IOError:
        return rows
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows.setdefault(row.get("kind"), []).append(row)
    return rows


def tracking_metrics(ctrl_rows):
    """How closely the aircraft flew the plan, and whether it could have flown closer.

    Args:
        ctrl_rows: The ``kind == "ctrl"`` rows, each carrying the follower's
            whole per-tick trace.

    Returns:
        A dict of distributions, or ``None`` if no tick carried tracking data.
    """
    along, cross, perr, yawerr = [], [], [], []
    along_mv, cross_mv, perr_mv = [], [], []
    saturated = moving = holding = diverged = past_end = 0
    ticks = 0
    bound = {}
    for row in ctrl_rows:
        trace = row.get("trace") or {}
        track = trace.get("tracking")
        if not track:
            continue
        ticks += 1
        ref = trace.get("reference") or {}
        is_moving = bool(ref.get("moving"))
        moving += is_moving
        # Split, because they are not the same fault. `holding` used to mean
        # only "no fresh trajectory", which the scorer penalises. Since the
        # finished-plan hold landed, a deliberate stop at the last waypoint also
        # sets `holding` -- and that is correct behaviour, not a follower
        # failure. Runs recorded before the change carry no `past_end` key, so
        # the .get default keeps them scoring exactly as they did.
        finished = bool(track.get("past_end", False))
        past_end += finished
        holding += bool(track.get("holding")) and not finished
        diverged += bool(track.get("diverged"))
        lag = abs(track.get("along_track_lag_m") or 0.0)
        off = abs(track.get("cross_track_error_m") or 0.0)
        dist = track.get("position_error_m")
        along.append(lag)
        cross.append(off)
        perr.append(dist)
        yaw = track.get("yaw_error_rad")
        if yaw is not None:
            yawerr.append(abs(math.degrees(yaw)))
        if is_moving:
            along_mv.append(lag)
            cross_mv.append(off)
            perr_mv.append(dist)
        terms = trace.get("terms") or {}
        correction = terms.get("correction")
        if correction and len(correction) >= 2:
            if max(abs(correction[0]), abs(correction[1])) >= 0.999 * HORIZONTAL_OUT_LIMIT:
                saturated += 1
        for name in (terms.get("limits") or []):
            bound[name] = bound.get(name, 0) + 1
    if not ticks:
        return None
    along_sq = sum(v * v for v in along)
    cross_sq = sum(v * v for v in cross)
    total_sq = along_sq + cross_sq
    return dict(
        ticks=ticks,
        frac_reference_moving=moving / float(ticks),
        frac_holding=holding / float(ticks),
        #: Ticks holding at a FINISHED plan's last waypoint. Reported beside
        #: frac_holding rather than inside it: this one is the aircraft doing
        #: the right thing, and scoring it as a fault would penalise the fix.
        past_end_ticks=past_end,
        frac_past_end=past_end / float(ticks),
        frac_diverged=diverged / float(ticks),
        #: Share of ticks whose position correction was railed at its own limit.
        #: A saturated loop cannot answer a gain change.
        frac_correction_saturated=saturated / float(ticks),
        #: How often each of the tracker's own envelope limits bound the command,
        #: as a share of ticks. Names come from the follower's trace.
        frac_limit_bound=dict((k, v / float(ticks)) for k, v in bound.items()),
        #: Share of the squared position error that is lateness rather than
        #: being off the route. The two have different fixes.
        frac_error_along_track=(along_sq / total_sq) if total_sq > 0 else None,
        along_track_m=_stats(along), cross_track_m=_stats(cross),
        position_error_m=_stats(perr), yaw_error_deg=_stats(yawerr),
        moving=dict(along_track_m=_stats(along_mv), cross_track_m=_stats(cross_mv),
                    position_error_m=_stats(perr_mv)))


def _series_rate(samples, key):
    """Per-second first difference of ``key`` over ``(t, value)`` samples."""
    rates = []
    for older, newer in zip(samples, samples[1:]):
        dt = newer[0] - older[0]
        if dt <= 0.0:
            continue
        rates.append(((newer[0], (newer[1] - older[1]) / dt)))
    return rates


def mapping_metrics(rows):
    """How fast the map was built, and how the discovery rate decayed.

    Prefers ``/voxel_mapping/map_stats`` (2 Hz, added by
    ``patches/sparx_map_stats.sh``); falls back to the plain coverage volume when
    the image predates that patch, in which case the voxel/area/cell fields are
    simply absent rather than guessed.

    Returns:
        A dict with the cumulative series, the per-second discovery rates and
        their decay, or ``None`` if the flight carried neither stream.
    """
    stats = rows.get("map") or []
    coverage = rows.get("coverage") or []
    frontier = rows.get("frontier") or []
    out = {}

    if coverage:
        vol = [(r["t"], r["volume_m3"]) for r in coverage if r.get("volume_m3") is not None]
        if len(vol) > 1:
            span = vol[-1][0] - vol[0][0]
            gained = vol[-1][1] - vol[0][1]
            out["volume"] = dict(
                samples=len(vol), span_s=span, first_m3=vol[0][1], final_m3=vol[-1][1],
                gained_m3=gained,
                rate_m3_per_min=(60.0 * gained / span) if span > 0 else None,
                #: Seconds of flight per cubic metre mapped -- the operator's
                #: "how long does 1x1x1 take", in its simplest form.
                seconds_per_m3=(span / gained) if gained > 0 else None,
                rate_series=_series_rate(vol, "volume_m3"))

    if stats:
        def series(field):
            return [(r["t"], r[field]) for r in stats if r.get(field) is not None]
        known = series("known_voxels")
        area = series("known_area_m2")
        cells = series("known_cells_1m3")
        if len(known) > 1:
            span = known[-1][0] - known[0][0]
            gained = known[-1][1] - known[0][1]
            out["voxels"] = dict(
                samples=len(known), span_s=span,
                final=known[-1][1], gained=gained,
                per_second=(gained / span) if span > 0 else None,
                #: THE decay curve: new known voxels per second over time. Early
                #: flight discovers fast, later flight revisits.
                discovery_rate=_series_rate(known, "known_voxels"))
        for name, samples in (("area_m2", area), ("cells_1m3", cells)):
            if len(samples) > 1:
                span = samples[-1][0] - samples[0][0]
                gained = samples[-1][1] - samples[0][1]
                out[name] = dict(
                    final=samples[-1][1], gained=gained,
                    per_second=(gained / span) if span > 0 else None,
                    #: The operator's "how long does a square metre take".
                    seconds_per_unit=(span / gained) if gained > 0 else None,
                    rate_series=_series_rate(samples, name))
        last = stats[-1]
        for field in ("free_voxels", "occupied_voxels", "box_voxels",
                      "box_cells_1m3", "resolution"):
            if last.get(field) is not None:
                out.setdefault("final", {})[field] = last[field]

    if frontier:
        out["frontier_points"] = _stats([r.get("points") for r in frontier])
    return out or None


def bspline_metrics(bspline_rows):
    """The shape of the plans the aircraft was asked to fly.

    A trajectory the platform cannot fly is a tracking error nothing downstream
    can fix, so the plan's own peak speed is worth recording beside the error.

    Returns:
        Distributions of each published trajectory's duration, path length and
        mean control-point speed, or ``None``.
    """
    durations, lengths, speeds, yaw_spans = [], [], [], []
    for row in bspline_rows:
        knots = row.get("knots") or []
        pts = row.get("pos_pts") or []
        if len(knots) < 2 or len(pts) < 2:
            continue
        duration = knots[-1] - knots[0]
        length = sum(math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
                     for a, b in zip(pts, pts[1:]))
        durations.append(duration)
        lengths.append(length)
        if duration > 0:
            speeds.append(length / duration)
        yaw = row.get("yaw_pts") or []
        if len(yaw) > 1:
            yaw_spans.append(max(yaw) - min(yaw))
    if not durations:
        return None
    return dict(trajectories=len(durations), duration_s=_stats(durations),
                length_m=_stats(lengths), mean_speed_mps=_stats(speeds),
                yaw_span_rad=_stats(yaw_spans))


# -- distance to the curve itself ----------------------------------------
#: Degree of FALCON's position B-spline (`trajectory/Bspline.order`, always 3).
BSPLINE_DEGREE = 3

#: Coarse samples along a curve before the local refine. 60 over a ~5 m plan is
#: ~8 cm between samples, and the golden-section refine that follows takes the
#: residual to millimetres.
CURVE_COARSE_SAMPLES = 60

#: A plan older than this is not what the aircraft is being flown along any more;
#: traj_server freezes the endpoint rather than stopping, so without this the
#: distance would be measured against a curve nobody is following.
CURVE_MAX_PLAN_AGE_S = 12.0


def _deboor(t, knots, ctrl, degree=BSPLINE_DEGREE):
    """Evaluate a non-uniform B-spline at parameter ``t``.

    Args:
        t: Curve parameter, clamped into the valid span.
        knots: The knot vector as published.
        ctrl: Control points, each an ``(x, y, z)`` sequence.
        degree: Spline degree.

    Returns:
        The ``[x, y, z]`` point on the curve.
    """
    last = len(ctrl) - 1
    lo, hi = knots[degree], knots[last + 1]
    t = min(max(t, lo), hi)
    span = bisect.bisect_right(knots, t) - 1
    span = min(max(span, degree), last)
    work = [list(ctrl[j + span - degree]) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = j + span - degree
            den = knots[i + degree + 1 - r] - knots[i]
            a = 0.0 if den == 0 else (t - knots[i]) / den
            work[j] = [(1.0 - a) * work[j - 1][m] + a * work[j][m] for m in range(3)]
    return work[degree]


def curve_distance(pos, knots, ctrl):
    """Shortest distance from ``pos`` to the B-spline curve, metres.

    This is the operator's actual question -- *how far are we from the route* --
    and it is **not** what ``cross_track_error_m`` measures. That splits the
    error against the reference POINT and its velocity direction, so on a curved
    plan an aircraft exactly on the route but behind schedule still reports a
    cross-track equal to curvature x lag. Measured on a real flight, the point
    metric overstated the tail by 2-3x: p90 1.94 m against a true 1.25 m, and
    "further than 2 m" 8.6 % against a true 2.5 %.

    Coarse sample then golden-section refine; the curve is not convex in the
    parameter, hence the sampling rather than a solve.

    Returns:
        The distance, or ``None`` when the span is degenerate.
    """
    if len(ctrl) <= BSPLINE_DEGREE or len(knots) <= len(ctrl):
        return None
    lo, hi = knots[BSPLINE_DEGREE], knots[len(ctrl)]
    if hi <= lo:
        return None

    def dist_sq(t):
        point = _deboor(t, knots, ctrl)
        return sum((point[m] - pos[m]) ** 2 for m in range(3))

    step = (hi - lo) / CURVE_COARSE_SAMPLES
    best_t, best = lo, dist_sq(lo)
    for i in range(1, CURVE_COARSE_SAMPLES + 1):
        t = lo + i * step
        value = dist_sq(t)
        if value < best:
            best, best_t = value, t
    left, right = max(lo, best_t - step), min(hi, best_t + step)
    for _ in range(30):
        m1, m2 = left + (right - left) / 3.0, right - (right - left) / 3.0
        if dist_sq(m1) < dist_sq(m2):
            right = m2
        else:
            left = m1
    return math.sqrt(dist_sq((left + right) / 2.0))


def _odom_speeds(odom_rows):
    """Aircraft speed at each odom sample, by first difference.

    Conditioning on the AIRCRAFT moving, not on the reference moving. Any error
    metric can be gamed by standing still, and a stalled flight sits on the curve
    it never left -- measured 2026-09-02, the flight that tracked the route best
    (p50 0.043 m) mapped 233 m3 while the one at p50 0.478 m mapped 3894 m3. The
    reference kept moving through both, so conditioning on the reference does not
    remove the confound; conditioning on the aircraft does.

    Returns:
        A list the same length as ``odom_rows``, with ``None`` where no speed
        could be formed.
    """
    speeds = [None] * len(odom_rows)
    for i in range(1, len(odom_rows)):
        older, newer = odom_rows[i - 1], odom_rows[i]
        if older.get("t") is None or newer.get("t") is None:
            continue
        dt = newer["t"] - older["t"]
        if not (0.0 < dt < 1.0):
            continue
        speeds[i] = math.sqrt(sum((newer[k] - older[k]) ** 2
                                  for k in ("x", "y", "z"))) / dt
    return speeds


def route_metrics(bspline_rows, odom_rows, ref_rows=(), safe_distance_m=0.55,
                  moving_speed_mps=0.05):
    """How far the aircraft actually was from FALCON's planned curve.

    Args:
        bspline_rows: ``kind == "bspline"`` rows, in arrival order.
        odom_rows: ``kind == "odom"`` rows, in arrival order.
        safe_distance_m: The planner's own clearance margin, for the
            share-of-flight-outside figure.

    Returns:
        Distance distributions plus the share of the flight spent further from
        the route than the planner's margin, or ``None`` if either stream is
        missing.
    """
    plans = [(r["t"], r.get("knots") or [], r.get("pos_pts") or [])
             for r in bspline_rows]
    plans = [p for p in plans if len(p[1]) > BSPLINE_DEGREE and len(p[2]) > BSPLINE_DEGREE]
    if not plans or not odom_rows:
        return None
    stamps = [p[0] for p in plans]
    speeds = _odom_speeds(odom_rows)
    distances, moving_distances = [], []
    for i, row in enumerate(odom_rows):
        t = row.get("t")
        if t is None:
            continue
        idx = bisect.bisect_right(stamps, t) - 1
        if idx < 0 or t - stamps[idx] > CURVE_MAX_PLAN_AGE_S:
            continue
        value = curve_distance((row["x"], row["y"], row["z"]), plans[idx][1], plans[idx][2])
        if value is None:
            continue
        distances.append(value)
        if speeds[i] is not None and speeds[i] >= moving_speed_mps:
            moving_distances.append(value)
    if not distances:
        return None

    def outside(values):
        return (sum(1 for v in values if v > safe_distance_m) / float(len(values))
                if values else None)

    return dict(distance_to_curve_m=_stats(distances),
                samples=len(distances),
                safe_distance_m=safe_distance_m,
                #: Share of the flight spent further from the planned route than
                #: the planner's own clearance margin -- i.e. in geometry the
                #: plan never promised was free.
                frac_outside_safe_distance=outside(distances),
                #: The same, restricted to ticks where the AIRCRAFT was moving.
                #: This is the comparison-safe one: a stalled flight sits on the
                #: curve it never left and would otherwise score a perfect route
                #: error while mapping nothing.
                moving=dict(distance_to_curve_m=_stats(moving_distances),
                            samples=len(moving_distances),
                            frac_outside_safe_distance=outside(moving_distances)))


def motion_from_odom(odom_rows, stop_speed_mps=0.05):
    """Speed, distance and stationary share from the trace's own odometry.

    Deliberate redundancy with ``analyze.motion_metrics``, which reads
    ``truth.jsonl`` from the ROS 2 recorder in the ``it`` container. The two
    recorders live in different containers and fail independently -- when the
    ROS 2 one died on 2026-09-02 the whole flight's motion metrics went null and
    the guard on "did this change just make the aircraft stop moving" went with
    them. That guard is the one protecting against the confound the primary
    metric was corrected for, so it must not depend on a single process.

    ``/odom_world`` is FALCON's own view of the aircraft, differenced by
    ``falcon_adapter_node`` at 25 Hz and sampled here at 10 Hz.

    Args:
        odom_rows: The ``kind == "odom"`` rows.
        stop_speed_mps: Below this the aircraft counts as stationary.

    Returns:
        A dict of motion figures, or ``None`` with fewer than two samples.
    """
    rows = [r for r in odom_rows if r.get("t") is not None]
    if len(rows) < 2:
        return None
    speeds, distance, moving_time, total = [], 0.0, 0.0, 0.0
    for older, newer in zip(rows, rows[1:]):
        dt = newer["t"] - older["t"]
        if not (0.0 < dt < 1.0):
            continue
        step = math.sqrt(sum((newer[k] - older[k]) ** 2 for k in ("x", "y", "z")))
        speed = step / dt
        speeds.append(speed)
        distance += step
        total += dt
        if speed >= stop_speed_mps:
            moving_time += dt
    if not speeds:
        return None
    return dict(samples=len(speeds), duration_s=total, distance_m=distance,
                speed_mps=_stats(speeds),
                frac_time_below_stop_speed=(1.0 - moving_time / total) if total else None)


COURSE_SLEW_CEILING_RAD_S = math.radians(45.0)

#: Ticks after the steering gate re-opens that still count as the resume
#: transient. Measured 2026-09-03: saturation is 98% on the first tick and only
#: reaches its steady 47% after about this many. See LOOP_FIXES.md F19.
COURSE_RESUME_TICKS = 50


#: Reference speed below which the PLAN counts as standing still, m/s.
REFERENCE_STILL_MPS = 0.05


def reference_motion_metrics(ctrl_rows):
    """How much of the flight FALCON's own reference is standing still.

    The tracker cannot follow a plan that is not moving, and coverage cannot
    exceed what the plan asks for, so this sits upstream of every control
    metric. Measured 2026-09-03 at 50% of ticks across eight flights, with the
    reference going dead about three seconds after each trajectory is published
    (LOOP_BUGS.md B34).

    Args:
        ctrl_rows: The ``ctrl`` records of one flight's trace.

    Returns:
        A dict, or ``None`` when no tick carried a reference velocity.
    """
    total = still = yawing_while_still = 0
    for row in ctrl_rows:
        ref = (row.get("trace") or {}).get("reference") or {}
        vx, vy = ref.get("vx"), ref.get("vy")
        if vx is None or vy is None:
            continue
        total += 1
        if math.hypot(vx, vy) < REFERENCE_STILL_MPS:
            still += 1
            if abs(ref.get("yaw_dot") or 0.0) > 0.05:
                yawing_while_still += 1
    if not total:
        return None
    return dict(ticks=total, still_ticks=still,
                frac_still=still / float(total),
                frac_still_but_yawing=(yawing_while_still / float(still)
                                       if still else None))


def course_metrics(ctrl_rows):
    """How hard the course-slew limiter is working, and how often it resumes.

    The limiter rate-limits the commanded course to ``course_slew_deg_s``. When
    it is railed, the nose is chasing a demand it cannot reach, which is the
    mechanism behind the standing heading error (LOOP_BUGS.md B31).

    A tick counts as *steering* exactly when the node produced a heading error;
    ``heading_err_rad`` is null otherwise, and it agrees with the speed gate on
    100% of ticks, so it works on traces recorded before the gate existed.

    Args:
        ctrl_rows: The ``ctrl`` records of one flight's trace.

    Returns:
        A dict, or ``None`` when no tick carried a course rate.
    """
    steering = saturated = 0
    steady = steady_sat = 0
    resumes = 0
    since = None
    total = 0
    for row in ctrl_rows:
        gate = (row.get("trace") or {}).get("gate") or {}
        rate = gate.get("course_rate")
        if rate is None:
            continue
        total += 1
        if gate.get("heading_err_rad") is None:
            since = None
            continue
        if since is None:
            resumes += 1
            since = 0
        else:
            since += 1
        steering += 1
        railed = abs(abs(rate) - COURSE_SLEW_CEILING_RAD_S) < 1e-3
        saturated += railed
        if since >= COURSE_RESUME_TICKS:
            steady += 1
            steady_sat += railed
    if not total:
        return None
    return dict(
        ticks=total, steering_ticks=steering,
        frac_steering=steering / float(total),
        frac_saturated=(saturated / float(steering)) if steering else None,
        frac_saturated_steady=(steady_sat / float(steady)) if steady else None,
        steady_ticks=steady, resumes=resumes)


def trace_metrics(run_dir):
    """Every metric derivable from one run's ``flight_trace.jsonl``.

    Args:
        run_dir: The run folder.

    Returns:
        A dict, or ``None`` when the flight has no trace (an older run, or a
        cycle whose probe never started).
    """
    rows = read_trace(run_dir / "flight_trace.jsonl")
    if not rows:
        return None
    out = dict(rows=dict((k, len(v)) for k, v in rows.items()))
    tracking = tracking_metrics(rows.get("ctrl") or [])
    if tracking:
        out["tracking"] = tracking
    reference = reference_motion_metrics(rows.get("ctrl") or [])
    if reference:
        out["reference"] = reference
    course = course_metrics(rows.get("ctrl") or [])
    if course:
        out["course"] = course
    mapping = mapping_metrics(rows)
    if mapping:
        out["mapping"] = mapping
    plans = bspline_metrics(rows.get("bspline") or [])
    if plans:
        out["plans"] = plans
    route = route_metrics(rows.get("bspline") or [], rows.get("odom") or [],
                          rows.get("ref") or [])
    if route:
        out["route"] = route
    motion = motion_from_odom(rows.get("odom") or [])
    if motion:
        out["motion"] = motion
    return out
