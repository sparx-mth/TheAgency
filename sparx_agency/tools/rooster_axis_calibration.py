#!/usr/bin/env python3
"""Automated ManualControl axis-calibration flight for the Rooster in Sphera.

MISSION.md P5 says the axis curve is still guessed: the standing-start
breakaway has never been measured per sign, the moving-regime gain has never
been measured at all, yaw has no calibrated inverse, and nothing anywhere models
what happens when two axes are commanded at once. Every one of those gaps is
currently filled by a constant somebody wrote down after two data points. This
tool replaces the guessing with an experiment.

**Mechanism.** It publishes ``{"action": "move", "axes": {...}}`` to
``/<drone>/cmd_nav`` and ``{"action": "stop"}`` between segments.
``rooster_command_unit`` holds the axes until they change and never accepts
``z`` from a ``move`` payload, so altitude hold keeps flying the aircraft for
the whole experiment and no segment can drop it. Nothing here ever sends
``up``/``down`` alongside a move.

**Why it must own the aircraft alone.** The twist control adapter publishes
``stop`` at 20 Hz whenever ``/cmd_vel`` is quiet, which would cancel every
segment within ~50 ms, and a second publisher on ``/R1/manual_control`` has
silently destroyed a full day of measurements before (LESSONS.md 2026-08-17).
Both are asserted before the first segment, not assumed.

**Why it checkpoints.** The whole sweep does not fit in one Sphera battery, and
``MIN_FLIGHT_BATTERY`` is a data-quality gate as much as a safety one -- below
it a "forward" command produces mostly lateral motion. So a battery floor ends
the run cleanly and the next invocation resumes at the first segment that has
not been flown.

Usage::

    python3 -m sparx_agency.tools.rooster_axis_calibration --dry-run
    python3 -m sparx_agency.tools.rooster_axis_calibration --run --blocks i,ii
    python3 -m sparx_agency.tools.rooster_axis_calibration --fit runs/axiscal_...

Python 3.8-compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import time

from sparx_agency.tools import falcon_flight_sequence as seq
from sparx_agency.tools import rooster_axis_calibration_fit as fitting
from sparx_agency.tools.falcon_campaign import bringup, campaign, config as C

#: Hard ceiling on any commanded axis. Block (ii) is the only exemption: its
#: 850 pre-load and 900/1000 samples exist to measure the top of the curve.
MAX_SAFE_AXIS = 800.0
MAX_ANY_AXIS = 1000.0

#: Roll/pitch magnitude that ends a segment. Signs are UNVERIFIED on
#: ``/<drone>/attitude_rpy`` -- magnitude only, never a signed comparison.
TILT_LIMIT_DEG = 25.0

#: Ranger band the aircraft must stay inside, metres.
RANGER_MIN_M, RANGER_MAX_M = 0.8, 3.0

#: A feedback stream older than this is not evidence of anything.
MAX_FEEDBACK_AGE_S = 1.0

#: A standing start is only a standing start below this measured speed.
STANDING_SPEED_MPS = 0.03

POLL_S = 0.3
#: Two ``ros2 topic pub -1`` calls per segment, measured at ~1.5 s each.
PUBLISH_OVERHEAD_S = 3.0

RUN_PREFIX = "axiscal_"
SEGMENTS_FILE = "segments.jsonl"
CHECKPOINT_FILE = "checkpoint.json"

BREAKAWAY_VALUES = (550, 580, 610, 640, 670, 700, 750, 800)
STEADY_VALUES = (450, 500, 550, 600, 620, 650, 700, 800, 900, 1000)
YAW_VALUES = (80, 100, 120, 150, 200, 300, 400, 600, 800, 1000)


class CalibrationAbort(RuntimeError):
    """A precondition or command failed; nothing may be flown."""


class BatteryExhausted(RuntimeError):
    """The pack fell below the floor mid-run; land, checkpoint and resume later."""


# ── commanding ───────────────────────────────────────────────────────────
def _publish(payload, timeout=15.0):
    """Publish one ``cmd_nav`` payload, the way ``falcon_flight_sequence`` does.

    Args:
        payload: The action dict, e.g. ``{"action": "move", "axes": {...}}``.
        timeout: Seconds to allow the publish to complete.

    Raises:
        CalibrationAbort: If the publish fails -- a segment commanded into a
            void would be recorded as a measurement of zero.
    """
    data = json.dumps({"data": json.dumps(payload)})
    cmd = (C.IT_ENV + "ros2 topic pub -1 /%s/cmd_nav std_msgs/msg/String %s"
           % (C.DRONE_ID, shlex.quote(data)))
    try:
        result = subprocess.run(
            ["docker", "exec", C.IT_CONTAINER, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CalibrationAbort("cmd_nav %r timed out after %.0fs" % (payload, timeout))
    if result.returncode != 0:
        raise CalibrationAbort("cmd_nav %r failed: %s"
                               % (payload, result.stderr.strip()[:200]))


def _stop():
    """Zero x/y/r while leaving z (throttle/altitude hold) untouched."""
    _publish({"action": "stop"})


def _move(axes):
    """Hold one axis triple until the next ``move`` or ``stop``."""
    _publish({"action": "move", "axes": axes})


# ── telemetry ────────────────────────────────────────────────────────────
def _tail_truth():
    """Newest recorder sample, read from inside ``it``, or ``None``.

    The 20 Hz flight recorder is already writing every stream this needs, so the
    safety monitor tails its last line rather than opening its own ROS
    subscriptions -- and ``ros2 topic echo`` piped into a short ``timeout``
    block-buffers and hangs (LESSONS.md 2026-08-18).
    """
    try:
        result = subprocess.run(
            ["docker", "exec", C.IT_CONTAINER, "tail", "-n", "1",
             C.RECORDER_DIR_IN_IT + "/truth.jsonl"],
            capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None                    # a torn last line; the next poll is whole


def _fresh(sample, stream):
    """One stream's payload if it is fresh enough to believe, else ``None``."""
    entry = (sample or {}).get(stream) or {}
    age = entry.get("age")
    return entry if age is not None and age <= MAX_FEEDBACK_AGE_S else None


def _speed_xy(sample):
    """Measured horizontal speed, or ``None`` when the feedback is stale."""
    velocity = _fresh(sample, "velocity")
    if not velocity or velocity.get("vx") is None or velocity.get("vy") is None:
        return None
    return math.hypot(velocity["vx"], velocity["vy"])


def _battery(sample):
    """Battery as a 0-1 fraction, or ``None`` when the state stream is stale."""
    state = _fresh(sample, "state")
    level = state.get("battery") if state else None
    if level is None:
        return None
    return level if level <= 1.0 else level / 100.0


def _safety_trip(sample, min_battery):
    """Check every continuous limit against one sample.

    Returns:
        ``(reason, fatal)``. ``reason`` is ``None`` when everything is inside
        its limit; ``fatal`` means the whole run must land, not just this
        segment.
    """
    if sample is None:
        return "the flight recorder produced no telemetry", False
    attitude = _fresh(sample, "attitude")
    if not attitude:
        return "attitude feedback stale", False
    for name in ("roll", "pitch"):
        value = attitude.get(name)
        if value is None:
            return "no %s on the attitude stream" % name, False
        if abs(math.degrees(value)) > TILT_LIMIT_DEG:
            return ("|%s| %.0f deg over the %.0f deg limit"
                    % (name, abs(math.degrees(value)), TILT_LIMIT_DEG), False)
    state = _fresh(sample, "state")
    ranger = state.get("ranger") if state else None
    if ranger is None:
        return "ranger unreadable", False
    if not RANGER_MIN_M <= ranger <= RANGER_MAX_M:
        return ("ranger %.2f m outside [%.1f, %.1f] m"
                % (ranger, RANGER_MIN_M, RANGER_MAX_M), False)
    if _speed_xy(sample) is None:
        return "velocity feedback stale", False
    level = _battery(sample)
    if level is not None and level < min_battery:
        return ("battery %.0f%% below the %.0f%% floor"
                % (level * 100.0, min_battery * 100.0), True)
    return None, False


# ── the experiment ───────────────────────────────────────────────────────
def _axes(x=0.0, y=0.0, r=0.0):
    """One axis triple. ``z`` is deliberately absent -- it is never ours."""
    return {"x": float(x), "y": float(y), "r": float(r)}


def block_standing_start():
    """Block (i): the lowest axis value that moves a stationary aircraft.

    Per axis AND per sign, because the two directions are separate actuators
    until measurement says otherwise -- the only published numbers (x dead below
    ~620, y below ~700) were taken in one direction only.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for axis in ("x", "y", "r"):
        for sign in (1, -1):
            for value in BREAKAWAY_VALUES:
                label = "i/%s%s/%d" % (axis, "+" if sign > 0 else "-", value)
                segments.append((label, _axes(**{axis: sign * value}), 3.0, 4.0))
    return segments


def block_steady_state():
    """Block (ii): forward gain while the aircraft is ALREADY moving.

    Each sample is approached twice, from an 850 pre-load ("down") and a 650 one
    ("up"); the gap between the two curves at the same axis value *is* the
    standing-vs-moving hysteresis that made the 2026-08-18 curve under-deliver
    by 3x in flight. The pre-load is its own zero-settle segment so the step
    lands with no stop in between; at the sweep's ends the step direction
    inverts, but the pre-load each was approached from is what the fit uses.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for approach, preload in (("down", 850), ("up", 650)):
        for value in STEADY_VALUES:
            label = "ii/%s/%d" % (approach, value)
            segments.append((label + "/preload", _axes(x=preload), 2.0, 4.0))
            segments.append((label, _axes(x=value), 5.0, 0.0))
    return segments


def block_combined():
    """Block (iii): what happens when more than one axis is commanded at once.

    Nothing in the stack models this today: the dead-band offset is added per
    axis, so a diagonal pays it twice, and yaw's coupling into translation has a
    measured law (``turn_coordination``) that was never wired to the Rooster.
    The singles here are references measured in the same conditions as the
    combinations, so the ratio between them is not contaminated by the block-(i)
    standing start.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for value in (700, 750, 800):
        segments.append(("iii/a/xy%d" % value, _axes(x=value, y=value), 5.0, 4.0))
    for value in (700, 800):
        segments.append(("iii/a/x%d" % value, _axes(x=value), 5.0, 4.0))
        segments.append(("iii/a/y%d" % value, _axes(y=value), 5.0, 4.0))
    for forward in (700, 800):
        for turn in (0, 150, 250, 400, 600):
            segments.append(("iii/b/x%d_r%d" % (forward, turn),
                             _axes(x=forward, r=turn), 5.0, 4.0))
    for forward, lateral, turn in ((700, 700, 250), (800, 800, 400)):
        segments.append(("iii/c/x%d_y%d_r%d" % (forward, lateral, turn),
                         _axes(forward, lateral, turn), 5.0, 4.0))
    for sign in (1, -1):
        for value in YAW_VALUES:
            segments.append(("iii/d/r%s%d" % ("+" if sign > 0 else "-", value),
                             _axes(r=sign * value), 4.0, 4.0))
    return segments


BLOCKS = {"i": block_standing_start, "ii": block_steady_state,
          "iii": block_combined}


def build_plan(blocks):
    """Concatenate the requested blocks in order.

    Raises:
        ValueError: On an unknown block name, rather than silently flying less
            than was asked for.
    """
    plan = []
    for key in blocks:
        if key not in BLOCKS:
            raise ValueError("unknown block %r; known: %s"
                             % (key, ", ".join(sorted(BLOCKS))))
        plan.extend(BLOCKS[key]())
    return plan


def _assert_within_limits(label, axes):
    """Refuse any axis past its cap for this block."""
    limit = MAX_ANY_AXIS if label.startswith("ii/") else MAX_SAFE_AXIS
    for name, value in axes.items():
        if abs(value) > limit:
            raise CalibrationAbort(
                "segment %s asks for %s=%g, past the %g cap" % (label, name, value, limit))


# ── flying it ────────────────────────────────────────────────────────────
def _monitor(seconds, min_battery):
    """Hold for a while, polling every limit.

    Returns:
        ``(reason, fatal)`` -- ``(None, False)`` if the whole interval passed.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(POLL_S, max(0.0, deadline - time.monotonic())))
        reason, fatal = _safety_trip(_tail_truth(), min_battery)
        if reason:
            return reason, fatal
    return None, False


def _await_rest(min_battery, extra_s=6.0):
    """Wait until the aircraft is measurably still, or say why it is not."""
    deadline = time.monotonic() + extra_s
    while True:
        sample = _tail_truth()
        reason, fatal = _safety_trip(sample, min_battery)
        if reason:
            return reason, fatal
        speed = _speed_xy(sample)
        if speed is not None and speed < STANDING_SPEED_MPS:
            return None, False
        if time.monotonic() > deadline:
            return ("never came to rest: %.3f m/s still measured"
                    % (speed if speed is not None else float("nan")), False)
        time.sleep(POLL_S)


def _run_segment(segment, min_battery):
    """Fly one segment and return its record.

    Raises:
        BatteryExhausted: On the battery floor, so the caller lands and
            checkpoints rather than continuing on a corrupted pack.
    """
    label, axes, hold_s, settle_s = segment
    _assert_within_limits(label, axes)
    record = {"label": label, "block": label.split("/")[0], "axes": axes,
              "hold_s": hold_s, "settle_s": settle_s, "aborted": False,
              "reason": None, "t_cmd": None, "t_end": None}

    _stop()
    reason, fatal = _monitor(settle_s, min_battery)
    if reason is None and label.startswith("i/"):
        reason, fatal = _await_rest(min_battery)

    if reason is None:
        record["t_cmd"] = round(time.time(), 4)
        _move(axes)
        reason, fatal = _monitor(hold_s, min_battery)
        record["t_end"] = round(time.time(), 4)

    if reason is not None:
        _stop()
        record["aborted"] = True
        record["reason"] = reason
        if fatal:
            raise BatteryExhausted(reason)
    return record


# ── run bookkeeping ──────────────────────────────────────────────────────
def _read_checkpoint(run_dir):
    """Load a run's checkpoint, or ``None`` if it has none."""
    path = run_dir / CHECKPOINT_FILE
    if not path.exists():
        return None
    with open(str(path)) as handle:
        return json.load(handle)


def _resolve_run_dir():
    """Resume the newest unfinished calibration run, or start a new one.

    Resuming is the normal case, not the exception: the sweep outlives the
    battery, so a fresh directory per invocation would restart the experiment
    every time Sphera was restarted.

    Returns:
        ``(run_dir, checkpoint)``.
    """
    for path in sorted(C.RUNS_DIR.glob(RUN_PREFIX + "*"), reverse=True):
        checkpoint = _read_checkpoint(path)
        if checkpoint and not checkpoint.get("complete"):
            return path, checkpoint
    run_dir = C.RUNS_DIR / (RUN_PREFIX + time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, {"done": [], "complete": False}


def _checkpoint(run_dir, checkpoint, record):
    """Append one finished segment to both the log and the checkpoint."""
    with open(str(run_dir / SEGMENTS_FILE), "a") as handle:
        handle.write(json.dumps(record) + "\n")
    checkpoint["done"].append(record["label"])
    checkpoint["updated"] = time.time()
    with open(str(run_dir / CHECKPOINT_FILE), "w") as handle:
        json.dump(checkpoint, handle, indent=2)


def _collect(run_dir):
    """Bring this session's recorder output into the run directory.

    Appended rather than copied: ``start_recorder`` wipes the in-container
    directory each session, so a plain ``docker cp`` would overwrite everything
    the previous battery's worth of segments recorded.
    """
    bringup.sh("docker exec %s cat %s/truth.jsonl >> %s"
               % (C.IT_CONTAINER, C.RECORDER_DIR_IN_IT,
                  shlex.quote(str(run_dir / fitting.TRUTH_FILE))), 180)
    index = len(list(run_dir.glob("recorder_meta_*.json")))
    bringup.sh("docker cp %s:%s/recorder_meta.json %s"
               % (C.IT_CONTAINER, C.RECORDER_DIR_IN_IT,
                  shlex.quote(str(run_dir / ("recorder_meta_%d.json" % index)))), 60)


def _preflight(min_battery):
    """Assert every condition that makes a measurement worth taking.

    Raises:
        CalibrationAbort: On any failure, with the reason spelled out.
    """
    image = bringup.assert_simulator()
    bringup.stop_twist_adapter()
    ok, detail = bringup.manual_control_authority()
    if not ok:
        raise CalibrationAbort("the throttle axis is not ours alone: %s" % detail)
    level = bringup.battery_fraction()
    if level is None:
        raise CalibrationAbort(
            "battery unreadable -- an unverifiable pack invalidates every speed "
            "this flight would measure")
    if level < min_battery:
        raise CalibrationAbort("battery %.0f%% below the %.0f%% floor"
                               % (level * 100.0, min_battery * 100.0))
    for action in ("arm", "takeoff"):
        error = seq.send_cmd_nav(action, C.DRONE_ID)
        if error:
            raise CalibrationAbort("%s failed: %s" % (action, error))
    settled, message = seq.wait_for_stable_hover(
        C.DRONE_ID, timeout=C.HOVER_SETTLE_TIMEOUT_S)
    if not settled:
        raise CalibrationAbort("never reached a settled hover: %s" % message)
    return {"drone_image": image, "battery": level,
            "manual_authority": detail, "hover": message}


def _land():
    """Bring the aircraft down. Safe from any state, including twice."""
    for action in ("stop", "land"):
        seq.send_cmd_nav(action, C.DRONE_ID)
        time.sleep(2)
    time.sleep(20)
    seq.send_cmd_nav("disarm", C.DRONE_ID)


def dry_run(blocks):
    """Print the plan and its estimated duration without commanding anything."""
    plan = build_plan(blocks)
    total = 0.0
    for label, axes, hold_s, settle_s in plan:
        _assert_within_limits(label, axes)
        total += hold_s + settle_s + PUBLISH_OVERHEAD_S
        print("%-22s x=%-7g y=%-7g r=%-7g  settle %.1fs  hold %.1fs"
              % (label, axes["x"], axes["y"], axes["r"], settle_s, hold_s))
    print("\n%d segments, ~%.0f s (%.1f min) including ~%.0f s of publish overhead"
          % (len(plan), total, total / 60.0, PUBLISH_OVERHEAD_S * len(plan)))
    return plan, total


def run(blocks, min_battery):
    """Fly the experiment, resuming whatever a previous battery left unflown.

    Returns:
        A summary dict; also written to ``session.json`` in the run directory.
    """
    plan = build_plan(blocks)
    run_dir, checkpoint = _resolve_run_dir()
    done = set(checkpoint.get("done", []))
    remaining = [s for s in plan if s[0] not in done]
    summary = {"run_dir": str(run_dir), "planned": len(plan),
               "already_done": len(plan) - len(remaining), "flown": 0,
               "aborted_segments": 0, "ended": "completed"}
    print("[axiscal] %s -- %d of %d segments left"
          % (run_dir, len(remaining), len(plan)), flush=True)
    if not remaining:
        checkpoint["complete"] = True
        _checkpoint_only(run_dir, checkpoint)
        return summary

    try:
        summary["preflight"] = _preflight(min_battery)
        estimate = sum(s[2] + s[3] + PUBLISH_OVERHEAD_S for s in remaining)
        campaign.start_recorder(run_dir, estimate)
        time.sleep(3)
        for segment in remaining:
            record = _run_segment(segment, min_battery)
            _checkpoint(run_dir, checkpoint, record)
            summary["flown"] += 1
            summary["aborted_segments"] += int(record["aborted"])
            print("[axiscal] %-22s %s" % (
                record["label"], record["reason"] or "ok"), flush=True)
        checkpoint["complete"] = True
    except BatteryExhausted as exc:
        summary["ended"] = "battery: %s -- restart Sphera and re-run to resume" % exc
    except CalibrationAbort as exc:
        summary["ended"] = "aborted: %s" % exc
    finally:
        try:
            _stop()
        except CalibrationAbort:
            pass
        campaign.stop_recorder()
        _land()
        _collect(run_dir)
        _checkpoint_only(run_dir, checkpoint)
        with open(str(run_dir / "session.json"), "w") as handle:
            json.dump(summary, handle, indent=2, default=str)
    print("[axiscal] %s" % summary["ended"], flush=True)
    return summary


def _checkpoint_only(run_dir, checkpoint):
    """Persist the checkpoint without appending a segment."""
    with open(str(run_dir / CHECKPOINT_FILE), "w") as handle:
        json.dump(checkpoint, handle, indent=2)


def main(argv=None):
    """Command line entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="Fly the experiment, resuming any unfinished run.")
    parser.add_argument("--fit", metavar="RUN_DIR",
                        help="Fit an existing run and write calibration.json/.md.")
    parser.add_argument("--blocks", default="i,ii,iii",
                        help="Comma-separated blocks to fly (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and its duration; command nothing.")
    parser.add_argument("--min-battery", type=float,
                        default=bringup.MIN_FLIGHT_BATTERY,
                        help="Battery floor, 0-1 (default: %(default)s). The "
                             "campaign's pre-flight gate is also a data-quality "
                             "gate, so lowering it trades sample count for "
                             "sample quality.")
    args = parser.parse_args(argv)
    blocks = [part.strip() for part in args.blocks.split(",") if part.strip()]

    if args.dry_run:
        dry_run(blocks)
        return 0
    if args.fit:
        print(json.dumps(fitting.fit(args.fit)["recommended"], indent=2))
        return 0
    if args.run:
        return 0 if run(blocks, args.min_battery)["ended"] == "completed" else 1
    parser.error("choose one of --run, --fit RUN_DIR or --dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
