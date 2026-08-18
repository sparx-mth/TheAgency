#!/usr/bin/env python3
"""Automated ManualControl axis-calibration flight for the Rooster in Sphera.

MISSION.md P5 says the axis curve is still guessed: the standing-start
breakaway was never measured per sign, the moving-regime gain never at all, yaw
has no calibrated inverse, and nothing models two axes commanded at once. Every
one of those gaps is filled today by a constant written down after two data
points. This tool measures them; the fit lives in
``rooster_axis_calibration_fit``.

It publishes ``{"action": "move", "axes": {...}}`` to ``/<drone>/cmd_nav`` and
``{"action": "stop"}`` between segments. ``rooster_command_unit`` holds the axes
until they change and never takes ``z`` from a ``move`` payload, so altitude
hold keeps flying the aircraft throughout and no segment can drop it. Nothing
here ever sends ``up``/``down`` alongside a move.

It must own the aircraft alone: the twist control adapter publishes ``stop`` at
20 Hz whenever ``/cmd_vel`` is quiet, which would cancel every segment within
~50 ms, and a second publisher on ``/R1/manual_control`` has silently destroyed
a full day of measurements before (LESSONS.md 2026-08-17). Both are asserted
before the first segment, never assumed.

It checkpoints because the sweep outlives one Sphera battery, and
``MIN_FLIGHT_BATTERY`` is a data-quality gate as much as a safety one -- below
it a "forward" command produces mostly lateral motion. Hitting the floor lands
the run cleanly; the next invocation resumes at the first unflown segment.

Usage::

    python3 -m sparx_agency.tools.rooster_axis_calibration --dry-run
    python3 -m sparx_agency.tools.rooster_axis_calibration --run --blocks i,ii
    python3 -m sparx_agency.tools.rooster_axis_calibration --fit runs/axiscal_X

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
from sparx_agency.tools import rooster_axis_calibration_plan as design
from sparx_agency.tools.falcon_campaign import bringup, campaign, config as C

#: Roll/pitch magnitude that ends a segment. Signs on ``/<drone>/attitude_rpy``
#: are UNVERIFIED -- magnitude only, never a signed comparison.
TILT_LIMIT_DEG = 25.0
RANGER_MIN_M, RANGER_MAX_M = 0.8, 3.0

#: A feedback stream older than this is not evidence of anything.
MAX_FEEDBACK_AGE_S = 1.0

#: A standing start is only a standing start below this measured speed.
STANDING_SPEED_MPS = 0.03

POLL_S = 0.3

RUN_PREFIX = "axiscal_"
SEGMENTS_FILE = "segments.jsonl"
CHECKPOINT_FILE = "checkpoint.json"


class CalibrationAbort(RuntimeError):
    """A precondition or a command failed; nothing more may be flown."""


class BatteryExhausted(RuntimeError):
    """The pack hit the floor mid-run; land, checkpoint, resume next battery."""


# ── commanding ───────────────────────────────────────────────────────────
def _publish(payload, timeout=15.0):
    """Publish one ``cmd_nav`` payload the way ``falcon_flight_sequence`` does.

    Raises:
        CalibrationAbort: If it fails -- a segment commanded into a void would
            otherwise be recorded as a measurement of zero.
    """
    data = json.dumps({"data": json.dumps(payload)})
    cmd = (C.IT_ENV + "ros2 topic pub -1 /%s/cmd_nav std_msgs/msg/String %s"
           % (C.DRONE_ID, shlex.quote(data)))
    try:
        done = subprocess.run(["docker", "exec", C.IT_CONTAINER, "bash", "-c", cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise CalibrationAbort("cmd_nav %r timed out after %.0fs" % (payload, timeout))
    if done.returncode != 0:
        raise CalibrationAbort("cmd_nav %r failed: %s" % (payload, done.stderr.strip()[:200]))


def _stop():
    """Zero x/y/r, leaving z (throttle / altitude hold) untouched."""
    _publish({"action": "stop"})


def _move(axes):
    """Hold one axis triple until the next ``move`` or ``stop``."""
    _publish({"action": "move", "axes": axes})


# ── telemetry ────────────────────────────────────────────────────────────
def _tail_truth():
    """Newest flight-recorder sample from inside ``it``, or ``None``.

    The 20 Hz recorder already writes every stream the monitor needs, and
    ``ros2 topic echo`` piped into a short ``timeout`` block-buffers and hangs
    (LESSONS.md 2026-08-18), so this tails its last line instead.
    """
    try:
        done = subprocess.run(
            ["docker", "exec", C.IT_CONTAINER, "tail", "-n", "1",
             C.RECORDER_DIR_IN_IT + "/truth.jsonl"],
            capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None
    line = done.stdout.strip()
    if done.returncode != 0 or not line:
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None               # a torn last line; the next poll is whole


def _fresh(sample, stream):
    """One stream's payload if it is fresh enough to believe, else ``None``."""
    entry = (sample or {}).get(stream) or {}
    age = entry.get("age")
    return entry if age is not None and age <= MAX_FEEDBACK_AGE_S else None


def _speed_xy(sample):
    """Measured horizontal speed, or ``None`` when the feedback is stale."""
    velocity = _fresh(sample, "velocity") or {}
    if velocity.get("vx") is None or velocity.get("vy") is None:
        return None
    return math.hypot(velocity["vx"], velocity["vy"])


def _battery(sample):
    """Battery as a 0-1 fraction, or ``None`` when the state stream is stale."""
    level = (_fresh(sample, "state") or {}).get("battery")
    if level is None:
        return None
    return level if level <= 1.0 else level / 100.0


def _safety_trip(sample, min_battery):
    """Check every continuous limit against one sample.

    Returns:
        ``(reason, fatal)``; ``reason`` is ``None`` while everything is inside
        its limit, and ``fatal`` means the whole run must land, not just this
        segment.
    """
    if sample is None:
        return "the flight recorder produced no telemetry", False
    attitude = _fresh(sample, "attitude")
    if not attitude:
        return "attitude feedback stale", False
    for name in ("roll", "pitch"):
        if attitude.get(name) is None:
            return "no %s on the attitude stream" % name, False
        degrees = abs(math.degrees(attitude[name]))
        if degrees > TILT_LIMIT_DEG:
            return "|%s| %.0f deg over the %.0f deg limit" % (
                name, degrees, TILT_LIMIT_DEG), False
    ranger = (_fresh(sample, "state") or {}).get("ranger")
    if ranger is None or not RANGER_MIN_M <= ranger <= RANGER_MAX_M:
        return "ranger %s outside [%.1f, %.1f] m" % (
            ranger, RANGER_MIN_M, RANGER_MAX_M), False
    if _speed_xy(sample) is None:
        return "velocity feedback stale", False
    level = _battery(sample)
    if level is not None and level < min_battery:
        return "battery %.0f%% below the %.0f%% floor" % (
            level * 100.0, min_battery * 100.0), True
    return None, False


# ── flying it ────────────────────────────────────────────────────────────
def _monitor(seconds, min_battery):
    """Hold for a while, polling every limit. Returns ``(reason, fatal)``."""
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
            return "never came to rest: %.3f m/s still measured" % (
                speed if speed is not None else float("nan")), False
        time.sleep(POLL_S)


def _run_segment(segment, min_battery):
    """Fly one segment and return its record.

    Raises:
        BatteryExhausted: On the battery floor, so the caller lands and
            checkpoints instead of measuring on a corrupted pack.
    """
    label, axes, hold_s, settle_s = segment
    design.assert_within_limits(label, axes)
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
        record["aborted"], record["reason"] = True, reason
        if fatal:
            raise BatteryExhausted(reason)
    return record


# ── run bookkeeping ──────────────────────────────────────────────────────
def _write_json(path, payload):
    """Write one indented JSON file, replacing whatever was there."""
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _resolve_run_dir():
    """Resume the newest unfinished run, or start a new one.

    Resuming is the normal case: the sweep outlives the battery, so a fresh
    directory per invocation would restart the experiment on every Sphera
    restart.

    Returns:
        ``(run_dir, checkpoint)``.
    """
    for path in sorted(C.RUNS_DIR.glob(RUN_PREFIX + "*"), reverse=True):
        marker = path / CHECKPOINT_FILE
        if not marker.exists():
            continue
        with open(str(marker)) as handle:
            checkpoint = json.load(handle)
        if not checkpoint.get("complete"):
            return path, checkpoint
    run_dir = C.RUNS_DIR / (RUN_PREFIX + time.strftime("%Y%m%d_%H%M%SZ", time.gmtime()))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, {"done": [], "complete": False}


def _checkpoint(run_dir, checkpoint, record):
    """Append one finished segment to the log and to the checkpoint."""
    with open(str(run_dir / SEGMENTS_FILE), "a") as handle:
        handle.write(json.dumps(record) + "\n")
    checkpoint["done"].append(record["label"])
    checkpoint["updated"] = time.time()
    _write_json(run_dir / CHECKPOINT_FILE, checkpoint)


def _collect(run_dir):
    """Bring this session's recorder output into the run directory.

    Appended, not copied: ``start_recorder`` wipes the in-container directory
    each session, so a plain ``docker cp`` would overwrite everything the
    previous battery's segments recorded.
    """
    bringup.sh("docker exec %s cat %s/truth.jsonl >> %s"
               % (C.IT_CONTAINER, C.RECORDER_DIR_IN_IT,
                  shlex.quote(str(run_dir / fitting.TRUTH_FILE))), 180)
    index = len(list(run_dir.glob("recorder_meta_*.json")))
    bringup.sh("docker cp %s:%s/recorder_meta.json %s"
               % (C.IT_CONTAINER, C.RECORDER_DIR_IN_IT,
                  shlex.quote(str(run_dir / ("recorder_meta_%d.json" % index)))), 60)


def _preflight(min_battery):
    """Assert everything that makes a measurement worth taking, then hover.

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
        raise CalibrationAbort("battery unreadable -- an unverifiable pack "
                               "invalidates every speed this flight would measure")
    if level < min_battery:
        raise CalibrationAbort("battery %.0f%% below the %.0f%% floor"
                               % (level * 100.0, min_battery * 100.0))
    for action in ("arm", "takeoff"):
        error = seq.send_cmd_nav(action, C.DRONE_ID)
        if error:
            raise CalibrationAbort("%s failed: %s" % (action, error))
    settled, message = seq.wait_for_stable_hover(C.DRONE_ID, timeout=C.HOVER_SETTLE_TIMEOUT_S)
    if not settled:
        raise CalibrationAbort("never reached a settled hover: %s" % message)
    return {"drone_image": image, "battery": level, "manual_authority": detail,
            "hover": message}


def _land():
    """Bring the aircraft down. Safe from any state, including twice."""
    for action in ("stop", "land"):
        seq.send_cmd_nav(action, C.DRONE_ID)
        time.sleep(2)
    time.sleep(20)
    seq.send_cmd_nav("disarm", C.DRONE_ID)


#: Battery floor for ABORTING a sweep in progress, as opposed to starting one.
#:
#: These are deliberately different numbers. Starting needs a near-full pack
#: (bringup.MIN_FLIGHT_BATTERY) because a sweep that dies a third of the way
#: through wastes a Sphera restart. Aborting only needs the point past which the
#: samples stop being trustworthy -- measured below ~25% a "forward" command
#: produces mostly lateral motion as thrust authority runs out (LESSONS.md).
#: Using the start threshold as the abort floor would end every sweep after
#: about a minute of flying and need ~20 Sphera restarts to finish.
ABORT_BATTERY_FRACTION = 0.25


def run(blocks, min_battery, abort_battery=ABORT_BATTERY_FRACTION):
    """Fly the experiment, resuming whatever a previous battery left unflown.

    Returns:
        A summary dict, also written to ``session.json`` in the run directory.
    """
    # Before anything at all: the teardown below lands and disarms on every
    # exit path, so "is this the simulator?" cannot live inside the try.
    bringup.assert_simulator()
    plan = design.build_plan(blocks)
    run_dir, checkpoint = _resolve_run_dir()
    done = set(checkpoint.get("done", []))
    remaining = [segment for segment in plan if segment[0] not in done]
    summary = {"run_dir": str(run_dir), "planned": len(plan), "flown": 0,
               "already_done": len(plan) - len(remaining), "aborted_segments": 0,
               "ended": "completed"}
    print("[axiscal] %s -- %d of %d segments left" % (run_dir, len(remaining), len(plan)),
          flush=True)
    if not remaining:
        checkpoint["complete"] = True
        _write_json(run_dir / CHECKPOINT_FILE, checkpoint)
        return summary
    recording = False
    try:
        summary["preflight"] = _preflight(min_battery)
        campaign.start_recorder(run_dir, design.estimate_seconds(remaining))
        recording = True
        time.sleep(3)
        for segment in remaining:
            record = _run_segment(segment, abort_battery)
            _checkpoint(run_dir, checkpoint, record)
            summary["flown"] += 1
            summary["aborted_segments"] += int(record["aborted"])
            print("[axiscal] %-22s %s" % (record["label"], record["reason"] or "ok"),
                  flush=True)
        checkpoint["complete"] = True
    except BatteryExhausted as exc:
        summary["ended"] = "battery: %s -- restart Sphera and re-run to resume" % exc
    except (CalibrationAbort, ValueError, bringup.BringupError) as exc:
        summary["ended"] = "aborted: %s" % exc
    finally:
        try:
            _stop()
        except CalibrationAbort:
            pass
        campaign.stop_recorder()
        _land()
        # Only ours to collect: /tmp/campaign_run may still hold another run's
        # telemetry if this one never got as far as starting the recorder.
        if recording:
            _collect(run_dir)
        _write_json(run_dir / CHECKPOINT_FILE, checkpoint)
        _write_json(run_dir / "session.json", summary)
    print("[axiscal] %s" % summary["ended"], flush=True)
    return summary


def main(argv=None):
    """Command line entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="Fly the experiment, resuming any unfinished run.")
    parser.add_argument("--fit", metavar="RUN_DIR",
                        help="Fit an existing run; writes calibration.json/.md.")
    parser.add_argument("--blocks", default="i,ii,iii",
                        help="Comma-separated blocks to fly (default: %(default)s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and its duration; command nothing.")
    parser.add_argument("--min-battery", type=float, default=bringup.MIN_FLIGHT_BATTERY,
                        help="Battery floor, 0-1 (default: %(default)s). It is a "
                             "data-quality gate as well as a safety one, so lowering "
                             "it trades sample quality for sample count.")
    parser.add_argument("--abort-battery", type=float, default=ABORT_BATTERY_FRACTION,
                        help="Battery floor for aborting a sweep already in the "
                             "air, 0-1 (default: %(default)s). Separate from "
                             "--min-battery, which only gates STARTING one.")
    args = parser.parse_args(argv)
    blocks = [part.strip() for part in args.blocks.split(",") if part.strip()]
    if args.dry_run:
        design.describe(blocks)
        return 0
    if args.fit:
        print(json.dumps(fitting.fit(args.fit)["recommended"], indent=2))
        return 0
    if args.run:
        summary = run(blocks, args.min_battery, args.abort_battery)
        return 0 if summary["ended"] == "completed" else 1
    parser.error("choose one of --run, --fit RUN_DIR or --dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
