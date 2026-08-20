"""One full campaign cycle: restart everything, fly, log, land, analyse.

This is the unit the outer supervisor loop repeats forever. It is written so that
*every* exit path lands the aircraft and writes whatever data it managed to
collect -- a cycle that crashes half way through must still leave a run folder
the next iteration can learn from, because nobody is watching.

The order of the flight itself is not negotiable and is documented in
``falcon_flight_sequence.py``: the twist control adapter must be stopped during
takeoff (its 20 Hz stop-watchdog cancels a climb within ~50 ms) and exploration
only works from a settled hover (``exploration_node`` has no route from a ground
pose and spins logging ``[FSM] Plan fail`` instead of failing loudly).

Usage::

    python3 -m sparx_agency.tools.falcon_campaign.campaign --duration 600
"""
from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import time
import traceback

from sparx_agency.tools.falcon_campaign import bringup, config as C
from sparx_agency.tools import falcon_flight_sequence as seq

#: Logs copied into every run folder. Container logs first, host logs second.
_IT_LOGS = ["rooster_command_unit_%s.log" % C.DRONE_ID, "rooster_gtl.log"]
_HOST_LOGS = [
    "/tmp/rooster_twist_control_%s.log" % C.DRONE_ID,
    "/tmp/falcon_roslaunch.log",
    "/tmp/falcon_container.log",
    "/tmp/rooster_depth_processor.log",
    "/tmp/rooster_frame_capture.log",
    "/tmp/bridge.log",
]


def log(run_dir, message):
    """Append a timestamped line to the run's own journal and to stdout."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = "[%s] %s" % (stamp, message)
    print(line, flush=True)
    try:
        with open(str(run_dir / "cycle.log"), "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def new_run_dir():
    """Create and return a fresh timestamped run folder."""
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    run = C.RUNS_DIR / stamp
    (run / "logs").mkdir(parents=True, exist_ok=True)
    return run


def start_clearance_probe():
    """Trace clearance to the nearest mapped obstacle for the whole flight.

    Collisions are half the mission's goal and nothing measured them: the
    aircraft was found flying inside its own safety margin 66 % of the time, and
    its REFERENCE 73 %, which is what showed the fault was in the plan rather
    than in the tracking. That took a hand-run probe and a container that had to
    survive long enough to copy the file off, so it now runs every flight.

    Failure here must never cost a cycle -- it is instrumentation, not flight.
    """
    bringup.sh(
        "docker cp %s %s:/tmp/probe_clearance_trace.py 2>/dev/null"
        % (shlex.quote(str(C.REPO_ROOT / "sparx_agency" / "tools" /
                          "falcon_campaign" / "probe_clearance_trace.py")),
           C.FALCON_CONTAINER), 30)
    bringup.sh(
        "docker exec -d %s bash -lc %s"
        % (C.FALCON_CONTAINER,
           shlex.quote(C.FALCON_ENV + "python3 -u /tmp/probe_clearance_trace.py "
                       "> /tmp/clearance.jsonl 2>/tmp/clearance.err")), 30)


def start_recorder(run_dir, duration_s):
    """Launch the flight recorder inside the vendor container, detached.

    Runs in ``it`` rather than ``robotican_dev``: only ``it`` has the vendor
    message definitions for ground truth and the rangefinder. See
    ``config.RECORDER_DIR_IN_IT`` for why the output is copied out afterwards
    instead of written straight into the run folder.
    """
    bringup.kill_in(C.IT_CONTAINER, "falcon_campaign.recorder")
    subprocess.run(["docker", "exec", C.IT_CONTAINER, "bash", "-lc",
                    "rm -rf %s && mkdir -p %s"
                    % (C.RECORDER_DIR_IN_IT, C.RECORDER_DIR_IN_IT)],
                   capture_output=True, timeout=30)
    cmd = (C.IT_ENV +
           "python3 -m sparx_agency.tools.falcon_campaign.recorder "
           "--run-dir %s --rooster-id %s --duration-sec %d"
           % (C.RECORDER_DIR_IN_IT, C.DRONE_ID, int(duration_s) + 30))
    subprocess.run(
        ["docker", "exec", "-d", C.IT_CONTAINER, "bash", "-lc",
         cmd + " > /tmp/campaign_recorder.log 2>&1"],
        capture_output=True, timeout=30)


def stop_recorder():
    """Ask the recorder to finish its current line and write its metadata."""
    bringup.kill_in(C.IT_CONTAINER, "falcon_campaign.recorder")
    time.sleep(2)


def collect_logs(run_dir, recorded):
    """Copy every log a post-mortem might need into the run folder.

    Args:
        run_dir: This cycle's folder.
        recorded: Whether THIS cycle actually started the flight recorder. The
            recorder writes inside the vendor container and is only wiped when
            it starts, so a cycle that never got that far would otherwise copy
            out the PREVIOUS flight's telemetry and be analysed as if it had
            flown it. Two takeoff failures did exactly that, reporting the last
            good flight's 336 m and 12309 samples as their own.
    """
    dest = run_dir / "logs"
    for name in _IT_LOGS:
        bringup.sh("docker cp %s:/tmp/%s %s/ 2>/dev/null"
                   % (C.IT_CONTAINER, name, shlex.quote(str(dest))), 60)
    for path in _HOST_LOGS:
        bringup.sh("cp %s %s/ 2>/dev/null" % (shlex.quote(path), shlex.quote(str(dest))), 30)
    # The recorder's own output, which lives inside `it` (see config).
    if recorded:
        bringup.sh("docker cp %s:%s/. %s/ 2>/dev/null"
                   % (C.IT_CONTAINER, C.RECORDER_DIR_IN_IT,
                      shlex.quote(str(run_dir))), 90)
        bringup.sh("docker cp %s:/tmp/campaign_recorder.log %s/ 2>/dev/null"
                   % (C.IT_CONTAINER, shlex.quote(str(dest))), 30)
    else:
        log(run_dir, "no recorder ran this cycle -- not copying telemetry, it "
                     "would be the previous flight's")
    bringup.sh("docker cp %s:/tmp/clearance.jsonl %s/ 2>/dev/null"
               % (C.FALCON_CONTAINER, shlex.quote(str(run_dir))), 30)
    # The follower/FSM logs live in falcon's rotating roslaunch log dir.
    #
    # rosout* is excluded on purpose. It is the master's AGGREGATE of the very
    # node output already captured here and in falcon_roslaunch.log, so the
    # analyzer counted every FSM line twice -- and because rosout rotates
    # mid-flight, how MUCH got double-counted varied per run, which is the worst
    # kind of noise to have in a number used to compare runs. It was also 264 MB
    # of the 289 MB each run wrote, and 202 MB of that (rosout.log.1/.2) was
    # never read by anything, since the analyzer globs *.log.
    bringup.sh(
        "docker exec %s bash -lc 'L=$(ls -td /root/.ros/log/*/ | head -1); "
        "tar -C \"$L\" --exclude=\"rosout*\" -cf - . 2>/dev/null' "
        "| tar -C %s/ -xf - 2>/dev/null"
        % (C.FALCON_CONTAINER, shlex.quote(str(dest))), 120)


def land_and_disarm(run_dir):
    """Bring the aircraft down. Safe to call from any state, including twice."""
    bringup.stop_twist_adapter()
    for action in ("stop", "land"):
        err = seq.send_cmd_nav(action, C.DRONE_ID)
        if err:
            log(run_dir, "cmd_nav %s failed: %s" % (action, err))
        time.sleep(2)
    time.sleep(20)
    seq.send_cmd_nav("disarm", C.DRONE_ID)


def sample_coverage(run_dir):
    """Append one coverage sample to ``coverage.jsonl``.

    Coverage is the mission's actual goal, and until now nothing recorded it --
    every metric the campaign optimised (smoothness, tracking error, stops) was
    a proxy. FALCON publishes it on a ROS 1 topic inside the falcon container,
    so it is sampled by exec rather than by the ROS 2 recorder. A plateau in
    this number is the honest signal that exploration has stalled, whatever the
    aircraft looks like it is doing.
    """
    cmd = (C.FALCON_ENV +
           "timeout 6 rostopic echo -n1 /voxel_mapping/map_coverage 2>/dev/null "
           "| head -1; timeout 6 rostopic echo -n1 "
           "/planning_vis/frontier_pcl/width 2>/dev/null | head -1")
    r = bringup.sh("docker exec %s bash -lc %s" % (C.FALCON_CONTAINER, shlex.quote(cmd)), 30)
    numbers = []
    for line in r.stdout.splitlines():
        token = line.replace("data:", "").strip()
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if not numbers:
        # Record the gap instead of returning silently. rostopic echo -n1 blocks
        # until a message arrives, and /voxel_mapping/map_coverage goes quiet
        # while the FSM sits in FINISH -- so a run that re-opens repeatedly logs
        # a sparse trace, and a rate computed over it is not comparable with one
        # sampled across a whole flight. A run once reported a record 90.7
        # m3/min from six samples spanning 110s of a 600s flight.
        try:
            with open(str(run_dir / "coverage.jsonl"), "a") as fh:
                fh.write(json.dumps({"wall": round(time.time(), 3),
                                     "ok": False}) + "\n")
        except OSError:
            pass
        return
    row = {"wall": round(time.time(), 3), "coverage_m3": numbers[0], "ok": True}
    if len(numbers) > 1:
        row["frontier_points"] = int(numbers[1])
    try:
        with open(str(run_dir / "coverage.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def liveness_check(run_dir, tick):
    """Cheap mid-flight checks. Returns a reason to abort, or None.

    Deliberately narrow: only conditions that make the rest of the flight
    worthless are worth ending a run over. A stalled aircraft still produces
    useful data about why it stalled.
    """
    if not bringup.container_up(C.DRONE_CONTAINER):
        return "the drone container disappeared (Sphera died)"
    if not bringup.container_up(C.FALCON_CONTAINER):
        return "the falcon container died"
    if tick % 4 == 0 and not bringup.frames_fresh(max_age_s=25):
        return "camera frames went stale (video_trigger froze)"
    if tick % 4 == 0 and not bringup.rosnode_exists("/exploration_node"):
        return "exploration_node died"
    return None


def fly(run_dir, duration_s):
    """Arm, take off, hand over to FALCON, and hold for the flight window.

    Returns:
        A dict describing how the flight went, including why it ended.
    """
    result = {"armed": False, "hover": None, "ended": "completed",
              "flight_seconds": 0.0, "recorded": False}

    # The FCU connects a little after its container does, and arming into that
    # gap fails outright ("Not connected to FCU") -- then the hover wait burns
    # its whole 60 s timeout before the cycle gives up. Waiting costs seconds.
    if not bringup.wait_for_armable():
        result["ended"] = "FCU never became armable"
        return result

    # Back-to-back with no gap: a ~30 s pause between them has silently
    # disarmed the aircraft and forced an internal re-arm inside takeoff().
    log(run_dir, "arm + takeoff")
    err = seq.send_cmd_nav("arm", C.DRONE_ID)
    if err:
        result["ended"] = "arm failed: %s" % err
        return result
    err = seq.send_cmd_nav("takeoff", C.DRONE_ID)
    if err:
        result["ended"] = "takeoff failed: %s" % err
        return result
    result["armed"] = True

    ok, message = seq.wait_for_stable_hover(
        C.DRONE_ID, timeout=C.HOVER_SETTLE_TIMEOUT_S)
    result["hover"] = message
    log(run_dir, "hover: %s" % message)
    if not ok:
        result["ended"] = "hover never settled: %s" % message
        return result

    log(run_dir, "hover settled -- starting recorder and handing over to FALCON")
    start_recorder(run_dir, duration_s)
    start_clearance_probe()
    result["recorded"] = True
    time.sleep(2)
    bringup.start_twist_adapter()

    started = time.time()
    tick = 0
    while time.time() - started < duration_s:
        time.sleep(15)
        tick += 1
        sample_coverage(run_dir)
        reason = liveness_check(run_dir, tick)
        if reason:
            result["ended"] = "aborted: %s" % reason
            log(run_dir, "ABORT: %s" % reason)
            break
    result["flight_seconds"] = round(time.time() - started, 1)
    return result


def run_cycle(duration_s=None, follower=None):
    """Do one complete cycle and return its summary dict.

    Never raises: a failure is recorded in the run folder and reported back so
    the supervisor can decide whether to continue, because there is nobody to
    ask.
    """
    duration_s = C.FLIGHT_SECONDS if duration_s is None else duration_s
    run_dir = new_run_dir()
    summary = {"run_dir": str(run_dir), "started": time.time(),
               "follower": follower or C.EXPLORATION_FOLLOWER,
               "duration_requested_s": duration_s}
    log(run_dir, "=== cycle start: %s ===" % run_dir.name)

    try:
        log(run_dir, "bringing up the stack")
        summary["health"] = bringup.full_bringup(follower=follower)
        log(run_dir, "health: %s" % json.dumps(summary["health"], default=str))
        if not summary["health"].get("ok"):
            summary["ended"] = "unhealthy stack; flew nothing"
            return _finish(run_dir, summary)

        summary["flight"] = fly(run_dir, duration_s)
        summary["ended"] = summary["flight"]["ended"]
    except Exception as exc:                       # noqa: BLE001 -- unattended
        summary["ended"] = "exception: %s: %s" % (type(exc).__name__, exc)
        summary["traceback"] = traceback.format_exc()
        log(run_dir, "EXCEPTION:\n%s" % summary["traceback"])
    finally:
        try:
            stop_recorder()
            land_and_disarm(run_dir)
            collect_logs(run_dir, (summary.get("flight") or {}).get("recorded", False))
        except Exception as exc:                   # noqa: BLE001
            log(run_dir, "teardown problem: %s" % exc)
    return _finish(run_dir, summary)


def _finish(run_dir, summary):
    """Analyse the run, write the summary, and return it."""
    summary["finished"] = time.time()
    try:
        from sparx_agency.tools.falcon_campaign import analyze
        summary["metrics"] = analyze.analyze(run_dir)
        log(run_dir, "analysis written")
    except Exception as exc:                       # noqa: BLE001
        summary["metrics"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        log(run_dir, "analysis failed: %s" % exc)
    try:
        with open(str(run_dir / "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
    except OSError as exc:
        log(run_dir, "could not write summary.json: %s" % exc)
    log(run_dir, "=== cycle end: %s ===" % summary.get("ended"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=C.FLIGHT_SECONDS,
                        help="Flight window in seconds (default: %(default)s).")
    parser.add_argument("--follower", default=None,
                        choices=["reference", "bspline"],
                        help="Which exploration follower to fly.")
    args = parser.parse_args()
    summary = run_cycle(args.duration, args.follower)
    print(json.dumps({k: v for k, v in summary.items() if k != "traceback"},
                     indent=2, default=str))
    return 0 if summary.get("ended") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
