#!/usr/bin/env python3
"""Digest one campaign run directory into a diagnosis.

Reads what campaign_run.sh left behind and prints a compact report that
separates the three failure families the campaign is hunting:

* control        -- the follower tracked its reference badly (gap/cross/yaw
                    percentiles from tracking.csv, saturation fraction);
* localization   -- the pose the planner used was late or wrong (here the sim
                    odometry is ground truth, so this shows up as bridge lag /
                    real-time-factor skew, not noise);
* planning       -- FALCON itself: respawns, No-path loops, plan cadence,
                    time to finish.

Pure stdlib; runs on the host venv or bare python3.

    ./analyze_run.py /tmp/falcon_sjtu/campaign/003_hospital_baseline
"""
import json
import math
import os
import sys


def _percentiles(vals, ps=(50, 90, 99)):
    if not vals:
        return {p: float("nan") for p in ps}
    s = sorted(vals)
    return {p: s[min(len(s) - 1, int(len(s) * p / 100.0))] for p in ps}


def load_trace(path):
    rows, events = [], []
    if not os.path.exists(path):
        return rows, events
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            (events if "event" in d else rows).append(d)
    return rows, events


def load_tracking(path):
    """tracking.csv from `rostopic echo -p`: header %time,field0..field12."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = f.readline()
        if not header.startswith("%"):
            return rows
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 14:
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    return rows


def main():
    run_dir = sys.argv[1]
    verdict = {}
    vpath = os.path.join(run_dir, "verdict.json")
    if os.path.exists(vpath):
        with open(vpath) as f:
            verdict = json.load(f)
    print("== %s ==" % run_dir)
    print("verdict: %s (%s)  elapsed %ss  contacts=%s wedges=%s respawns=%s"
          % (verdict.get("verdict"), verdict.get("detail"), verdict.get("elapsed_s"),
             verdict.get("contacts"), verdict.get("wedges"),
             verdict.get("planner_respawns")))
    if verdict.get("finish_line"):
        print("finish:  %s" % verdict["finish_line"])

    # ── physics side ────────────────────────────────────────────────────
    rows, events = load_trace(os.path.join(run_dir, "trace.jsonl"))
    if rows:
        top_speed = max(r.get("speed", 0.0) for r in rows)
        max_tilt = max(max(abs(r.get("roll_deg", 0)), abs(r.get("pitch_deg", 0)))
                       for r in rows)
        xs = [r["x"] for r in rows]; ys = [r["y"] for r in rows]
        print("trace:   %d samples over %.0fs; top speed %.2f m/s; max tilt %.0f deg; "
              "x[%.1f,%.1f] y[%.1f,%.1f]"
              % (len(rows), rows[-1]["t"] - rows[0]["t"], top_speed, max_tilt,
                 min(xs), max(xs), min(ys), max(ys)))
    for e in events:
        print("event:   t=%7.1fs %-8s %s" % (e["t"], e["event"], e.get("detail", "")))
        # context: speed and tilt just before the event
        pre = [r for r in rows if 0 <= e["t"] - r["t"] <= 2.0]
        if pre:
            print("         2s prior: speed %.2f m/s, tilt %.0f deg, pos (%.1f, %.1f, %.2f)"
                  % (pre[-1].get("speed", 0),
                     max(abs(pre[-1].get("roll_deg", 0)), abs(pre[-1].get("pitch_deg", 0))),
                     pre[-1]["x"], pre[-1]["y"], pre[-1]["z"]))

    # ── control side ────────────────────────────────────────────────────
    trk = load_tracking(os.path.join(run_dir, "tracking.csv"))
    if trk:
        # `rostopic echo -p` on Float32MultiArray: %time, layout.data_offset,
        # THEN data0.. -- so every documented field sits one column later:
        # r[2]=gap r[3]=along r[4]=cross r[5]=yaw_err r[6..8]=world_v
        # r[9]=yaw_rate r[10]=traj_id r[11]=ref_t r[12]=saturated r[13]=holding
        # r[14]=past_end
        FOLLOW = [r for r in trk if len(r) >= 15 and r[13] < 0.5 and r[14] < 0.5]
        gaps = [r[2] for r in FOLLOW]; cross = [abs(r[4]) for r in FOLLOW]
        yawe = [abs(r[5]) for r in FOLLOW]
        sat = sum(1 for r in FOLLOW if r[12] > 0.5)
        pg, pc, py = _percentiles(gaps), _percentiles(cross), _percentiles(yawe)
        print("control: %d ticks following; gap p50/p90/p99 = %.2f/%.2f/%.2f m; "
              "|cross| %.2f/%.2f/%.2f m; |yaw| %.0f/%.0f/%.0f deg; saturated %.0f%%"
              % (len(FOLLOW), pg[50], pg[90], pg[99], pc[50], pc[90], pc[99],
                 math.degrees(py[50]), math.degrees(py[90]), math.degrees(py[99]),
                 100.0 * sat / max(1, len(FOLLOW))))
        worst = sorted(FOLLOW, key=lambda r: -r[2])[:3]
        for w in worst:
            print("         worst gap %.2f m (along %+.2f, cross %+.2f) at ref_t %.1f traj %d"
                  % (w[2], w[3], w[4], w[11], int(w[10])))

    # ── sim health ──────────────────────────────────────────────────────
    rtf_path = os.path.join(run_dir, "rtf.log")
    if os.path.exists(rtf_path):
        factors = []
        for line in open(rtf_path):
            # gz stats -p lines: "Factor[0.88] SimTime[..] RealTime[..] ..."
            for tok in line.replace("[", " ").replace("]", " ").split():
                try:
                    v = float(tok)
                    if 0.05 <= v <= 1.5:
                        factors.append(v)
                    break
                except ValueError:
                    break
        if factors:
            print("rtf:     mean %.2f  min %.2f  (%d samples)"
                  % (sum(factors) / len(factors), min(factors), len(factors)))

    # ── planner side ────────────────────────────────────────────────────
    fl = os.path.join(run_dir, "falcon.log")
    if os.path.exists(fl):
        died = no_path = bsplines = 0
        with open(fl, errors="replace") as f:
            for line in f:
                if "process has died" in line:
                    died += 1
                if "No path to next viewpoint" in line:
                    no_path += 1
                if "[FSM] Exploration finished. Start" in line:
                    print("planner: %s" % line.strip()[line.find("[FSM]"):])
        print("planner: %d process deaths, %d no-path lines" % (died, no_path))


if __name__ == "__main__":
    main()
