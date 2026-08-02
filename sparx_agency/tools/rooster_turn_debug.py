#!/usr/bin/env python3
"""rooster_turn_debug.py -- live dashboard for the "turns left when the path
asks for right" investigation, plus a one-button Falcon restart.

Reads (does not publish/control anything else):
  - it:/tmp/rooster_turn_debug/{commands.jsonl,pose.jsonl} (manual_flight_logger.py)
  - falcon:/tmp/path_logger_out/{paths.jsonl,goals.jsonl}  (path_logger.py)

Both loggers are read-only w.r.t. flight and must already be running (see
fly-rooster-sphera skill / this session's bring-up). This dashboard only
tails their output via `docker exec cat` on each refresh -- it does not
subscribe to ROS itself, so it has zero risk of interfering with flight.

Turn-direction check: takes the most recent A* path's first waypoint,
computes the bearing from the pose at that path's timestamp, and compares
the SIGN of the required turn against the actual sign of yaw change over the
next ~2s of pose samples. Convention (confirmed live this session, see
LESSONS.md "turn_right/turn_left was ~4x too low" entry and the
rooster_command_unit.py ACTION_MAP comment): increasing yaw = physical RIGHT
turn, decreasing yaw = LEFT, for this drone's /localization convention --
this is the OPPOSITE of the textbook ROS CCW-positive-is-left assumption,
verified against Sphera visually, not assumed.

This page also makes sure both loggers are actually running (it copies them
in from the repo and starts them if a check finds them missing), so opening
the dashboard is self-sufficient -- it doesn't depend on a prior manual
`docker cp`/launch from a chat session.

Run: streamlit run sparx_agency/tools/rooster_turn_debug.py
"""
import json
import math
import subprocess
import time

import streamlit as st

REPO = "/home/user1/GIT/TheAgency"
MANUAL_FLIGHT_LOGGER_HOST = f"{REPO}/sparx_agency/robots/ROBOTICAN/debug/manual_flight_logger.py"
PATH_LOGGER_HOST = f"{REPO}/sparx_agency/tasks/planning/falcon/debug/path_logger.py"
EXPLORATION_WATCHDOG_HOST = f"{REPO}/sparx_agency/tasks/planning/falcon/debug/exploration_watchdog.sh"

POSE_DIR = "/tmp/rooster_turn_debug"  # manual_flight_logger.py --out-dir, inside `it`
PATH_DIR = "/tmp/path_logger_out"     # path_logger.py --out-dir, inside `falcon`
REFRESH_SEC = 2

FALCON_LAUNCH_CMD = (
    "roslaunch falcon_adapter sphera_drone.launch map_name:=sphera_jail "
    "real_pose_topic:=/R1/localization real_depth_path_topic:=/R1/depth_frame_path "
    "real_rgb_path_topic:=/R1/rgb_frame_path "
    "cam_fx:=111.837662 cam_fy:=180.0 cam_cx:=269.5 cam_cy:=179.5 "
    "cam_width:=540 cam_height:=360 cam_min_depth:=0.45 "
    "sync_tolerance:=0.05 max_interp_gap:=0.12 "
    "goal_x:=54.75 goal_y:=-11.66 "
    "bev_xmin:=38.75 bev_ymin:=-30.66 bev_xmax:=70.75 bev_ymax:=1.34 "
    "apf_max_total_shift_m:=0.3 vel_x:=0.15 mx_lateral_speed_max:=0.15 mx_yaw_rate:=0.4 "
    "bev_t_on:=3.0 bev_occ_conf_full:=4.0 bev_min_wall_run:=4 yaw_rate:=1.8"
)


def run(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def tail_jsonl_from_container(container, path, n=400):
    res = run(f"docker exec {container} tail -n {n} {path}")
    rows = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def ensure_loggers_running():
    """Copy in + start manual_flight_logger.py / path_logger.py if either is
    missing. Cheap to call every rerun -- pgrep is near-instant, and the
    launch commands are no-ops when the process already exists."""
    running = run("docker exec it pgrep -f manual_flight_logger.py").returncode == 0
    if not running:
        run(f"docker cp {MANUAL_FLIGHT_LOGGER_HOST} it:/tmp/manual_flight_logger.py")
        run(
            "docker exec -d -e ROS_DOMAIN_ID=9 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "-e CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml it bash -lc \""
            "source /opt/ros/foxy/setup.bash && source /home/rooster/workspace/install/setup.bash && "
            f"python3 /tmp/manual_flight_logger.py --rooster-id R1 --out-dir {POSE_DIR} "
            "> /tmp/manual_flight_logger.log 2>&1\""
        )

    running = run("docker exec falcon pgrep -f path_logger.py").returncode == 0
    if not running:
        run(f"docker cp {PATH_LOGGER_HOST} falcon:/tmp/path_logger.py")
        run(
            "docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
            "source /catkin_ws/devel/setup.bash && python3 /tmp/path_logger.py "
            f"--out-dir {PATH_DIR} > /tmp/path_logger.log 2>&1'"
        )


def container_up(name):
    res = run(f"docker ps --filter name=^{name}$ --format '{{{{.Names}}}}'")
    return res.stdout.strip() == name


def restart_falcon(status):
    # run_falcon_sphera.sh does `docker run -it` (needed for the container's
    # own interactive tooling) -- it FAILS SILENTLY ("cannot attach stdin to
    # a TTY-enabled container") without a real pseudo-terminal, and a plain
    # backgrounded `nohup ... &` from subprocess.run gives it none. `script -qc`
    # fakes one, exactly like every manual bring-up this session used.
    steps = [
        ("Removing falcon + ros1_bridge", "docker rm -f falcon ros1_bridge", None),
        ("Starting falcon container",
         "cd /home/user1/GIT/TheAgency/sparx_agency/tasks/planning/falcon && "
         "script -qc './run_falcon_sphera.sh sphera_jail' /dev/null > /tmp/falcon.log 2>&1 & sleep 8",
         lambda: container_up("falcon")),
        ("Installing python3-requests/pil (stopgap)",
         "docker exec falcon bash -c 'apt-get update -qq && apt-get install -y python3-requests python3-pil'",
         None),
        ("Launching falcon_adapter",
         f"docker exec -d falcon bash -lc \"source /opt/ros/noetic/setup.bash && "
         f"source /catkin_ws/devel/setup.bash && {FALCON_LAUNCH_CMD} > /tmp/falcon_roslaunch.log 2>&1\" && sleep 6",
         None),
        ("Restarting ros1_bridge",
         "cd /home/user1/GIT/TheAgency/sparx_agency/tasks/planning/falcon/bridge && "
         "script -qc 'ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
         "CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh' "
         "/dev/null > /tmp/bridge.log 2>&1 & sleep 10",
         lambda: container_up("ros1_bridge")),
        ("Launching RViz",
         "docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
         "source /catkin_ws/devel/setup.bash && roslaunch exploration_manager rviz.launch "
         "> /tmp/rviz.log 2>&1'",
         None),
        ("Launching BEV click node",
         "docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
         "source /catkin_ws/devel/setup.bash && rosrun falcon_adapter bev_click_goal_node.py "
         "> /tmp/bev_click.log 2>&1'",
         None),
        ("Redeploying path_logger.py",
         f"docker cp {PATH_LOGGER_HOST} falcon:/tmp/path_logger.py && "
         "docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
         "source /catkin_ws/devel/setup.bash && python3 /tmp/path_logger.py "
         f"--out-dir {PATH_DIR} > /tmp/path_logger.log 2>&1'",
         None),
        ("Redeploying exploration_watchdog.sh",
         f"docker cp {EXPLORATION_WATCHDOG_HOST} falcon:/tmp/exploration_watchdog.sh && "
         "docker exec falcon chmod +x /tmp/exploration_watchdog.sh && "
         "docker exec -d falcon /tmp/exploration_watchdog.sh",
         None),
    ]
    for label, cmd, verify in steps:
        status.write(f"⏳ {label}...")
        res = run(cmd, timeout=40)
        ok = (res.returncode == 0 or "rm -f" in cmd) and (verify is None or verify())
        if ok:
            status.write(f"✅ {label}")
        else:
            status.write(f"⚠️ {label} — exit {res.returncode}: {res.stderr[:300]}")
            if verify is not None:
                status.write("   (container never came up -- stopping here, later steps would just fail too)")
                return
    status.write("**Done. Verify `controller=waypoint` and mapping_sync health before flying.**")


def proc_running(location, pattern):
    prefix = "" if location == "host" else f"docker exec {location} "
    return run(f"{prefix}pgrep -f '{pattern}'").returncode == 0


def tail_log(location, path, n=15):
    if path is None:
        return None
    prefix = "" if location == "host" else f"docker exec {location} "
    res = run(f"{prefix}tail -n {n} {path}")
    return res.stdout if res.returncode == 0 else f"(no log at {path})"


# (display name, kind, check location, check pattern/name, log location, log path)
SERVICES = [
    ("R1 (Sphera drone)", "container", None, "R1", None, None),
    ("drone_simulator", "container", None, "drone_simulator", None, None),
    ("it (vendor container)", "container", None, "it", None, None),
    ("robotican_dev", "container", None, "robotican_dev", None, None),
    ("falcon", "container", None, "falcon", None, None),
    ("ros1_bridge", "container", None, "ros1_bridge", None, None),
    ("ground_truth_localization", "proc", "it", "rooster_ground_truth_localization", "host", "/tmp/rooster_gt_loc.log"),
    ("video_trigger.py", "proc", "it", "video_trigger.py", "it", "/tmp/video_trigger.log"),
    ("rooster_command_unit.py", "proc", "it", "rooster_command_unit.py", "it", "/tmp/rooster_command_unit_R1.log"),
    ("manual_flight_logger.py", "proc", "it", "manual_flight_logger.py", "it", "/tmp/manual_flight_logger.log"),
    ("rooster_frame_dir_publisher.py", "proc", "robotican_dev", "rooster_frame_dir_publisher.py", "host", "/tmp/rooster_frame_capture.log"),
    ("rooster_depth_processor.py", "proc", "robotican_dev", "rooster_depth_processor.py", "host", "/tmp/rooster_depth_processor.log"),
    ("rooster_twist_control_adapter", "proc", "robotican_dev", "rooster_twist_control_adapter", "host", "/tmp/rooster_twist_control.log"),
    ("video_freshness_watchdog.sh", "proc", "host", "video_freshness_watchdog.sh", "host", "/tmp/claude-1000/-home-user1-GIT-TheAgency/e1140fa8-92d8-40b3-9af5-d89820a30552/scratchpad/video_freshness_watchdog.log"),
    ("falcon_adapter (sphera_drone.launch)", "proc", "falcon", "sphera_drone.launch", "falcon", "/tmp/falcon_roslaunch.log"),
    ("RViz", "proc", "falcon", "rviz.launch", "falcon", "/tmp/rviz.log"),
    ("bev_click_goal_node.py", "proc", "falcon", "bev_click_goal_node.py", "falcon", "/tmp/bev_click.log"),
    ("exploration_watchdog.sh", "proc", "falcon", "exploration_watchdog.sh", "falcon", "/tmp/exploration_watchdog.log"),
    ("path_logger.py", "proc", "falcon", "path_logger.py", "falcon", "/tmp/path_logger.log"),
]


def bearing_deg(from_xy, to_xy):
    dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dy, dx))


def wrap180(deg):
    return (deg + 180) % 360 - 180


st.set_page_config(page_title="Rooster turn-direction debug", layout="wide")
st.title("Rooster turn-direction debug")

col_restart, col_stop_twist, col_refresh = st.columns([1, 1, 2])
with col_restart:
    if st.button("🔄 Restart Falcon (full rebuild)", type="primary"):
        with st.status("Restarting Falcon...", expanded=True) as status:
            restart_falcon(status)
with col_stop_twist:
    # See LESSONS.md "twist-control adapter's stop-watchdog..." (2026-07-27,
    # recurred 07-30 and 08-02): this process fights ui.py's manual control
    # whenever it's left running, because FALCON's waypoint_follower always
    # has a non-empty goal from launch (goal_x/goal_y), even with no click.
    if st.button("🛑 Stop twist adapter (before manual flying)"):
        run("docker exec robotican_dev pkill -f rooster_twist_control_adapter")
        run("pkill -f rooster_twist_control_adapter")
        st.success("Stopped (or was already down).")
with col_refresh:
    st.caption(f"Auto-refreshes every {REFRESH_SEC}s. Reads manual_flight_logger.py "
               f"(`it:{POSE_DIR}/`) and path_logger.py (`falcon:{PATH_DIR}/`) -- "
               "started automatically if not already running. This page never "
               "publishes to ROS itself.")

ensure_loggers_running()

st.subheader("Services")
up_count = 0
for name, kind, chk_loc, chk_pat, log_loc, log_path in SERVICES:
    is_up = container_up(chk_pat) if kind == "container" else proc_running(chk_loc, chk_pat)
    up_count += int(is_up)
    icon = "🟢" if is_up else "🔴"
    with st.expander(f"{icon} {name}"):
        log = tail_log(log_loc, log_path)
        if log is not None:
            st.code(log or "(empty)", language=None)
        else:
            st.caption("No log file tracked for this one.")
st.caption(f"{up_count} / {len(SERVICES)} up")

st.divider()

poses = tail_jsonl_from_container("it", f"{POSE_DIR}/pose.jsonl", n=300)
raw_poses = tail_jsonl_from_container("it", f"{POSE_DIR}/raw_pose.jsonl", n=300)
cmds = tail_jsonl_from_container("it", f"{POSE_DIR}/commands.jsonl", n=50)
paths = tail_jsonl_from_container("falcon", f"{PATH_DIR}/paths.jsonl", n=20)
goals = tail_jsonl_from_container("falcon", f"{PATH_DIR}/goals.jsonl", n=20)

c1, c1b, c2, c3 = st.columns(4)

with c1:
    st.subheader("Pose (transformed)")
    if poses:
        p = poses[-1]
        st.metric("x", f"{p['x']:.2f} m")
        st.metric("y", f"{p['y']:.2f} m")
        st.metric("yaw", f"{math.degrees(p['yaw']):.1f}°")
        st.line_chart({"yaw_deg": [math.degrees(pp["yaw"]) for pp in poses[-100:]]})
    else:
        st.warning("No pose data yet")

with c1b:
    st.subheader("Ground truth (Sphera raw)")
    if raw_poses:
        rp = raw_poses[-1]
        speed = math.hypot(rp["vx"], rp["vy"])
        st.metric("speed (xy)", f"{speed:.2f} m/s")
        st.metric("vx / vy", f"{rp['vx']:+.2f} / {rp['vy']:+.2f} m/s")
        st.metric("raw yaw", f"{math.degrees(rp['yaw']):.1f}°")
        st.line_chart({"speed_mps": [math.hypot(pp["vx"], pp["vy"]) for pp in raw_poses[-200:]]})
    else:
        st.warning("No raw ground-truth data yet")

with c2:
    st.subheader("Last commands (cmd_nav)")
    if cmds:
        for c in cmds[-8:]:
            axes = c.get("axes")
            if axes:
                st.text(f"move x={axes.get('x',0):.0f} y={axes.get('y',0):.0f} r={axes.get('r',0):.0f}")
            else:
                st.text(f"{c.get('action','?')} value={c.get('value','')}")
    else:
        st.warning("No commands yet")

with c3:
    st.subheader("Latest goal / path")
    if goals:
        g = goals[-1]
        st.text(f"goal: ({g['x']:.2f}, {g['y']:.2f})")
    if paths:
        pa = paths[-1]
        wps = pa["waypoints"]
        st.text(f"{len(wps)} waypoints, first={tuple(wps[0]) if wps else '?'}")
    else:
        st.warning("No path data yet")

st.divider()
st.subheader("Turn-direction check (most recent path)")

if paths and poses:
    pa = paths[-1]
    path_t = pa["t"]
    wps = pa["waypoints"]
    if len(wps) >= 1:
        pose_before = max((pp for pp in poses if pp["t"] <= path_t), key=lambda pp: pp["t"], default=None)
        poses_after = [pp for pp in poses if path_t < pp["t"] <= path_t + 2.5]
        if pose_before and poses_after:
            target = wps[1] if len(wps) > 1 else wps[0]
            bearing = bearing_deg((pose_before["x"], pose_before["y"]), target)
            yaw0 = math.degrees(pose_before["yaw"])
            diff = wrap180(bearing - yaw0)
            needed = "RIGHT (yaw should increase)" if diff > 0 else "LEFT (yaw should decrease)"
            yaw_end = math.degrees(poses_after[-1]["yaw"])
            actual_delta = wrap180(yaw_end - yaw0)
            actual = "RIGHT (increased)" if actual_delta > 0.5 else ("LEFT (decreased)" if actual_delta < -0.5 else "~flat")
            match = (diff > 0 and actual_delta > 0.5) or (diff < 0 and actual_delta < -0.5)

            m1, m2, m3 = st.columns(3)
            m1.metric("Needed turn", needed, f"{diff:+.1f}° to bearing")
            m2.metric("Actual turn", actual, f"{actual_delta:+.1f}° over {poses_after[-1]['t']-path_t:.1f}s")
            m3.metric("Consistent?", "✅ YES" if match else "❌ MISMATCH")
        else:
            st.info("Not enough pose samples around this path yet -- wait for next refresh")
    else:
        st.info("Latest path has no waypoints")
else:
    st.info("Waiting for both path and pose data...")

time.sleep(REFRESH_SEC)
st.rerun()
