"""Single source of operational truth for the autonomous FALCON campaign.

Every container name, topic, launch command and tuning knob the campaign needs
lives here and nowhere else. ``mission_control.py`` holds the same commands but
is a Streamlit app with module-level side effects, so it cannot be imported by a
headless supervisor -- these are kept deliberately in sync with it by hand, and
any divergence is a bug in whichever was edited last.

Python 3.8-compatible: the campaign modules are imported inside the Noetic
container as well as by the 3.12 host venv.
"""
from __future__ import annotations

import os
import pathlib

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(
    os.environ.get("SPARX_REPO", "/home/user1/GIT/TheAgency")).resolve()
RUNS_DIR = REPO_ROOT / "runs"
PAUSE_FILE = RUNS_DIR / "PAUSE"
STOP_FILE = RUNS_DIR / "STOP"

#: Host dir bind-mounted read-write into the falcon container at the same path.
FALCON_SHARED_DIR = pathlib.Path("/tmp/falcon")

#: Where the flight recorder writes INSIDE the vendor container.
#:
#: The recorder has to run in ``it``: it is the only container with the vendor
#: message definitions (``sphera_common_interfaces``, ``rooster_manager_
#: interfaces``) needed to read ground truth and the rangefinder at all --
#: ``robotican_dev`` has ROS 2 Humble but not those packages. Only
#: ``sparx_agency/`` is mounted into ``it``, not the repo root, so ``runs/`` is
#: not visible from there and the data is copied out after the flight instead.
RECORDER_DIR_IN_IT = "/tmp/campaign_run"

# ── Identity ─────────────────────────────────────────────────────────────
DRONE_ID = "R1"
MAP_NAME = "sphera_jail"

#: Sphera's ROS 2 domain, and the CycloneDDS profile that pins the NIC.
ROS_DOMAIN_ID = "9"
RMW = "rmw_cyclonedds_cpp"

# ── Containers ───────────────────────────────────────────────────────────
FALCON_CONTAINER = "falcon"          # ROS1 Noetic, FALCON planner + adapter
IT_CONTAINER = "it"                  # ROS2 Foxy, vendor Rooster backend
DEV_CONTAINER = "robotican_dev"      # ROS2 Humble, frame capture / depth / twist
BRIDGE_CONTAINER = "ros1_bridge"
DRONE_CONTAINER = DRONE_ID           # the Sphera-spawned drone backend

#: A drone container whose image does not start with this is NOT the simulator.
#: The campaign refuses to arm anything else -- see bringup.assert_simulator().
SIM_IMAGE_PREFIX = "sphera-backend"

# ── Shell environments ───────────────────────────────────────────────────
IT_ENV = (
    "source /opt/ros/foxy/setup.bash && "
    "source /home/rooster/workspace/install/setup.bash && "
    "export ROS_DOMAIN_ID={domain} && "
    "export RMW_IMPLEMENTATION={rmw} && "
    "export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml && "
    "export PYTHONPATH=/home/rooster:$PYTHONPATH && "
).format(domain=ROS_DOMAIN_ID, rmw=RMW)

DEV_ENV = (
    "source /opt/ros/humble/setup.bash && "
    "export ROS_DOMAIN_ID={domain} && "
    "export RMW_IMPLEMENTATION={rmw} && "
    "export CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml && "
    "export PYTHONPATH={repo}:$PYTHONPATH && "
).format(domain=ROS_DOMAIN_ID, rmw=RMW, repo=REPO_ROOT)

FALCON_ENV = (
    "source /opt/ros/noetic/setup.bash && "
    "source /catkin_ws/devel/setup.bash && "
)

# ── Geometry (Rooster spawns at ~(54.75, -14.66) after the X/Y sign fix) ──
GOAL_X, GOAL_Y = 57.25, -10.16
BEV_XMIN, BEV_YMIN, BEV_XMAX, BEV_YMAX = 38.75, -30.66, 70.75, 1.34

# ── Camera intrinsics (hfov 135deg, confirmed against Sphera's own config) ─
CAM = dict(fx=111.837662, fy=180.0, cx=269.5, cy=179.5,
           width=540, height=360, min_depth=0.45)

# ── Flight profile ───────────────────────────────────────────────────────
#: Ranger altitude the hold loop chases, metres, and the ceiling the twist
#: adapter's nudges may push that setpoint to.
#:
#: These are set from the loop's MEASURED behaviour rather than from the height
#: wanted. The loop parks a steady ~0.22 m above whatever target it is given
#: (1.35 -> a held 1.50-1.60 m across many runs), and the adapter's climb nudges
#: pin the live target at MAX_RANGER_M, so the height actually flown is
#: approximately MAX_RANGER_M + 0.22.
#:
#: Raising the descent gain to close that offset was tried and reverted: it
#: fixed altitude but horizontal speed fell with it (see MISSION.md P11), and on
#: this airframe z and translation appear to share thrust authority. Biasing the
#: setpoint down costs nothing, because the offset is steady and predictable.
#:
#: 1.00 + 0.22 lands near 1.22 m, which clears the ~1.0 m floor-clutter limit
#: (LESSONS.md) while flying low enough for the map's doorways -- the point of
#: P4. Re-derive both if the offset itself changes.
TARGET_RANGER_M = 0.90
MAX_RANGER_M = 1.00

#: The z axis response is a ~10-count step gate near 700, not a thrust curve.
CLIMB_Z = 700.0
HOVER_Z = 700.0

#: Which follower consumes FALCON's plan.
#:
#: "reference" (traj_server -> /planning/pos_cmd -> ReferenceTracker3D) is the
#: campaign's baseline: it is the proven one, it carries the measured-speed
#: taper and the pinned-escape reflex, and every smoothness fix so far (turn
#: creep, the dropped duplicate dead-band quantiser, the yaw-rate cap) lives in
#: it. Its weakness is that traj_server EXITS when FALCON reaches FINISH, which
#: leaves /planning/pos_cmd publisher-less and the follower holding forever.
#:
#: "bspline" reads /planning/bspline directly, so it needs no traj_server at
#: all, and its control law measured 2.8-3.8x tighter on SJTU's airframe -- but
#: its plant constants are UNMEASURED for Rooster. Switch to it only as a
#: deliberate A/B against a baseline run, after the plant is measured.
EXPLORATION_FOLLOWER = "reference"

#: Clearance FALCON's planner and optimiser keep from occupied voxels, metres.
#: See adapter_launch_cmd for why this is 0.40 rather than the inherited 0.85.
OBSTACLES_INFLATION = 0.40
SAFE_DISTANCE = 0.40

#: Speed FALCON PLANS at, m/s, and the follower's ceiling above it.
#:
#: These are passed on the roslaunch command line, not left to a launch-file
#: default, because ``sphera_drone.launch`` re-declares many of ``nav_stack``'s
#: args and passes its own values down -- editing the nav_stack default is a
#: silent no-op for any of them (LESSONS.md). ``assert_launch_params`` reads
#: every one of these back from the live parameter server.
#:
#: 0.6 rather than the inherited 0.4: coverage is speed x sensor swath, the
#: swath cannot be raised (DA3 returns nothing beyond 3.5 m in this map), and
#: the plan -- not the follower -- was the binding constraint. A 0.6 m/s demand
#: asks 802 axis counts standing and 603 moving, both under the 900 ceiling;
#: 1.0 m/s would ask 924 and clip.
PLAN_MAX_VEL = 0.6
#: Must exceed PLAN_MAX_VEL with headroom; full stick measures ~1.25 m/s.
EXPLORE_MAX_SPEED_XY = 0.8
#: The slow-trajectory rescale targets the planner's own max_vel; a lower value
#: would quietly undo the raise on exactly the segments it fires on.
FSM_SLOW_TRAJ_TARGET_VEL = PLAN_MAX_VEL

#: Top of the BEV column band, metres. 1.50 (the sphera_drone default) drops
#: everything above head height out of the 2D obstacle view.
BEV_Z_CEIL = 2.20

#: Volume of the exploration box, m^3 -- the denominator for coverage.
#: From `python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize` on
#: maps/sphera_jail.yaml: box 32.0 x 32.0 x 4.8 m. Re-derive it there if the
#: map's flight_band or bounds change; it is a reporting scale only and nothing
#: in the flight path reads it.
EXPLORABLE_VOLUME_M3 = 4915.0

#: Flight window, seconds.
#:
#: 430, not the operator's nominal 600. Measured across four runs, the battery
#: reaches 25% at ~430 s every time and hits zero by the end, and below 25% this
#: platform loses thrust authority (LESSONS.md). The last ~170 s contributed
#: 0.9-2.0 m of travel in recent runs at a mean speed of 0.003-0.009 m/s -- it is
#: not flight, it is a flat battery being recorded. Cutting it also stops ~30% of
#: every run being averaged into metrics from a regime the project's own notes
#: call corrupted.
#:
#: NOTE when comparing with history: older runs are 600 s. Compare against their
#: distance over the first 430 s (253 / 152 / 140 / 57 m for the runs on
#: 2026-08-19), not their totals.
FLIGHT_SECONDS = 430
HOVER_SETTLE_TIMEOUT_S = 60.0

# ── Commands ─────────────────────────────────────────────────────────────
FALCON_CONTAINER_CMD = (
    "cd {repo}/sparx_agency/tasks/planning/falcon && "
    "./run_falcon_sphera.sh {map}"
).format(repo=REPO_ROOT, map=MAP_NAME)

BRIDGE_CMD = (
    "cd {repo}/sparx_agency/tasks/planning/falcon/bridge && "
    "ROS_DOMAIN_ID={domain} RMW_IMPLEMENTATION={rmw} "
    "CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh"
).format(repo=REPO_ROOT, domain=ROS_DOMAIN_ID, rmw=RMW)


#: rosparam name -> value the campaign requires it to have after bring-up.
#:
#: Every entry here is a value that was ONCE set in the wrong file and silently
#: ignored. A launch arg is not a setting until the parameter server agrees.
EXPECTED_ROSPARAMS = {
    "/uav_model/dynamics_parameters/max_linear_velocity": PLAN_MAX_VEL,
    "/fsm/slow_traj_target_vel": FSM_SLOW_TRAJ_TARGET_VEL,
    "/falcon_exploration_follower/max_speed_xy": EXPLORE_MAX_SPEED_XY,
    "/bev_publisher/z_ceil": BEV_Z_CEIL,
}


def adapter_launch_cmd(follower=None, extra=""):
    # type: (str, str) -> str
    """The roslaunch line that starts FALCON's adapter in exploration mode.

    Args:
        follower: ``reference`` or ``bspline``; defaults to
            :data:`EXPLORATION_FOLLOWER`.
        extra: Additional ``key:=value`` args appended verbatim.

    Returns:
        A shell command suitable for ``docker exec``.
    """
    follower = follower or EXPLORATION_FOLLOWER
    args = (
        "map_name:={map} nav_mode:=exploration exploration_follower:={follower} "
        "real_pose_topic:=/{drone}/localization "
        "real_depth_path_topic:=/{drone}/depth_frame_path "
        "real_rgb_path_topic:=/{drone}/rgb_frame_path "
        "cam_fx:={fx} cam_fy:={fy} cam_cx:={cx} cam_cy:={cy} "
        "cam_width:={w} cam_height:={h} cam_min_depth:={mind} "
        "sync_tolerance:=0.05 max_interp_gap:=0.12 "
        "goal_x:={gx} goal_y:={gy} "
        "bev_xmin:={bxmin} bev_ymin:={bymin} bev_xmax:={bxmax} bev_ymax:={bymax} "
        "apf_max_total_shift_m:=0.3 "
        "bev_t_on:=3.0 bev_occ_conf_full:=4.0 bev_min_wall_run:=4 "
        # 0.85 (the inherited default) against a 0.20 m voxel grid demands a
        # 1.7 m-wide free corridor, and measured live 2026-08-18 that is what
        # made exploration fail: FALCON kept picking reachable-looking
        # viewpoints and A* could not route to any of them
        # ("No path to next viewpoint using default A*" then "coarse A*",
        # 1156 consecutive [FSM] Plan fail). 0.40 matches this stack's own 2D
        # planner (inflate_radius_m) and is still two full voxels of margin.
        "obstacles_inflation:={infl} safe_distance:={safe} "
        # Shadowed by sphera_drone.launch if left to nav_stack's defaults.
        "max_vel:={maxvel} fsm_slow_traj_target_vel:={slowvel} "
        "explore_max_speed_xy:={expspeed} bev_z_ceil:={zceil} "
    ).format(map=MAP_NAME, follower=follower, drone=DRONE_ID,
             fx=CAM["fx"], fy=CAM["fy"], cx=CAM["cx"], cy=CAM["cy"],
             w=CAM["width"], h=CAM["height"], mind=CAM["min_depth"],
             gx=GOAL_X, gy=GOAL_Y,
             bxmin=BEV_XMIN, bymin=BEV_YMIN, bxmax=BEV_XMAX, bymax=BEV_YMAX,
             infl=OBSTACLES_INFLATION, safe=SAFE_DISTANCE,
             maxvel=PLAN_MAX_VEL, slowvel=FSM_SLOW_TRAJ_TARGET_VEL,
             expspeed=EXPLORE_MAX_SPEED_XY, zceil=BEV_Z_CEIL)
    return ("docker exec {c} bash -lc '{env} roslaunch falcon_adapter "
            "sphera_drone.launch {args}{extra}'").format(
        c=FALCON_CONTAINER, env=FALCON_ENV, args=args, extra=extra)


COMMAND_UNIT_CMD = (
    "python3 /home/rooster/sparx_agency/robots/ROBOTICAN/adapters/"
    "rooster_command_unit.py --ros-args "
    "-p rooster_id:={drone} -p climb_z:={climb} -p hover_z:={hover} "
    "-p max_ranger_m:={maxr} -p target_ranger_m:={tgtr} "
    "-p altitude_hold_max_correction:=380.0 -p altitude_hold_interval_sec:=0.1"
).format(drone=DRONE_ID, climb=CLIMB_Z, hover=HOVER_Z,
         maxr=MAX_RANGER_M, tgtr=TARGET_RANGER_M)

GTL_CMD = (
    "python3 -m sparx_agency.robots.ROBOTICAN."
    "rooster_ground_truth_localization --ros-args -p rooster_id:={drone}"
).format(drone=DRONE_ID)

VIDEO_TRIGGER_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/run_video_trigger.sh "
    "--drone-id {drone} --host-ip 127.0.0.1 --port 5001 --width {w} --height {h}"
).format(repo=REPO_ROOT, drone=DRONE_ID, w=CAM["width"], h=CAM["height"])

FRAME_CAPTURE_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/run_rooster_frame_dir_publisher.sh"
).format(repo=REPO_ROOT)

DEPTH_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/run_depth_processor.sh"
).format(repo=REPO_ROOT)

TWIST_ADAPTER_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/adapters/"
    "run_twist_control_adapter.sh --rooster-id {drone}"
).format(repo=REPO_ROOT, drone=DRONE_ID)

SPHERA_RESTART_CMD = (
    "cd {repo} && python3 -m sparx_agency.tools.sphera_battery_watchdog --once"
).format(repo=REPO_ROOT)

# ── Topics the recorders subscribe to ────────────────────────────────────
ROS1_TOPICS = dict(
    bspline="/planning/bspline",
    pos_cmd="/planning/pos_cmd",
    replan="/planning/replan",
    odom="/odom_world",
    cmd_vel_raw="/cmd_vel_raw",
    cmd_vel="/cmd_vel",
    demo_mode="/{}/demo_mode".format(DRONE_ID),
    attitude="/{}/attitude_rpy".format(DRONE_ID),
    frontier="/planning_vis/frontier_pcl",
    viewpoints="/planning_vis/viewpoints",
    go_status="/mission/go_status",
    recovery="/recovery/status",
)

ROS2_TOPICS = dict(
    truth="/{}/sphera/state".format(DRONE_ID),
    localization="/{}/localization".format(DRONE_ID),
    velocity="/{}/velocity_truth".format(DRONE_ID),
    attitude="/{}/attitude_rpy".format(DRONE_ID),
    cmd_nav="/{}/cmd_nav".format(DRONE_ID),
    manual="/{}/manual_control".format(DRONE_ID),
    state="/{}/state".format(DRONE_ID),
    status="/{}/rooster_status".format(DRONE_ID),
)
