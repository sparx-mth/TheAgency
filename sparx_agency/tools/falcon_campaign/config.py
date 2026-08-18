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
#: Ranger altitude the hold loop chases, metres. Lowered from 1.6 so the
#: aircraft can pass through the map's lower doorways (MISSION.md P4).
TARGET_RANGER_M = 1.20
MAX_RANGER_M = 1.35

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

#: Volume of the exploration box, m^3 -- the denominator for coverage.
#: From `python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize` on
#: maps/sphera_jail.yaml: box 32.0 x 32.0 x 4.8 m. Re-derive it there if the
#: map's flight_band or bounds change; it is a reporting scale only and nothing
#: in the flight path reads it.
EXPLORABLE_VOLUME_M3 = 4915.0

FLIGHT_SECONDS = 600           # the operator's 10-minute window
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
    ).format(map=MAP_NAME, follower=follower, drone=DRONE_ID,
             fx=CAM["fx"], fy=CAM["fy"], cx=CAM["cx"], cy=CAM["cy"],
             w=CAM["width"], h=CAM["height"], mind=CAM["min_depth"],
             gx=GOAL_X, gy=GOAL_Y,
             bxmin=BEV_XMIN, bymin=BEV_YMIN, bxmax=BEV_XMAX, bymax=BEV_YMAX,
             infl=OBSTACLES_INFLATION, safe=SAFE_DISTANCE)
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
