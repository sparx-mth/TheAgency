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
#: The measured prison extents are x [-13.5, 89.9], y [-41.4, 19.3]; the box
#: is that envelope +1 m, snapped to 0.2 m. A 2026-08-31 enlargement to a
#: padded x [-75, 116] was reverted the same day when the coverage tour went
#: from ~0.2 ms to ~10.5 s per solve; this one holds because hgrid/
#: cell_size_max is pinned to 8.0 m in nav_stack.launch (112 tour cells, not
#: 624), measured back at 0.55 ms.
#: Must match maps/sphera_jail.yaml `building` and the hand-synced copies in
#: mission_control.py / rooster_turn_debug.py / mission_sphera.yaml.
BEV_XMIN, BEV_YMIN, BEV_XMAX, BEV_YMAX = -14.6, -42.4, 91.0, 20.4

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

#: Vertical speed a ranger STEP may imply before the altitude hold rejects the
#: sample as terrain rather than aircraft motion, m/s. The hold is
#: terrain-relative and accepted any finite reading, so a step in the floor
#: produced a huge apparent error and a huge correction: measured 2026-08-31,
#: two flights ran away to ~3.6 m (against a 3.8 m flight_band beyond which
#: exploration_node segfaults) with the rangefinder reading 11.6 m -- ~83 m/s
#: of implied motion. Replayed over recorded flights, 3.0 rejects 1.7% of
#: samples on the runaway flight and 0.0-0.3% on normal ones, so it filters
#: the impossible without binding in ordinary flight.
ALTITUDE_MAX_RANGER_RATE = 3.0

#: The z axis response is a ~10-count step gate near 700, not a thrust curve.
CLIMB_Z = 700.0
HOVER_Z = 700.0

#: Which controller generation flies, end to end. One switch so an A/B arm
#: cannot be half-assembled:
#:
#:   "powerlaw_lateral" (default) -- the 2026-08-31 rebuild: both horizontal
#:       axes fly the measured expo curve (robots/ROBOTICAN/rooster_axis_curve)
#:       through per-axis velocity servos, lateral is live (ceiling 900), and
#:       the follower commands the tracker's full velocity vector.
#:   "legacy" -- the pre-2026-08-31 baseline, verbatim: dead-band/two-regime
#:       feedforward on x, lateral hard-zeroed, course-projected forward only.
#:
#: Overridable per run via the environment so the A/B operator never edits
#: code: SPARX_CONTROLLER_VARIANT=legacy python3 -m ...campaign
CONTROLLER_VARIANT = os.environ.get(
    "SPARX_CONTROLLER_VARIANT", "powerlaw_lateral").strip().lower()
if CONTROLLER_VARIANT not in ("powerlaw_lateral", "legacy"):
    raise ValueError("SPARX_CONTROLLER_VARIANT must be powerlaw_lateral or "
                     "legacy, got %r" % CONTROLLER_VARIANT)

#: The follower half of the variant (twist-adapter half: TWIST_ADAPTER_CMD).
USE_LATERAL = CONTROLLER_VARIANT == "powerlaw_lateral"

#: Which follower consumes FALCON's plan.
#:
#: "reference" (traj_server -> /planning/pos_cmd -> ReferenceTracker3D) is the
#: campaign's baseline: it is the proven one, it carries the measured-speed
#: taper and the pinned-escape reflex, and every smoothness fix so far (turn
#: creep, the dropped duplicate dead-band quantiser, the yaw-rate cap) lives in
#: it. Note (verified in traj_server.cpp 2026-08-31): since the finish_reopen
#: patch, traj_server does NOT exit at FINISH by default (exit_on_finish=false)
#: -- past each trajectory's end it publishes the frozen endpoint with ZERO
#: velocity, fresh stamps, at 100 Hz. So the follower's staleness timeout never
#: fires between trajectories or at FINISH; "parked on a fresh zero-velocity
#: reference" is what planner starvation looks like from the follower's side.
#:
#: "bspline" reads /planning/bspline directly, so it needs no traj_server at
#: all, and its control law measured 2.8-3.8x tighter on SJTU's airframe -- but
#: its plant constants are UNMEASURED for Rooster. Switch to it only as a
#: deliberate A/B against a baseline run, after the plant is measured.
EXPLORATION_FOLLOWER = "reference"

#: Clearance FALCON's planner and optimiser keep from occupied voxels, metres.
#: See adapter_launch_cmd for why this is 0.40 rather than the inherited 0.85.
# 0.30 was TRIED AND REVERTED on 2026-08-31 (v3.0, three flights). It cut A*
# "no path to viewpoint" failures ~10x but bought NOTHING in coverage (0.98x
# against a pre-registered 1.15x bar) and halved the clearance margin: the
# aircraft sat within 0.3 m of geometry 40% of the flight (vs 27%), and the
# PLAN itself rode 0.33 m from mapped walls (vs 0.83 m) -- inside the
# aircraft's own p90 tracking error, on a platform that already makes contact.
# Why: inflation gates what A* considers reachable, and the A* route seeds the
# B-spline, so a lower value lets the seed hug walls while the optimiser's
# SOFT clearance only partly pulls it back. Deeper reason not to retry it:
# reachability was never the binding constraint on coverage (see
# runs/AUTOLOOP_JOURNAL.md Finding C -- coverage tracks distance r=0.95 and
# the aircraft stops because nothing commands it, not because it is stuck).
OBSTACLES_INFLATION = 0.40
#: The B-spline optimiser's own clearance, which is a SOFT cost (weight 50
#: against smoothness 20), not a constraint — so it gets traded away. Measured
#: 2026-08-20 with it at 0.40: the published reference passed within 0.40 m of
#: mapped geometry 73 % of the time, median 0.36, and the aircraft (0.52 m of
#: tracking error on top) sat at median 0.23 m and got PINNED a dozen times a
#: flight. The map is not to blame — it has zero isolated voxels and a median of
#: 13 occupied neighbours out of 26, so those surfaces are real.
#:
#: Deliberately raised ABOVE ``OBSTACLES_INFLATION``: A*'s inflation decides what
#: is reachable at all, and 0.85 there once made exploration fail outright, so it
#: stays at 0.40 and narrow doorways stay plannable. This only asks the curve
#: fitted through them to ride nearer the middle.
SAFE_DISTANCE = 0.55
#: Weight on the optimiser's clearance term, against smoothness at 20. The
#: median clearance was already fine at 0.50 m; it is the TAIL that causes
#: contacts, and the weight is what decides how much clearance gets traded away
#: in the tight spots where it matters.
BSPLINE_DISTANCE_WEIGHT = 150.0

#: Speed FALCON PLANS at, m/s, and the follower's ceiling above it.
#:
#: These are passed on the roslaunch command line, not left to a launch-file
#: default, because ``sphera_drone.launch`` re-declares many of ``nav_stack``'s
#: args and passes its own values down -- editing the nav_stack default is a
#: silent no-op for any of them (LESSONS.md). ``assert_launch_params`` reads
#: every one of these back from the live parameter server.
#:
#: Raised from the inherited 0.4: coverage is speed x sensor swath, the swath
#: cannot be raised (DA3 returns nothing beyond 3.5 m in this map), and the
#: plan -- not the follower -- was the binding constraint. On the measured
#: curve (rooster_axis_curve), 0.8 m/s asks ~702 counts and even the
#: follower's 1.0 m/s ceiling asks ~745 -- both comfortably under the
#: 900-count / 1.566 m/s platform ceiling.
PLAN_MAX_VEL = 0.8
#: Must exceed PLAN_MAX_VEL with headroom; full stick measures ~1.25 m/s.
EXPLORE_MAX_SPEED_XY = 1.0
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
#: Tilt at which the follower cuts drive, and the tilt it must fall back below
#: before drive resumes. Both live here so the readback guard covers them: the
#: limit is declared in BOTH launch files and the entry one wins.
#: Gain on horizontal position error in the follower's tracker. Cross-track
#: error is the only half of the tracking error that can cause a collision, and
#: it measured p50 0.20 m against a reference clearance of ~0.5 m.
TRACKER_POS_KP = 1.0

#: Seconds of reference-acceleration lead in the follower's tracker.
#: Measured over five v2.1 flights: 74% of the remaining tracking error is
#: ALONG-TRACK (timing), p90 0.45 m ~ 0.9 s at cruise -- which is the plant's
#: own lag (tau ~1.15 s + 0.14 s dead time) going uncompensated. 0.25 is the
#: historical default and keeps present behaviour; raising it is a
#: pre-registered A/B, not a free tweak.
# 0.60 was TRIED AND REVERTED 2026-08-31 (v5.0, two flights): along-track p90
# 0.565 / 0.525 against a 0.463 baseline median -- worse, not better. An
# offline sweep of the real tracker+servo+measured plant says more lead SHOULD
# help monotonically, so the dominant real effect is absent from a
# smooth-reference model. Finding F says why: the along-track error is made
# almost entirely while the aircraft is STATIONARY (when moving it is on
# schedule), and no amount of transient anticipation fixes an error
# accumulated while the loop commands zero. Do not retune this knob; reduce
# stationary time instead. See runs/AUTOLOOP_JOURNAL.md.
ACCEL_LEAD_S = 0.25

#: How the nose is aimed. "course" points it along travel -- a workaround from
#: the era when lateral was disabled, so sideways demand had to become forward
#: demand. "reference" follows FALCON's own yaw plan, which exists to point the
#: depth camera at the frontiers it wants to map; with lateral working the
#: workaround is no longer needed. Measured in course mode: heading error p50
#: 18-28 deg, p90 46-50, i.e. the camera is aimed well off the travel
#: direction much of the time, by a heuristic rather than by the planner.
YAW_MODE = "course"

#: Weight on traj_server's own yaw_dot (the B-spline's analytic yaw rate),
#: which the follower discarded -- driving yaw on proportional error alone
#: necessarily lags a moving yaw reference. Only acts in YAW_MODE="reference";
#: the two are one change and move together.
YAW_DOT_FF = 0.0

#: Slow yaw sweep while the plan is parked, rad/s (0 disables).
#:
#: Findings C/F: both the coverage shortfall and the tracking error are made
#: during the time the aircraft is stationary. While parked the camera never
#: sweeps -- course yaw only aims the nose when there is travel to aim it
#: along -- so nothing enters the map, no frontier resolves, and the planner
#: re-picks the same spot. v4.0 proved the converse by accident: removing the
#: sweep deadlocked exploration outright. Yaw only, so it cannot fly the
#: aircraft into anything, and it yields as soon as the plan asks for travel.
#: Overridable per run so an interleaved A/B never needs a file edit between
#: flights: SPARX_PARK_SCAN=0 python3 -m ...campaign
#:
#: 0.5 rad/s was flown (v7.0, 3 runs) and is NOT adopted: the mechanism worked
#: (moving-reference fraction 0.803 vs 0.594) but coverage did not follow
#: (1.02x, which at n=3 is indistinguishable from anything under ~40% -- see
#: Finding G), the aircraft turned 118 deg/m on one flight against 42-53 at
#: baseline, and altitude destabilised (ranger sd 0.345 vs 0.085: the spinning
#: rangefinder sweeps varied floor and the hold loop chases it). 0.25 rad/s
#: with a longer trigger is the gentler retry.
#: DEFAULT 0 -- the parked yaw scan was flown at 0.5 (v7.0, 3 runs) and 0.25
#: (v7.1, 5 interleaved runs) and is NOT adopted. At n=2-3 its mechanism metric
#: looked strong (moving-reference 0.84 vs 0.53); at n=5 interleaved against a
#: contemporaneous control it was 0.536 vs 0.527 -- i.e. nothing. That is
#: regression to the mean, exactly what Finding G predicts for this platform.
#: It also carries a TAIL HAZARD absent from the control arm: 2 of 8
#: scan-enabled flights ran away in altitude to ~3.6 m (17% of one flight above
#: 2 m, rangefinder reading up to 11.6 m) against 0 of 4 controls, which never
#: exceeded 1.93 m. Spinning sweeps the downward rangefinder across varied
#: floor, the altitude hold chases the jump, and the aircraft climbs toward the
#: 3.8 m flight_band ceiling where exploration_node segfaults.
PARK_SCAN_RATE = float(os.environ.get("SPARX_PARK_SCAN", "0"))

#: Seconds parked before the scan starts. 2.0 also fired on brief pauses; 4.0
#: restricts it to genuinely stalled stretches.
PARK_SCAN_AFTER_S = 4.0

#: Seconds the follower may hold translation at zero after giving up on escapes,
#: before re-arming and driving again. The hold was previously unbounded and its
#: release condition unreachable, which parked one flight for 250 s.
PINNED_HOLD_SEC = 4.0

#: Quiet time between escape attempts. Getting unstuck costs ~74 s of a 430 s
#: flight, most of it in escape-plus-cooldown cycles rather than the manoeuvre.
ESCAPE_COOLDOWN_SEC = 4.0

TILT_LIMIT_DEG = 35.0
TILT_RESUME_DEG = 27.0

#: Smallest frontier cluster FALCON will treat as a cluster at all. Set from
#: nav_stack.launch, which must win over the package's own frontier_finder.yaml
#: — hence the readback below.
FRONTIER_CLUSTER_MIN = 50.0

#: Base radius of a blacklist shadow around an unreachable viewpoint, metres.
#: Env-overridable so an interleaved A/B needs no file edit between flights:
#: SPARX_BLOCKED_RADIUS=1.5 python3 -m ...campaign
#:
#: 1.5 is the C++ default and was never set. Measured over 41 flights
#: (runs/AUTOLOOP_JOURNAL.md, Finding I): 91.9% of no-moving-reference time is
#: A*-plan-fail flooding; 86% of that is ONE terminal lock on a single
#: unreachable viewpoint -- worst case 254 s emitting the identical "Next pos"
#: 17,551 times with 17,925 plan fails and the reference never moving once --
#: and sweepBlockedFrontiers retired ZERO clusters in those runs while 10-11
#: shadows were re-struck. A 1.5 m first strike cannot retire a frontier whose
#: viewpoints are sampled to candidate_rmax 5.5 m. 2.75 makes strike 1 cover
#: 2.75 m and strike >=2 escalate to exactly 5.5 m.
# 2.75 was TRIED AND REVERTED 2026-09-01 (v8.0, 5 interleaved flights). It did
# what it was designed to do -- re-strikes fell (candidate 5,0,11,3,7 vs control
# 11,9,2,9) -- but it STERILISED THE MAP, which is the documented failure mode
# of a wider shadow: 4 of 5 candidate flights emptied their frontier set at
# least once and one emptied it twice (guard G3), against 1 of 4 controls.
# Coverage did not improve either (median 1340 vs 1255). The lock in Finding I
# is real, but blacklisting a bigger disc trades one starvation mode for
# another -- the aircraft runs out of places it is ALLOWED to go.
BLOCKED_REGION_RADIUS = float(os.environ.get("SPARX_BLOCKED_RADIUS", "1.5"))

#: Seconds the coverage tour may hold its chosen cell before re-picking. The
#: bound is the safety property, not the feature: a commitment to an unreachable
#: cell is the lock P16 fixed, and the timeout is what keeps one cheap.
TOUR_COMMIT_MAX_S = 0.0

#: Range at which the TSDF stops raycasting. 5.0 discarded 4.8 % of every depth
#: frame; sensing max_depth already allows 10 m.
VOXEL_RAYCAST_MAX = 8.0

EXPECTED_ROSPARAMS = {
    "/voxel_mapping/tsdf/raycast_max": VOXEL_RAYCAST_MAX,
    # A*'s hard reachability gate. Added to the readback 2026-08-31 because a
    # campaign now turns on its value: without this, "the change did nothing"
    # and "the change never reached the planner" look identical.
    "/voxel_mapping/obstacles_inflation": OBSTACLES_INFLATION,
    "/bspline_opt/safe_distance": SAFE_DISTANCE,
    "/bspline_opt/pos/distance": BSPLINE_DISTANCE_WEIGHT,
    "/frontier_finder/cluster_min": FRONTIER_CLUSTER_MIN,
    "/frontier_finder/blocked_region_radius": BLOCKED_REGION_RADIUS,
    "/exploration/tour_commit_max_s": TOUR_COMMIT_MAX_S,
    "/falcon_exploration_follower/tracker_pos_kp": TRACKER_POS_KP,
    "/falcon_exploration_follower/accel_lead_s": ACCEL_LEAD_S,
    "/falcon_exploration_follower/yaw_mode": YAW_MODE,
    "/falcon_exploration_follower/yaw_dot_ff_gain": YAW_DOT_FF,
    "/falcon_exploration_follower/park_scan_rate": PARK_SCAN_RATE,
    "/falcon_exploration_follower/park_scan_after_s": PARK_SCAN_AFTER_S,
    "/falcon_exploration_follower/pinned_hold_sec": PINNED_HOLD_SEC,
    "/falcon_exploration_follower/escape_cooldown_sec": ESCAPE_COOLDOWN_SEC,
    "/falcon_exploration_follower/tilt_limit_deg": TILT_LIMIT_DEG,
    "/falcon_exploration_follower/tilt_resume_deg": TILT_RESUME_DEG,
    "/uav_model/dynamics_parameters/max_linear_velocity": PLAN_MAX_VEL,
    "/fsm/slow_traj_target_vel": FSM_SLOW_TRAJ_TARGET_VEL,
    "/falcon_exploration_follower/max_speed_xy": EXPLORE_MAX_SPEED_XY,
    "/falcon_exploration_follower/use_lateral": USE_LATERAL,
    "/bev_publisher/z_ceil": BEV_Z_CEIL,
    # Map-epoch guard: the map yaml is mounted when the falcon CONTAINER is
    # created, so a stale container silently keeps flying an old map through
    # any number of roslaunch restarts (run 10 flew the reverted-away big map
    # this way). If these disagree, recreate the container: docker rm -f falcon.
    "/bev_publisher/bbox_xmin": BEV_XMIN,
    "/bev_publisher/bbox_xmax": BEV_XMAX,
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
        "explore_accel_lead_s:={lead} explore_yaw_mode:={yawmode} "
        "explore_yaw_dot_ff:={yawff} explore_park_scan:={parkscan} "
        "explore_park_scan_after:={parkafter} "
        "frontier_blocked_radius:={blockrad} "
        "explore_use_lateral:={usel} "
    ).format(map=MAP_NAME, follower=follower, drone=DRONE_ID,
             fx=CAM["fx"], fy=CAM["fy"], cx=CAM["cx"], cy=CAM["cy"],
             w=CAM["width"], h=CAM["height"], mind=CAM["min_depth"],
             gx=GOAL_X, gy=GOAL_Y,
             bxmin=BEV_XMIN, bymin=BEV_YMIN, bxmax=BEV_XMAX, bymax=BEV_YMAX,
             infl=OBSTACLES_INFLATION, safe=SAFE_DISTANCE,
             maxvel=PLAN_MAX_VEL, slowvel=FSM_SLOW_TRAJ_TARGET_VEL,
             expspeed=EXPLORE_MAX_SPEED_XY, zceil=BEV_Z_CEIL,
             usel=str(USE_LATERAL).lower(), lead=ACCEL_LEAD_S, yawmode=YAW_MODE, yawff=YAW_DOT_FF, parkscan=PARK_SCAN_RATE, parkafter=PARK_SCAN_AFTER_S, blockrad=BLOCKED_REGION_RADIUS)
    return ("docker exec {c} bash -lc '{env} roslaunch falcon_adapter "
            "sphera_drone.launch {args}{extra}'").format(
        c=FALCON_CONTAINER, env=FALCON_ENV, args=args, extra=extra)


COMMAND_UNIT_CMD = (
    "python3 /home/rooster/sparx_agency/robots/ROBOTICAN/adapters/"
    "rooster_command_unit.py --ros-args "
    "-p rooster_id:={drone} -p climb_z:={climb} -p hover_z:={hover} "
    "-p max_ranger_m:={maxr} -p target_ranger_m:={tgtr} "
    "-p altitude_hold_max_correction:=380.0 -p altitude_hold_interval_sec:=0.1 "
    "-p altitude_hold_max_ranger_rate:={rangerrate}"
).format(drone=DRONE_ID, climb=CLIMB_Z, hover=HOVER_Z,
         maxr=MAX_RANGER_M, tgtr=TARGET_RANGER_M,
         rangerrate=ALTITUDE_MAX_RANGER_RATE)

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

#: Ceiling on the lateral axis for the candidate arm, counts.
#:
#: 600 (~0.43 m/s), not the 900 curve ceiling, since 2026-08-31 round 2. The
#: round-1 A/B measured lateral swinging to p50 ~600 / p90 ~900 counts, 9-16
#: sign flips a minute, 9-18 % of flight above 600 -- not cross-track work
#: (cross-track sat at p50 0.15 m, needing under 0.3 m/s) but the turn-crab
#: feedforward banking the airframe hard enough that the operator flagged the
#: roll as too aggressive. 600 keeps every correction the tracker ever needs
#: and bounds the bank; the curve itself is untouched.
LATERAL_AXIS_CAP = 600.0

#: How far the twist adapter's altitude nudges may move the hold height from
#: where it was when tracking began, metres. 0.3 -> 0.60 for v6.0.
#:
#: Finding E: FALCON's reference sits ABOVE the aircraft in every flight
#: (mean dz +0.06..+0.38 m, p90 up to 1.13), because it plans inside a
#: flight_band the aircraft cannot use -- the aircraft is pinned near 1.2 m
#: and could previously bias that by only +/-0.3 m total. Viewpoints beyond
#: that are unreachable by construction and FALCON cannot know.
#:
#: Care: band 1.0 WITH a coarse 0.3 m nudge was flown before and railed the
#: live target, driving the hold loop hard (z sd 42 -> 114) and costing
#: horizontal speed. The nudge is now 0.15 m, so this doubles the range at
#: half the step size; 0.60 is deliberately short of the old 1.0.
# 0.60 was TRIED AND REVERTED 2026-08-31 (v6.0, two flights). Coverage 1538 /
# 1451 against a 1516 baseline (no gain), and stationary time got WORSE:
# 0.41 / 0.52 against 0.32. Chasing altitude appears to COST horizontal time —
# the aircraft spends the extra vertical authority climbing/descending instead
# of travelling — which is the same shape as the historical band=1.0
# regression, just milder. Finding E is real (the reference does sit above the
# aircraft) but widening the band is not the way to collect it.
ALTITUDE_BAND_M = 0.30



#: Bump this string on ANY controller-affecting change (adapter constants,
#: follower params, slew rates, caps) so every run's summary.json says exactly
#: which controller revision flew. Round 2's sample silently mixed two slew
#: configurations and two maps because nothing recorded the revision; this is
#: the guard. "v2.1" = cap 600 + gentle lateral slew (400/s attack, 600/s
#: release) + small map. "v7.0" = v2.1 + parked yaw scan 0.5 rad/s.
CONTROLLER_REV = "v8.0" if BLOCKED_REGION_RADIUS > 1.5 else "v2.1d"

#: Lateral is opt-in at the adapter (its default keeps the axis disabled for
#: every non-campaign /cmd_vel producer); the candidate arm enables it at
#: LATERAL_AXIS_CAP, the baseline arm flies the verbatim pre-2026-08-31 stack.
TWIST_ADAPTER_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/adapters/"
    "run_twist_control_adapter.sh --rooster-id {drone}{variant}"
).format(repo=REPO_ROOT, drone=DRONE_ID,
         variant=((" --max-lateral-axis %.0f --altitude-band-m %.2f"
                   % (LATERAL_AXIS_CAP, ALTITUDE_BAND_M)) if USE_LATERAL
                  else " --legacy-feedforward"))

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
