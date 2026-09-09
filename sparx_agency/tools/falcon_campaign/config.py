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

import yaml

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
#
# DEAD KNOB, confirmed 2026-09-03 (LOOP_BUGS.md B36). The string "inflation"
# appears nowhere in this FALCON -- not in the C++, headers, yaml, or the built
# binary -- and the pre-rebuild image has none either, so it has never been live
# in this deployment. FALCON enforces clearance through SAFE_DISTANCE against
# the voxel mapper's ESDF instead. Kept because it is harmless and a future port
# from falcon_sjtu may make it real; do NOT run an experiment on it, and do not
# read `clearance.aircraft_frac_inside_inflation` as "inside a margin the
# planner applied" -- it is just "within 0.40 m of a mapped obstacle".
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

#: Floor on the slow-trajectory rescale ratio. 1.0 makes the rescale a no-op.
#:
#: CONFIRMED on a real flight 2026-09-02: the ratio is computed against a
#: slow_traj_target_yaw that was never set (C++ default 1.57 rad/s), so it lands
#: at 0.196-0.299 and is clamped to this floor EVERY time -- 11 fires in one
#: 453 s flight, each compressing the middle of the curve by 1/0.6 = 1.667x. The
#: affected plans are the long ones (8.1 / 5.8 / 11.3 s), roughly 12 % of flight
#: time, and they are slow precisely because the yaw turn needs the time. See
#: LOOP_BUGS.md B12/B13. Env-overridable so the A/B needs no file edit:
#: SPARX_SLOW_RATIO_MIN=1.0 python3 -m ...campaign
FSM_SLOW_TRAJ_RATIO_MIN = float(os.environ.get("SPARX_SLOW_RATIO_MIN", "0.60"))

#: A* search-time budgets, seconds. Defaults reproduce astar.yaml exactly, so
#: nothing changes until an experiment sets them.
#:
#: Measured 2026-09-03 over two full flights with the sparx-astar diagnostics:
#: 400/184 and 436/163 TIMEOUT vs OPEN_SET_EMPTY -- about 70 % of every A*
#: failure is the clock, not the geometry -- and the timeouts report **iters
#: 13-15, nodes ~130-160**, i.e. the search is abandoned after expanding barely
#: a dozen cells. Plan-fail flooding is 21.7 % of all flight time (B6), and this
#: is the first evidence that says which of its two possible causes is real.
#: Env-overridable so an A/B needs no file edit:
#: SPARX_ASTAR_DEFAULT_T=0.005 SPARX_ASTAR_COARSE_T=0.001 python3 -m ...campaign
ASTAR_DEFAULT_MAX_SEARCH_TIME = float(
    os.environ.get("SPARX_ASTAR_DEFAULT_T", "0.001"))
ASTAR_COARSE_MAX_SEARCH_TIME = float(
    os.environ.get("SPARX_ASTAR_COARSE_T", "0.0001"))

#: Top of the BEV column band, metres. 1.50 (the sphera_drone default) drops
#: everything above head height out of the 2D obstacle view.
BEV_Z_CEIL = 2.20

def _explorable_box():
    """The exploration box from the live map yaml: (volume m^3, footprint m^2).

    Derived rather than written down. The constant here read 4915.0 -- a
    32 x 32 x 4.8 m box that has not been the map since the bounds were widened
    -- so every `frac_of_box` printed to 2026-09-02 was inflated 6.5x.

    Resolved relative to THIS FILE, not to ``REPO_ROOT``: ``recorder.py`` imports
    this module inside the ``it`` container, where the repo is mounted at
    ``/home/rooster/sparx_agency`` and the host path does not exist. Reading a
    host path at import time took the recorder down on every cycle
    (2026-09-02) -- an import must not depend on where it is imported from.
    Falls back rather than raising for the same reason.

    Returns:
        ``(volume_m3, footprint_m2)`` of ``area.building`` x ``area.flight_band``,
        or ``(None, None)`` when the map file cannot be read here.
    """
    path = (pathlib.Path(__file__).resolve().parents[2] / "tasks" / "planning" /
            "falcon" / "maps" / (MAP_NAME + ".yaml"))
    try:
        with open(str(path)) as handle:
            area = yaml.safe_load(handle)["map_config"]["area"]
        x0, y0, x1, y1 = [float(v) for v in area["building"]]
        z0, z1 = [float(v) for v in area["flight_band"]]
    except Exception:                    # noqa: BLE001 -- a report scale, not flight
        return None, None
    footprint = abs(x1 - x0) * abs(y1 - y0)
    return footprint * abs(z1 - z0), footprint


#: Volume and footprint of the exploration box -- the denominators for coverage.
#: Reporting scale only; nothing in the flight path reads either.
EXPLORABLE_VOLUME_M3, EXPLORABLE_AREA_M2 = _explorable_box()

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
#
# Env-driven 2026-09-03 so F23 can arm it without a file edit. "reference" hands
# the nose back to the planner; the rate is already capped at max_yaw_rate
# (45 deg/s), the same ceiling course mode slews at, so no new limiter is needed.
YAW_MODE = os.environ.get("SPARX_YAW_MODE", "course").strip().lower()

#: Weight on traj_server's own yaw_dot (the B-spline's analytic yaw rate),
#: which the follower discarded -- driving yaw on proportional error alone
#: necessarily lags a moving yaw reference. Only acts in YAW_MODE="reference";
#: the two are one change and move together.
YAW_DOT_FF = float(os.environ.get("SPARX_YAW_DOT_FF", "0.0"))

#: "blend" yaw mode thresholds, m/s: pure planner yaw at or below LO, pure
#: direction-of-travel at or above HI. Defaults chosen from the measured
#: commanded-speed distribution -- about half of all ticks sit below 0.25 m/s,
#: where aiming the camera is nearly free (F23/F24).
#: Seconds between the follower's re-reads of ~yaw_mode. 0 = read once at
#: start-up (shipped). >0 enables a WITHIN-FLIGHT PAIRED run (B39): the campaign
#: alternates the mode mid-flight so each flight is its own matched pair, which
#: removes the between-flight variance that makes every outcome metric here cost
#: 40-130 flights per arm to resolve.
YAW_MODE_POLL_S = float(os.environ.get("SPARX_YAW_MODE_POLL", "0.0"))
YAW_BLEND_LO = float(os.environ.get("SPARX_YAW_BLEND_LO", "0.15"))
YAW_BLEND_HI = float(os.environ.get("SPARX_YAW_BLEND_HI", "0.50"))

#: Seed FALCON's frontier blocked-regions from the recorded hazard cells, and
#: how long a blocked region lives (seconds; 0 = permanent).
#:
#: B38: parked plan, 41% trajectory rejection and physical wedging are one chain
#: rooted in ~12 map cells. frontier_finder.cpp reads
#: /frontier_finder/blocked_regions_runtime at construction, so seeding it makes
#: FALCON start the flight already avoiding them. The default TTL of 90 s
#: expires a third of the way into a flight, which is why F21 also sets it.
BLOCKED_SEED = os.environ.get("SPARX_BLOCKED_SEED", "false").strip().lower()
BLOCKED_TTL_S = float(os.environ.get("SPARX_BLOCKED_TTL", "90.0"))
#: Strike count a SEEDED blocked region enters at (B45). The restore path uses
#: 1, and strike 1 is deliberately too weak to retire a frontier -- so F21's
#: twelve seeded cells were loaded and then ignored. 2 is the strength the code
#: reserves for a region that has already failed twice, which they have.
BLOCKED_SEED_STRIKES = int(os.environ.get("SPARX_BLOCKED_SEED_STRIKES", "1"))

#: NLopt total budget for the B-spline solve, seconds (FALCON ships 0.01).
#: See LOOP_FIXES.md F25: clearance is a soft penalty in the optimiser and a
#: hard test at the gate, and 41% of curves are rejected.
BSPLINE_OPT_MAX_TIME = float(os.environ.get("SPARX_BSPLINE_OPT_TIME", "0.01"))

#: Lift control points out of obstacles before the B-spline solve (F28/B44).
#: The ESDF is unsigned, so such a point has zero gradient and the optimiser
#: cannot move it out however heavily the term is weighted.
BSPLINE_LIFT = os.environ.get("SPARX_BSPLINE_LIFT", "false").strip().lower()
BSPLINE_LIFT_RADIUS = float(os.environ.get("SPARX_BSPLINE_LIFT_RADIUS", "1.0"))

#: Give the clearance cost a usable gradient inside obstacles (F29/B44). The
#: unsigned ESDF makes dist_grad exactly zero there, so the penalty is flat and
#: the solver cannot descend it; this substitutes an occupancy-derived escape
#: direction. Lifting points out beforehand (F28) does not hold -- the solve
#: puts them back.
BSPLINE_ESCAPE = os.environ.get("SPARX_BSPLINE_ESCAPE", "false").strip().lower()
#: Strength of the escape push. F29 flew it at 1.0: rejections fell to 0.67x
#: but plan_fail rose 10.7x and coverage to 0.56x -- the push was blunt enough
#: to turn rejected trajectories into failed plans. <1 nudges over several
#: iterations instead of displacing in one.
BSPLINE_ESCAPE_GAIN = float(os.environ.get("SPARX_BSPLINE_ESCAPE_GAIN", "1.0"))

#: Weight on the COURSE's own turn rate in course mode -- the exact analogue of
#: YAW_DOT_FF for the heading the follower derives itself.
#:
#: Yaw is driven by a pure P-loop on heading error (`yaw_kp` 1.0), so holding a
#: course that slews at R rad/s costs a STANDING error of R/yaw_kp. At the
#: 45 deg/s course slew that is a standing 45 deg, and the measured heading error
#: sits exactly there: p50 24 deg, p90 42-44 deg over 477 heartbeats. The nose is
#: therefore chronically behind the direction of travel, which aims the depth
#: camera off-path and pushes demand onto the weak lateral axis.
#:
#: 1.0 cancels it in the ideal case. DEFAULT 0.0 -- present behaviour until its
#: own pre-registered A/B (LOOP_FIXES.md). Env-overridable so the A/B needs no
#: file edit between flights: SPARX_COURSE_RATE_FF=1.0 python3 -m ...campaign
COURSE_RATE_FF = float(os.environ.get("SPARX_COURSE_RATE_FF", "0.0"))

#: Speed gate for course STEERING only, m/s. Negative -> the node's
#: course_min_speed (0.05), i.e. unchanged behaviour.
#:
#: The desired course is atan2 of the commanded velocity, so at low commanded
#: speed it carries no information. Measured 2026-09-03 (B31): below 0.25 m/s
#: half of all direction changes exceed the 45 deg/s course slew ceiling, and
#: 48% of ticks sit there -- the limiter saturates 71% of the time absorbing it.
COURSE_STEER_MIN_SPEED = float(os.environ.get("SPARX_COURSE_STEER_MIN_SPEED", "-1.0"))

#: Hold the course demand across steering gaps shorter than this, seconds.
#: 0 = drop it on every gap (pre-2026-09-03 behaviour).
#:
#: Measured (F19 interim): every resume costs a run of ceiling-rate catch-up --
#: 98% of the first tick after a resume is saturated, decaying to 47% only after
#: ~50 ticks -- and a gated flight had 99 gaps, so the transients ate the entire
#: steady-state win. 2.0 s spans 81% of observed gaps (median 0.35 s, p90 2.6 s).
COURSE_HOLD_GAP_S = float(os.environ.get("SPARX_COURSE_HOLD_GAP", "0.0"))

#: FALCON's periodic replan interval, seconds (stock yaml value 3.0).
#:
#: B34: each published trajectory carries about three seconds of motion and then
#: goes dead, while plans arrive every 3.28 s (p50) / 4.88 s (p90) -- so the
#: aircraft races a plan that expires, and the reference is parked on 50% of
#: ticks. Lowering this makes the next plan land while the aircraft is still
#: moving. B35: this is a STOCK FALCON param under /exploration_manager/fsm/;
#: the launch wrote it to /fsm/ and it was ignored for the whole campaign.
FSM_REPLAN_THRESH3 = float(os.environ.get("SPARX_REPLAN_THRESH3", "3.0"))

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
#: Base blocked-region radius, m. Env-driven 2026-09-04 for F33.
#:
#: B46: the launch comment derives 2.75 as the geometric requirement -- strike 1
#: covers 2.75 m and strike >=2 escalates to min(5.5, radius_max) = 5.5 m, which
#: is candidate_rmax exactly -- and the value below it was never changed from
#: 1.5, so strike >=2 reaches only 3.0 m and frontiers 3.0-5.5 m from a shadow
#: are never retired. That is why F21's seeding loaded correctly and leaked.
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
    "/falcon_exploration_follower/course_rate_ff_gain": COURSE_RATE_FF,
    "/falcon_exploration_follower/course_steer_min_speed": COURSE_STEER_MIN_SPEED,
    "/falcon_exploration_follower/course_hold_gap_s": COURSE_HOLD_GAP_S,
    # Read back under the namespace FALCON actually reads (B35): the launch
    # spent the whole campaign writing these to /fsm/, where nothing read them.
    "/exploration_manager/fsm/replan_thresh3": FSM_REPLAN_THRESH3,
    "/fsm/slow_traj_ratio_min": FSM_SLOW_TRAJ_RATIO_MIN,
    "/astar/profile/default/max_search_time": ASTAR_DEFAULT_MAX_SEARCH_TIME,
    "/astar/profile/coarse/max_search_time": ASTAR_COARSE_MAX_SEARCH_TIME,
    "/falcon_exploration_follower/yaw_dot_ff_gain": YAW_DOT_FF,
    "/falcon_exploration_follower/yaw_mode_poll_s": YAW_MODE_POLL_S,
    "/falcon_exploration_follower/yaw_blend_lo": YAW_BLEND_LO,
    "/falcon_exploration_follower/yaw_blend_hi": YAW_BLEND_HI,
    "/frontier_finder/blocked_region_ttl_s": BLOCKED_TTL_S,
    "/frontier_finder/blocked_seed_strikes": BLOCKED_SEED_STRIKES,
    "/bspline_opt/max_iteration_time": BSPLINE_OPT_MAX_TIME,
    "/bspline_opt/lift_radius_m": BSPLINE_LIFT_RADIUS,
    "/bspline_opt/escape_gain": BSPLINE_ESCAPE_GAIN,
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
        "fsm_slow_traj_ratio_min:={slowratio} "
        "astar_default_max_search_time:={astardef} "
        "astar_coarse_max_search_time:={astarcoarse} "
        "explore_max_speed_xy:={expspeed} bev_z_ceil:={zceil} "
        "explore_accel_lead_s:={lead} explore_yaw_mode:={yawmode} "
        "explore_yaw_dot_ff:={yawff} explore_course_rate_ff:={courseff} "
        "explore_yaw_mode_poll:={yawpoll} "
        "explore_yaw_blend_lo:={blendlo} explore_yaw_blend_hi:={blendhi} "
        "frontier_blocked_seed:={bseed} frontier_blocked_ttl:={bttl} "
        "frontier_seed_strikes:={bstrk} "
        "bspline_opt_max_time:={bopt} "
        "bspline_lift_ctrlpts:={blift} bspline_lift_radius:={bliftr} "
        "bspline_escape_gradient:={besc} "
        "bspline_escape_gain:={bescg} "
        "explore_course_steer_min_speed:={coursegate} "
        "explore_course_hold_gap:={coursehold} "
        "fsm_replan_thresh3:={replan3} "
        "explore_park_scan:={parkscan} "
        "explore_park_scan_after:={parkafter} "
        "frontier_blocked_radius:={blockrad} "
        "explore_use_lateral:={usel} "
    ).format(map=MAP_NAME, follower=follower, drone=DRONE_ID,
             fx=CAM["fx"], fy=CAM["fy"], cx=CAM["cx"], cy=CAM["cy"],
             w=CAM["width"], h=CAM["height"], mind=CAM["min_depth"],
             gx=GOAL_X, gy=GOAL_Y,
             bxmin=BEV_XMIN, bymin=BEV_YMIN, bxmax=BEV_XMAX, bymax=BEV_YMAX,
             infl=OBSTACLES_INFLATION, safe=SAFE_DISTANCE,
             maxvel=PLAN_MAX_VEL, slowvel=FSM_SLOW_TRAJ_TARGET_VEL, slowratio=FSM_SLOW_TRAJ_RATIO_MIN, astardef=ASTAR_DEFAULT_MAX_SEARCH_TIME, astarcoarse=ASTAR_COARSE_MAX_SEARCH_TIME,
             expspeed=EXPLORE_MAX_SPEED_XY, zceil=BEV_Z_CEIL,
             usel=str(USE_LATERAL).lower(), lead=ACCEL_LEAD_S, yawmode=YAW_MODE, yawff=YAW_DOT_FF, yawpoll=YAW_MODE_POLL_S, blendlo=YAW_BLEND_LO, blendhi=YAW_BLEND_HI, bseed=BLOCKED_SEED, bttl=BLOCKED_TTL_S, bstrk=BLOCKED_SEED_STRIKES, bopt=BSPLINE_OPT_MAX_TIME, blift=BSPLINE_LIFT, bliftr=BSPLINE_LIFT_RADIUS, besc=BSPLINE_ESCAPE, bescg=BSPLINE_ESCAPE_GAIN, courseff=COURSE_RATE_FF, coursegate=COURSE_STEER_MIN_SPEED, coursehold=COURSE_HOLD_GAP_S, replan3=FSM_REPLAN_THRESH3, parkscan=PARK_SCAN_RATE, parkafter=PARK_SCAN_AFTER_S, blockrad=BLOCKED_REGION_RADIUS)
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

#: Velocity-estimator shape, both env-overridable so an interleaved A/B needs no
#: file edit between flights: SPARX_VEL_WINDOW=0 SPARX_VEL_TAU=0.25 restores the
#: pre-2026-09-02 estimator exactly.
#:
#: The servo closes its loop on /R1/velocity_truth, and on EVERY flight that is a
#: differentiated position -- SpheraPawnState.velocity is all-zero in this vendor
#: build, so the fallback is taken (confirmed in 11 of 11 recent runs). Measured
#: live 2026-09-02 off 2958 samples: Sphera stamps state at ~129 Hz with 2.3x dt
#: jitter, which puts p90 1.0 m/s of noise on a 0.5 m/s signal once differenced
#: between consecutive samples -- hence the old 0.25 s filter, and hence a
#: feedback estimate measured 0.20 s late (cross-correlation peak against the
#: true derivative over a whole flight). Differencing over a fixed 60 ms window
#: divides the same timing error by 8x, so the filter no longer has to. The pair
#: below is strictly better on both axes: 0.94x the tick-to-tick noise at 80 ms
#: of total lag against 250 ms.
VELOCITY_WINDOW_S = float(os.environ.get("SPARX_VEL_WINDOW", "0.06"))
VELOCITY_FILTER_TAU_S = float(os.environ.get("SPARX_VEL_TAU", "0.05"))

GTL_CMD = (
    "python3 -m sparx_agency.robots.ROBOTICAN."
    "rooster_ground_truth_localization --ros-args -p rooster_id:={drone} "
    "-p velocity_window_s:={win} -p velocity_filter_tau_s:={tau}"
).format(drone=DRONE_ID, win=VELOCITY_WINDOW_S, tau=VELOCITY_FILTER_TAU_S)

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
def _controller_rev():
    """A revision string naming whichever experiment's arm this run belongs to.

    One string per distinguishable configuration, because `compare_arms` selects
    arms by exact match. Composed rather than hand-set so a run can never be
    filed under the wrong arm -- the previous campaign lost a whole sample to
    exactly that ("Round 2's sample silently mixed two slew configurations and
    two maps because nothing recorded the revision").
    """
    if BSPLINE_ESCAPE == "true":
        return "v21.0-escape%g" % BSPLINE_ESCAPE_GAIN  # F29/F30 candidate
    if BSPLINE_LIFT == "true":
        return "v19.0-lift%g" % BSPLINE_LIFT_RADIUS  # F28 candidate
    if BSPLINE_OPT_MAX_TIME != 0.01:
        return "v18.0-optms%g" % (BSPLINE_OPT_MAX_TIME * 1000.0)  # F25 candidate
    if BLOCKED_REGION_RADIUS != 1.5:
        return "v22.0-shadow%g" % BLOCKED_REGION_RADIUS  # F33 candidate
    if BLOCKED_SEED == "true":
        return "v17.%d-hazard%g" % (BLOCKED_SEED_STRIKES, BLOCKED_TTL_S)  # F21 candidate
    if YAW_MODE == "blend":
        return "v16.0-blend%g-%g" % (YAW_BLEND_LO, YAW_BLEND_HI)  # F24 candidate
    if YAW_MODE != "course":
        return "v15.0-%syaw%g" % (YAW_MODE[:3], YAW_DOT_FF)  # F23 candidate
    if FSM_REPLAN_THRESH3 != 3.0:
        return "v14.0-replan%g" % FSM_REPLAN_THRESH3  # F22 candidate
    if COURSE_STEER_MIN_SPEED > 0.0 and COURSE_HOLD_GAP_S > 0.0:
        return "v13.0-coursehold%g" % COURSE_HOLD_GAP_S  # F19b candidate
    if COURSE_STEER_MIN_SPEED > 0.0:
        return "v12.0-coursegate%g" % COURSE_STEER_MIN_SPEED  # F19 candidate
    if COURSE_RATE_FF > 0.0:
        return "v11.0-courseff"           # F18 candidate: course-rate feedforward
    if FSM_SLOW_TRAJ_RATIO_MIN >= 1.0:
        return "v10.0-norescale"          # F17 candidate: reverted, kept for history
    if VELOCITY_WINDOW_S <= 0.0:
        return "v2.1d-legacyvel"          # F3 control
    return "v9.0-velwindow"               # F3 candidate, and the F17 control


CONTROLLER_REV = _controller_rev()

#: Lateral is opt-in at the adapter (its default keeps the axis disabled for
#: every non-campaign /cmd_vel producer); the candidate arm enables it at
#: LATERAL_AXIS_CAP, the baseline arm flies the verbatim pre-2026-08-31 stack.
#: Scale the (forward, lateral) demand pair together when either axis cannot
#: deliver it, rather than clipping each independently.
#:
#: The forward axis reaches 1.566 m/s at 900 counts and the lateral only
#: 0.428 m/s at its 600-count cap (both read off the frozen measured curve), and
#: `_servo_axis` clamps each separately -- so an over-large diagonal loses more
#: of its lateral component than its forward one and the aircraft flies a
#: DIFFERENT HEADING than it was asked for. Worked example: a 0.8 m/s demand at
#: 45 deg is flown at 37.1 deg today, and at exactly 45 deg (more slowly) with
#: this on. `ReferenceTracker3D._clamp_velocity` refuses to clip per axis for
#: this reason one layer up, and then the adapter did it anyway.
#:
#: Nearly inert while the nose points along travel; it matters the moment yaw is
#: decoupled, which is why it lands before the yaw experiment. DEFAULT off --
#: present behaviour until its own A/B. SPARX_PRESERVE_DIRECTION=1 enables it.
PRESERVE_DEMAND_DIRECTION = os.environ.get(
    "SPARX_PRESERVE_DIRECTION", "0").strip().lower() in ("1", "true", "yes", "on")

TWIST_ADAPTER_CMD = (
    "bash {repo}/sparx_agency/robots/ROBOTICAN/adapters/"
    "run_twist_control_adapter.sh --rooster-id {drone}{variant}{direction}"
).format(repo=REPO_ROOT, drone=DRONE_ID,
         variant=((" --max-lateral-axis %.0f --altitude-band-m %.2f"
                   % (LATERAL_AXIS_CAP, ALTITUDE_BAND_M)) if USE_LATERAL
                  else " --legacy-feedforward"),
         direction=(" --preserve-demand-direction"
                    if PRESERVE_DEMAND_DIRECTION else ""))

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
