"""Fly the policy. The same missions, once untrained and once fine-tuned.

Runs **inside the isaac-sim container**::

    docker exec isaac-sim /isaac-sim/python.sh \
        /tmp/dev/repo/sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/fly_navdp.py \
        --scene office --missions 6 --seed 4242 --arm trained \
        --server http://127.0.0.1:8888 --out /tmp/dev/navdp_flights --video

Everything offline is open-loop: one prediction, from one frame, scored on its
own. That cannot see the failure that actually matters, which is a small bias
compounding over a hundred inferences until the aircraft is against a wall. This
can, because the policy's output becomes the aircraft's motion and the next
observation.

**The aircraft commits to a plan before asking for another one.** Training and
offline scoring are frame-by-frame, and flying that way -- re-inferring on a
timer and steering at whatever the newest prediction says -- means the aircraft
executes the first third of a metre of every plan and no more, so the route the
policy actually predicted never gets flown. Instead one prediction is anchored
where it was made and flown as a route until roughly half of it is behind the
aircraft; only then is the policy asked again. ``--infer-hz`` is now a rate
*ceiling* rather than a schedule. The rule, and the guards that keep a
commitment from becoming a trap, live in
:mod:`sparx_agency.core.planning.vlas.common.plan_commit`.

**And it is flown by the expert's own follower.** The committed route goes
straight into :class:`~sparx_agency.tasks.planning.sim_flight_recording.path_follower.PathFollower`
with the ``FollowSpec`` the demonstrations were collected under -- Hermite
smoothing, a speed- and curvature-scaled carrot, an integrated rate-limited
heading aimed ``yaw_lookahead`` *beyond* that carrot, and a stop-and-pivot for
corners too sharp to fly through. This loop used to hand-roll all of that and
got two things wrong that the collection follower had already solved: it aimed
the nose at the carrot, which ``FollowSpec.yaw_lookahead`` documents as the
cause of the flown path weaving from side to side, and it slewed the yaw
setpoint from the measured pose instead of the previous command, which held the
achieved turn rate to 0.6 deg/s against the 40 deg/s asked for. Flying the
policy's route the way the expert's route is flown is also the only way the
comparison means anything.

The policy runs on the **host**, not here. Isaac's Python has torch but not
``diffusers``, and NavDP's scheduler needs it; rather than mutate the container,
inference is served over the HTTP contract this repo already uses for NavDP
(``serve/navdp_trt_server.py``) and reached with the client already in
``core`` -- the same contract the FALCON nodes speak, so what flies here is
exactly what would fly on the real aircraft.

    # host, navdp env, one per arm (different ports):
    python -m ...navdp.serve.navdp_trt_server --backend torch --port 8888 \
        --ckpt ~/Downloads/navdp-cross-modal.ckpt            # baseline
    python -m ...navdp.serve.navdp_trt_server --backend torch --port 8889 \
        --ckpt ~/navdp_world_goal/navdp-world-goal.ckpt      # fine-tuned

Both arms fly the **same missions from the same seed**, and every mission is
scored against the surveyed map: how close the aircraft really came to
geometry, whether it reached the goal, how far it flew to get there. That is a
different and much harder question than "was the predicted trajectory clear",
and it is the one that decides whether the fine-tune was worth it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CRUISE_SPEED_MPS = 1.0
LOOKAHEAD_M = 1.2
INFER_HZ = 3.0
GOAL_TOLERANCE_M = 1.5
STALL_DISTANCE_M = 0.6
STALL_WINDOW_S = 20.0
SECONDS_PER_METRE = 5.0
FIXED_OVERHEAD_S = 60.0
COLLISION_CLEARANCE_M = 0.05     # airframe centre this close to geometry = a hit
COMMIT_FRACTION = 0.5            # fly half a prediction before asking for another
COMMIT_TIMEOUT_S = 8.0
MIN_COMMIT_M = 0.4
MAX_DEVIATION_M = 2.0
SEARCH_DWELL_S = 2.0             # look this long before trying another heading


def sample_missions(world_map, seed: int, count: int, spec) -> List:
    """Draw ``count`` missions. Same seed, same missions, for every arm."""
    from sparx_agency.core.planning.mission import sample_start_goal

    rng = np.random.default_rng(seed)
    missions = []
    for _ in range(count):
        missions.append(sample_start_goal(
            world_map.grid, rng, clearance_m=spec.clearance_m,
            min_separation_m=spec.min_separation_m,
            max_separation_m=spec.max_separation_m,
            start_yaw_jitter_rad=spec.start_yaw_jitter_rad,
            region=world_map.landing_region))
    return missions


def _empty_result(mission, start, outcome: str, detail: str = "") -> Dict:
    """A result for a mission that never flew, with every key the caller reads.

    The summary averages ``min_clear_m``, ``path_len_m``, ``duration_s`` and
    ``goal_error_m`` across missions and counts ``collided``. A short dict here
    -- which is what an ``arm_timeout`` used to return, and a cold PX4 makes
    that the common case -- raises ``KeyError`` in the per-mission print, and
    since ``summary.json`` is only written after the loop, every mission already
    flown is lost with it. NaN propagates through ``np.mean`` visibly; a missing
    key does not.
    """
    return {
        "outcome": outcome,
        "detail": detail,
        "reached": False,
        "start_xy": [float(mission.start.x), float(mission.start.y)],
        "goal_xy": [float(mission.goal.x), float(mission.goal.y)],
        "separation_m": float(mission.separation_m),
        "goal_error_m": float("nan"),
        "min_clear_m": float("nan"),
        "p5_clear_m": float("nan"),
        "mean_clear_m": float("nan"),
        "collided": False,
        "path_len_m": 0.0,
        "duration_s": 0.0,
        "inferences": 0,
        "commitments": 0,
        "transport_failures": 0,
    }


def fly_mission(loop, px4, adapter, client, scene, mission, args,
                recorder=None, track_log=None) -> Dict:
    """Fly one A-to-B mission with the policy closing the loop.

    Args:
        track_log: Optional :class:`~.track_log.TrackLog`. Given one, every
            inference's proposed trajectory is recorded in world coordinates,
            which is what a map panel is drawn from afterwards. The flight is
            identical either way -- this only observes.

    Returns:
        A result dict: outcome, whether the goal was reached, the worst
        clearance the aircraft actually achieved against the surveyed map,
        collisions, path length and duration.
    """
    from sparx_agency.core.planning.vlas.common.plan_commit import (
        CommitSpec, PlanCommitExecutor,
    )
    from sparx_agency.core.planning.vlas.common.yaw_search import (
        YawSearch, YawSearchSpec,
    )
    from sparx_agency.core.planning.vlas.navdp.geometry import (
        point_to_pointgoal, world_to_body_2d,
    )
    from sparx_agency.core.common.types import Pose3D
    from sparx_agency.tasks.planning.sim_flight_recording.episode import (
        CRASH_HOLD_S, CRASH_TILT_DEG, TAKEOFF_TOLERANCE_M, arm_for_offboard,
        attitude_deg, hold_velocity, slew_towards,
    )
    from sparx_agency.tasks.planning.sim_flight_recording.path_follower import (
        FollowSpec, PathFollower, build_trajectory,
    )

    vehicle = adapter.vehicle
    goal = (mission.goal.x, mission.goal.y)
    altitude = args.altitude

    start = vehicle.state.position
    failure = arm_for_offboard(loop, px4, start, mission.start.yaw, altitude)
    if failure is not None:
        return _empty_result(mission, start, "arm_timeout", detail=failure)

    # The client builds the K matrix itself from an Intrinsics; handing it the
    # matrix instead fails inside reset with 'list' object has no attribute 'fx'.
    client.reset(adapter.intrinsics, stop_threshold=-999, batch_size=1)

    budget = FIXED_OVERHEAD_S + SECONDS_PER_METRE * mission.separation_m
    deadline = loop.sim_time + budget
    # One prediction at a time, flown as a route. --infer-hz becomes the ceiling
    # this can ask at, not the rate it asks at.
    executor = PlanCommitExecutor(CommitSpec(
        fraction=args.commit_fraction,
        lookahead_m=args.lookahead,
        min_commit_m=args.min_commit,
        max_commit_s=args.commit_timeout,
        max_deviation_m=args.max_deviation,
        min_period_s=1.0 / max(args.infer_hz, 0.1)))
    target_world = (float(start[0]), float(start[1]))
    # Two headings, and the difference between them is the whole yaw law: the
    # TARGET is where the route says to point, the COMMAND is the setpoint sent
    # to PX4, which is only ever allowed to move turn_yaw_rate per step away
    # from where it already was.
    yaw_command = mission.start.yaw
    # Where to hold when there is no route to fly. Latched at the moment the
    # aircraft runs out of plan, so a stopped policy parks it where it stopped
    # rather than letting it slide; cleared as soon as there is a route again.
    hold_xy = None                          # type: Optional[Tuple[float, float]]
    # Dwell long enough on each heading to actually ask from it: inference is
    # gated by the rate floor AND by the render tick, so a couple of seconds is
    # a handful of questions rather than none.
    searcher = YawSearch(YawSearchSpec(dwell_s=args.search_dwell))
    track: List[Tuple[float, float]] = []
    tilted_since: Optional[float] = None
    stall_reference = (float(start[0]), float(start[1]), loop.sim_time)
    inferences, transport_failures = 0, 0
    outcome = "flight_timeout"
    # arm_for_offboard only arms and switches modes -- it does not climb, so the
    # aircraft is still on the ground when this function's loop starts. Running
    # NavDP inference immediately would have it steering while the airframe is
    # still climbing through ground effect and out of takeoff attitude, so
    # inference is held off until the aircraft is at cruise altitude; the climb
    # itself already happens for free: hold_velocity climbs to `altitude`.
    climbed = False
    # Where to hold while climbing: where the aircraft actually is, not the
    # mission's nominal start, so the hold cannot fight a spawn offset.
    takeoff_xy = (float(start[0]), float(start[1]))
    # The SAME flight parameters the expert demonstrations were collected with,
    # so a NavDP route is flown the way an expert route was. Everything except
    # the speeds is left at the collection defaults on purpose -- each of them
    # was tuned against recorded flights and is documented in FollowSpec.
    follow_spec = FollowSpec(cruise_speed=args.cruise,
                             max_speed=max(args.cruise * 1.25, args.cruise + 0.2))

    def build_follower(plan, from_yaw):
        """A follower for a freshly committed route, starting under the aircraft.

        The WHOLE prediction is handed over, not just the committed half: the
        carrot and, more importantly, ``yaw_lookahead`` need route ahead of the
        commit point to aim at. The commitment governs when to ask the policy
        again; it is not a fence the follower has to see.
        """
        waypoints = [(float(px), float(py), altitude, 0.0)
                     for px, py in plan.world_xy[1:]]
        return PathFollower(
            build_trajectory(Pose3D(plan.anchor[0], plan.anchor[1], altitude,
                                    float(from_yaw)), waypoints, follow_spec),
            follow_spec, initial_yaw=float(from_yaw))

    follower = None
    # The clock the recording, the chase camera and the flown track all start on.
    # Read before the first step so it is the instant of track[0], which is what
    # a map panel needs to line itself up against the video.
    flight_started = loop.sim_time

    while loop.sim_time < deadline:
        rendered = loop.step()
        position = vehicle.state.position
        pose = adapter.pose_flu()
        track.append((float(position[0]), float(position[1])))
        # Re-derived every step: a stale flag would keep the stall detector
        # disarmed for the rest of the flight.
        turning = looking = False

        if not climbed and position[2] >= altitude - TAKEOFF_TOLERANCE_M:
            climbed = True
            # Don't let the stall check fire for time spent only climbing.
            stall_reference = (float(position[0]), float(position[1]), loop.sim_time)

        if rendered and recorder is not None:
            recorder.capture(stamp_s=loop.sim_time)

        if climbed:
            # Progress along the standing commitment, and whether it is over.
            # Evaluated every control step, not every render: the carrot has to
            # advance with the aircraft or the route is not being followed, it
            # is being aimed at.
            #
            # Deliberately NOT guarded, unlike the commit below. A non-finite
            # *prediction* is a policy hiccup and the flight can carry on without
            # it; a non-finite *pose* means the simulator or its estimator has
            # broken, and every number this episode goes on to record would be
            # meaningless. This is an offline harness, so failing loudly and
            # losing the run is better than writing a corrupt episode into the
            # fine-tuning set. The FALCON node makes the opposite call, because
            # there an aircraft is in the air.
            tick = executor.tick(pose[0], pose[1], loop.sim_time)
            if rendered and tick.replan_reason is not None:
                # Why this inference happened, kept before the tick is refreshed
                # below -- it is the one line that says whether the commitment
                # was flown or abandoned.
                reason = tick.replan_reason
                # Marked before the request, so a dropped inference still costs
                # a period and a dead server is not asked 250 times a second.
                executor.mark_attempt(loop.sim_time)
                frame = adapter.capture_frame(stamp_s=loop.sim_time)
                forward, left = world_to_body_2d(goal[0], goal[1],
                                                 pose[0], pose[1], pose[2])
                gx, gy = point_to_pointgoal(forward, left)
                result = client.pointgoal_step(
                    np.ascontiguousarray(frame.rgb[:, :, :3]), frame.depth, gx, gy,
                    altitude=altitude)
                inferences += 1
                trajectory = None
                if result is None:
                    transport_failures += 1
                else:
                    trajectory = client.best_trajectory(result)
                    # Anchored at the pose the frame was captured from: `pose`
                    # was read at the top of this step, and nothing has moved
                    # the aircraft since, so it is the pose behind this image.
                    # Anchoring anywhere later lays the route down ahead of
                    # where the policy actually looked.
                    try:
                        plan = executor.commit(trajectory, pose, loop.sim_time)
                    except ValueError as exc:
                        # A non-finite prediction or pose. Counted as the
                        # transport failure it resembles and dropped exactly
                        # like `result is None` above: trajectory back to None,
                        # no follower, so the position hold below takes over and
                        # the next step asks again at the rate floor. Falling
                        # through rather than skipping the step matters -- the
                        # track log, the target and the physics all live below.
                        print("[fly_navdp] rejected prediction: %s" % (exc,))
                        transport_failures += 1
                        trajectory = None
                    else:
                        tick = executor.tick(pose[0], pose[1], loop.sim_time)
                        # A prediction with no length is a route of identical
                        # points, and build_trajectory rightly refuses to smooth
                        # one -- so check before asking rather than losing a
                        # four-minute flight to the policy saying "stop". No
                        # follower means the hold below takes over, which is the
                        # correct response to a policy that has stopped.
                        follower = (build_follower(plan, pose[2])
                                    if plan.total_arc_m >= args.min_commit else None)
                if track_log is not None:
                    track_log.add(loop.sim_time, pose, trajectory,
                                  tick.target or target_world,
                                  commit_index=(executor.plan.commit_index
                                                if trajectory is not None else None),
                                  reason=reason)
            if tick.target is not None:
                target_world = tick.target

            # Nothing worth flying is a POSITION HOLD, not a zero velocity. A
            # policy that has stopped -- and the pretrained one stops on most
            # frames -- leaves a route with no length, so the follower is handed
            # a route it is already at the end of and commands nothing. PX4 is in
            # velocity control: "command nothing" has no position feedback, so
            # momentum and estimator bias are never corrected and the aircraft
            # slides. Measured on a real flight: 5.3 m of net drift over 79 s
            # while every one of 198 predictions said stop. That drift then
            # moves the aircraft into geometry it never chose to approach and
            # poisons every clearance number the mission reports.
            #
            # This is the same one-sidedness the climb below documents, in the
            # same file, for the same reason -- it was simply never applied to
            # the case where the *policy* has nothing to say.
            flyable = follower is not None and tick.commit_arc_m >= args.min_commit
            if flyable:
                # The expert's own follower, on the policy's route: a Hermite
                # spline through the prediction, a speed- and curvature-scaled
                # carrot along it, the heading aimed yaw_lookahead AHEAD of that
                # carrot, and a stop-and-pivot when the corner is too sharp to
                # fly through. Aiming the nose at the carrot instead is what
                # made the collected flights weave -- FollowSpec.yaw_lookahead
                # documents that, and it is the same mistake this loop made.
                follow = follower.update(position, pose[2],
                                         vehicle.state.linear_velocity, loop.dt)
                turning = follow.turning
                if follow.done:
                    flyable = False     # route flown out; hold rather than coast
                else:
                    hold_xy = None
                    searcher.reset()    # a route means the looking worked
                    vx, vy, vz = follow.velocity
                    yaw_command = follow.yaw
            if not flyable:
                if hold_xy is None:
                    hold_xy = (float(position[0]), float(position[1]))
                vx, vy, vz = hold_velocity(
                    position, (hold_xy[0], hold_xy[1], altitude), follow_spec)
                # Hold the POSITION and SWEEP the heading until the policy can
                # see somewhere to go. A forward-looking policy asked about a
                # view with no route in it answers "stop", correctly; asking it
                # the same question forever is the bug. Holding the heading as
                # well as the position guarantees exactly that, and it
                # deadlocked both arms for a whole flight budget on the mission
                # that spawns with the goal 116 degrees off the nose.
                #
                # The sweep looks at the goal first and widens either side --
                # see core/planning/vlas/common/yaw_search. Turning on the spot
                # costs nothing and risks nothing, and unlike the drift this
                # replaced it does not fake progress while doing it.
                yaw_command = slew_towards(
                    yaw_command,
                    searcher.heading(
                        math.atan2(goal[1] - pose[1], goal[0] - pose[0]),
                        pose[2], loop.sim_time),
                    follow_spec.turn_yaw_rate, loop.dt)
                # A first look around is deliberate, and it takes longer than
                # the stall window; only once every heading has been tried and
                # none produced a route is a stationary aircraft really stuck.
                looking = searcher.sweeps == 0
        else:
            # Station-keep over the take-off point until the aircraft is at cruise
            # altitude. hold_velocity flies AT the point regardless of heading,
            # which is what "stay put" needs and what a travel law cannot do:
            # anything that scales speed by the cosine of the heading error
            # corrects drift into the rear hemisphere at exactly zero speed, and
            # that one-sidedness let the aircraft wander 3.5 m off the take-off
            # point during the climb -- before a single inference, so NavDP's
            # first observation came from somewhere the mission never chose.
            vx, vy, vz = hold_velocity(position, (takeoff_xy[0], takeoff_xy[1],
                                                  altitude), follow_spec)
        # yaw_command comes from the follower, which INTEGRATES its own
        # rate-limited yaw rate. Nothing here slews it: this loop used to, from
        # the measured pose rather than the previous command, which capped the
        # setpoint 0.24 deg ahead of the aircraft at a 250 Hz step and held the
        # achieved turn rate to 0.6 deg/s against the 40 deg/s asked for. While
        # holding position the command simply stands still, which is what a hold
        # wants.
        px4.send_velocity_world(vx, vy, vz, yaw_command)

        if math.hypot(position[0] - goal[0], position[1] - goal[1]) < args.goal_tolerance:
            outcome = "reached"
            break

        roll, pitch, _ = attitude_deg(vehicle)
        if max(abs(roll), abs(pitch)) > CRASH_TILT_DEG:
            tilted_since = tilted_since or loop.sim_time
            if loop.sim_time - tilted_since > CRASH_HOLD_S:
                outcome = "crashed"
                break
        else:
            tilted_since = None

        moved = math.hypot(position[0] - stall_reference[0], position[1] - stall_reference[1])
        if moved > STALL_DISTANCE_M or turning or looking:
            # `turning` is the follower deliberately holding position to pivot
            # onto a corner too sharp to fly through. It is not a wedged
            # aircraft, and FollowSpec says so in as many words -- treating it
            # as one would call every hard turn a stall.
            stall_reference = (float(position[0]), float(position[1]), loop.sim_time)
        elif loop.sim_time - stall_reference[2] > STALL_WINDOW_S:
            outcome = "stalled"
            break

    flown = np.asarray(track, dtype=np.float64)
    clearance = scene.clearance(flown[:, 0], flown[:, 1]) if flown.shape[0] else np.array([0.0])
    steps = np.linalg.norm(np.diff(flown, axis=0), axis=1) if flown.shape[0] > 1 else np.array([0.0])
    final = flown[-1] if flown.shape[0] else np.array(start[:2])
    if track_log is not None:
        track_log.set_flown(track, started_s=flight_started, dt=loop.dt)
    return {
        "outcome": outcome,
        "reached": outcome == "reached",
        "start_xy": [float(mission.start.x), float(mission.start.y)],
        "goal_xy": [float(goal[0]), float(goal[1])],
        "separation_m": float(mission.separation_m),
        "goal_error_m": float(math.hypot(final[0] - goal[0], final[1] - goal[1])),
        "min_clear_m": float(clearance.min()),
        "p5_clear_m": float(np.percentile(clearance, 5)),
        "mean_clear_m": float(clearance.mean()),
        "collided": bool(clearance.min() < COLLISION_CLEARANCE_M),
        "path_len_m": float(steps.sum()),
        "duration_s": float(len(track) * loop.dt),
        "inferences": inferences,
        "commitments": executor.commitments,
        "transport_failures": transport_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="office")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--missions", type=int, default=6,
                        help="size of the mission set the seed draws; with "
                             "--mission-index only one of them is flown")
    parser.add_argument("--mission-index", type=int, default=-1,
                        help="fly only this mission of the set, in a session of "
                             "its own. This is the correct way to fly more than "
                             "one: the aircraft cannot be repositioned between "
                             "missions, so a single session can only fly the "
                             "first honestly. -1 flies them all in one session.")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--arm", required=True, help="label for this set of weights")
    parser.add_argument("--server", default="http://127.0.0.1:8888")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev-root", default="/tmp/dev")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--cruise", type=float, default=CRUISE_SPEED_MPS)
    parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_M)
    parser.add_argument("--infer-hz", type=float, default=INFER_HZ,
                        help="rate CEILING on the policy, not a schedule. The "
                             "commitment decides when to ask again; this only "
                             "stops 'when' from being 'every render'")
    parser.add_argument("--commit-fraction", type=float, default=COMMIT_FRACTION,
                        help="share of each prediction the aircraft flies before "
                             "asking again. 0.5 of NavDP's 24 waypoints commits "
                             "through waypoint 12, about 2.4 m")
    parser.add_argument("--commit-timeout", type=float, default=COMMIT_TIMEOUT_S,
                        help="abandon a commitment that has taken this long "
                             "(seconds): yawing in place, blocked, or stuck")
    parser.add_argument("--min-commit", type=float, default=MIN_COMMIT_M,
                        help="a commitment shorter than this is a predicted stop, "
                             "not a route; ask again instead of crawling")
    parser.add_argument("--max-deviation", type=float, default=MAX_DEVIATION_M,
                        help="abandon a commitment the aircraft is this far off")
    parser.add_argument("--search-dwell", type=float, default=SEARCH_DWELL_S,
                        help="when the policy will not move, how long to hold "
                             "each heading of the look-around sweep before "
                             "trying the next. Must cover several inferences")
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_TOLERANCE_M)
    parser.add_argument("--video", action="store_true")
    # campaign_setup.bring_up reads all of these off the namespace, so they have
    # to exist here even where this script has no reason to vary them. Defaults
    # mirror collect.py's.
    parser.add_argument("--resolution", default=None,
                        help="camera WxH; defaults to the platform's own 504x392")
    # type=Path, as collect.py has it: spawn_vehicle does path arithmetic on
    # this straight away, so a bare string dies with "unsupported operand
    # type(s) for /: 'str' and 'str'" only once Kit has finished booting.
    parser.add_argument("--pegasus-root", type=Path,
                        default=Path("/tmp/dev/PegasusSimulator/extensions/pegasus.simulator"))
    parser.add_argument("--px4-dir", type=Path, default=Path("/tmp/dev/PX4-Autopilot"))
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true",
                        help="throttle the simulation to wall-clock time")
    parser.add_argument("--stream", action="store_true",
                        help="WebRTC livestream on :49100")
    parser.add_argument("--settle-s", type=float, default=30.0,
                        help="simulated seconds to let PX4's estimator converge "
                             "before the first mission. Without it the first "
                             "mission of each arm flies on a half-converged EKF2 "
                             "and the two arms are not comparable")
    parser.add_argument("--no-vision", dest="vision", action="store_false",
                        help="fly on PX4's simulated GNSS and magnetometer rather "
                             "than the simulator's true pose. Both arms must agree, "
                             "or the comparison measures the estimator")
    parser.set_defaults(vision=True)
    args = parser.parse_args()

    repo = Path(args.dev_root) / "repo"
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from sparx_agency.core.planning.vlas.navdp.client import NavDPPointgoalClient
    from sparx_agency.tasks.planning.sim_flight_recording import (
        campaign_setup, flight_session, px4_launch, px4_params,
    )
    from sparx_agency.tasks.planning.sim_flight_recording.episode_plan import EpisodeSpec
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
        Scene, SceneConfig,
    )
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.track_log import (
        TrackLog,
    )

    spec = EpisodeSpec(altitude_m=args.altitude)
    world_map = campaign_setup.load_map(args.scene, args.altitude, spec)
    scene = Scene.load(SceneConfig(scene=args.scene, altitude_m=args.altitude))
    missions = sample_missions(world_map, args.seed, args.missions, spec)
    print(f"[fly] arm={args.arm}  {len(missions)} missions from seed {args.seed}",
          flush=True)
    for index, mission in enumerate(missions):
        print(f"[fly]   {index}: ({mission.start.x:.1f}, {mission.start.y:.1f}) -> "
              f"({mission.goal.x:.1f}, {mission.goal.y:.1f})  "
              f"{mission.separation_m:.1f} m", flush=True)

    # One mission per process is the only sound shape here. A mission begins
    # wherever the aircraft already is -- there is no teleport in the vehicle
    # adapter and none in flight_session, because collect.py spawns a fresh
    # Isaac and PX4 for every episode and never needed one. Flying several in
    # one session therefore starts mission N+1 from wherever mission N ended,
    # and once one crashes the aircraft is on the ground for good: every later
    # mission returns in three seconds having flown 0.1 m. Sampling still uses
    # the full list so the seed picks the same missions for every arm.
    selected = args.mission_index
    if selected >= 0:
        if selected >= len(missions):
            raise ValueError(
                f"--mission-index {selected} is past the {len(missions)} "
                f"missions seed {args.seed} produces")
        missions = [missions[selected]]
        print(f"[fly] flying mission {selected} only", flush=True)

    out = Path(args.out).expanduser() / args.arm
    out.mkdir(parents=True, exist_ok=True)
    client = NavDPPointgoalClient(args.server)

    # Order matters and is collect.py's: clear the persisted parameter store
    # first (PX4 keeps every parameter it is ever sent, so one arm's settings
    # would otherwise leak into the next), start PX4, then boot Kit.
    px4_launch.clear_saved_parameters(args.px4_dir, args.worker)
    px4_process = px4_launch.launch_px4(
        args.px4_dir, instance=args.worker,
        log_path=out / f"px4_worker{args.worker}.log",
        boot_params=px4_params.boot_params(args.vision))
    simulation_app = flight_session.boot_isaac(stream=args.stream)

    results: List[Dict] = []
    try:
        loop, adapter, px4_link, chase = campaign_setup.bring_up(
            simulation_app, args, missions[0].start, heartbeat_timeout_s=300.0)
        campaign_setup.configure_px4(loop, px4_link, px4_params.all_params(args.vision), 3.0)
        campaign_setup.settle_estimator(loop, px4_link, args.settle_s)
        if not campaign_setup.wait_until_armable(loop, px4_link):
            print("[fly] PX4 never became armable; nothing to fly", flush=True)
            return

        for index, mission in enumerate(missions):
            # The true index in the seed's mission set, not the position in this
            # session's list. Everything named after a mission uses this -- the
            # recording, the MP4 and the stored result -- so that ten one-mission
            # sessions in the same directory do not all call themselves "00".
            mission_id = selected if selected >= 0 else index
            recorder = None
            if args.video:
                recorder = flight_session.FlightRecorder(
                    adapter, out / f"mission_{mission_id:02d}", rate_hz=args.rate_hz,
                    camera_height_m=args.altitude,
                    video_out=out / f"mission_{mission_id:02d}.mp4",
                    video_source="chase", chase_camera=chase)
            # Logged whenever video is, because the map panel is drawn from it
            # and a video without the panel is half the comparison.
            log = TrackLog((mission.goal.x, mission.goal.y),
                           (mission.start.x, mission.start.y)) if args.video else None
            started = time.time()
            result = fly_mission(loop, px4_link, adapter, client, scene, mission,
                                 args, recorder, track_log=log)
            result.update({"mission": mission_id, "arm": args.arm,
                           "wall_s": round(time.time() - started, 1)})
            if log is not None:
                log.write(out / f"mission_{mission_id:02d}_track.json",
                          extra={"scene": args.scene, "arm": args.arm,
                                 "infer_hz": args.infer_hz, **result})
            if recorder is not None:
                recorder.finish({"scene": args.scene, "arm": args.arm, **result})
            results.append(result)
            print(f"[fly] mission {result['mission']}: {result['outcome']}  "
                  f"min_clear {result['min_clear_m']:.2f} m  "
                  f"goal_error {result['goal_error_m']:.2f} m  "
                  f"{result['commitments']} plans over "
                  f"{result['path_len_m']:.1f} m", flush=True)
            # Written every mission, not at the end: a run that dies on mission
            # five should not also lose the four that flew. One-mission runs get
            # their own file so that twenty sessions in the same directory
            # accumulate instead of overwriting each other.
            name = "results.json" if selected < 0 else f"results_{selected:02d}.json"
            (out / name).write_text(json.dumps(results, indent=2))
    except Exception:
        # Kit runs with --/app/fastShutdown=True, so ``simulation_app.close()``
        # in the finally below tears the process down before the interpreter
        # gets to print the traceback. Without this the run looks like a clean
        # "no missions completed" and the real error is never seen.
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        _write_summary(out, args.arm, results)
        # Leaving either alive costs the *other* arm its ports and its GPU
        # memory, and this comparison is two sequential runs.
        simulation_app.close()
        px4_launch.terminate_px4(px4_process, instance=args.worker)


def _write_summary(out: Path, arm: str, results: List[Dict]) -> None:
    """Aggregate whatever flew, even when the run ended early."""
    if not results:
        print(f"[fly] {arm}: no missions completed", flush=True)
        return
    reached = sum(1 for r in results if r["reached"])
    collided = sum(1 for r in results if r.get("collided"))
    mean = lambda key: float(np.nanmean([r.get(key, float("nan")) for r in results]))
    summary = {
        "arm": arm, "missions": len(results), "reached": reached,
        "collisions": collided,
        "min_clear_m": mean("min_clear_m"),
        "path_len_m": mean("path_len_m"),
        "duration_s": mean("duration_s"),
        "goal_error_m": mean("goal_error_m"),
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fly] {arm}: reached {reached}/{len(results)}, "
          f"{collided} collisions, mean min clearance "
          f"{summary['min_clear_m']:.2f} m", flush=True)


if __name__ == "__main__":
    main()
