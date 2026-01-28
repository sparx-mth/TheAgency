"""
Main simulation loop - CLEAN VERSION.

This file is PURE ORCHESTRATION - no algorithmic code.
All algorithms (planning, smoothing, tracking) come from core modules.

Architecture:
1. Build obstacle map from config
2. Global Planning: RRT* creates path
3. Smoother: Convert path to trajectory
4. Tracker: Pure pursuit follows trajectory
5. Visualization: Show progress, allow click-to-place obstacles

Click-placed obstacles are STATIC - they don't move.
The simulation just tracks the pre-planned trajectory.
"""
from __future__ import annotations

from typing import Tuple

from sparx_agency.core.common.types import Pose2D, Pose3D, Twist3D, State3D
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother, HermiteParams
from sparx_agency.core.planning.smoothers.minsnap import MinSnapSmoother, MinSnapParams
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker, PurePursuitParams

from sparx_agency.tasks.planning.dynamic_interaction_environment.config import ScenarioConfig
from sparx_agency.tasks.planning.dynamic_interaction_environment.drone_sim import DroneSimulator, DroneSimParams
from sparx_agency.tasks.planning.dynamic_interaction_environment.obstacle_map import ObstacleMap
from sparx_agency.tasks.planning.dynamic_interaction_environment.visualization import Visualizer


def build_obstacle_map(cfg: ScenarioConfig) -> ObstacleMap:
    """Build obstacle map from config."""
    m = cfg.map
    obs_map = ObstacleMap(m.width, m.height, m.origin_x, m.origin_y, m.resolution)
    for o in m.obstacles:
        if o.type == "rect":
            obs_map.add_rectangle(o.x, o.y, o.w, o.h)
        else:
            obs_map.add_circle(o.x, o.y, o.r)
    return obs_map


def run_simulation(cfg: ScenarioConfig) -> bool:
    """
    Run the trajectory tracking simulation.

    Returns True if goal was reached, False otherwise.
    """
    print(f"\n{'=' * 60}\nSCENARIO: {cfg.name}\n{'=' * 60}")

    # -------------------------------------------------------------------------
    # 1. Build obstacle map
    # -------------------------------------------------------------------------
    obs_map = build_obstacle_map(cfg)
    costmap = obs_map.to_costmap(cfg.map.inflate_radius, include_placed=False)
    print(f"Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # -------------------------------------------------------------------------
    # 2. Global planning (from core)
    # -------------------------------------------------------------------------
    start = Pose2D(x=cfg.start[0], y=cfg.start[1])
    goal = Pose2D(x=cfg.goal[0], y=cfg.goal[1])
    print(f"Planning: ({start.x:.2f}, {start.y:.2f}) -> ({goal.x:.2f}, {goal.y:.2f})")

    p = cfg.planner
    planner = RRTStarOmplPlanner(
        RRTStarOmplParams(
            timeout=p.timeout,
            use_clearance_objective=p.use_clearance,
            clearance_weight=p.clearance_weight,
            interpolation_spacing=p.interpolation_spacing,
        )
    )
    plan_result = planner.plan(PlanRequest(start=start, goal=goal, frame_id="map"), costmap)

    if not plan_result.ok:
        print(f"Planning failed: {plan_result.message}")
        return False
    print(f"Path: {len(plan_result.path.points)} waypoints, {plan_result.path.length():.2f}m")

    # -------------------------------------------------------------------------
    # 3. Smoothing (from core)
    # -------------------------------------------------------------------------
    s = cfg.smoother
    if s.type == "hermite":
        smoother = HermiteSmoother(HermiteParams(
            dt=s.dt,
            nominal_speed_xy=s.nominal_speed,
            tangent_scale=s.tangent_scale
        ))
    else:
        smoother = MinSnapSmoother(MinSnapParams(
            dt=s.dt,
            nominal_speed_xy=s.nominal_speed
        ))

    trajectory = smoother.smooth(SmootherRequest(path=plan_result.path))
    print(f"Trajectory: {trajectory.total_time:.2f}s")

    # -------------------------------------------------------------------------
    # 4. Tracker (from core)
    # -------------------------------------------------------------------------
    t = cfg.tracker
    tracker = PurePursuitTracker(
        PurePursuitParams(
            holonomic=t.holonomic,
            base_lookahead=t.base_lookahead,
            min_lookahead=t.min_lookahead,
            max_lookahead=t.max_lookahead,
            min_speed=t.min_speed,
            cruise_speed=t.cruise_speed,
            max_speed=t.max_speed,
            curvature_speed_factor=t.curvature_speed_factor,
            curvature_lookahead_factor=t.curvature_lookahead_factor,
            goal_tolerance=t.goal_tolerance,
            path_tolerance=t.path_tolerance,
        )
    )
    tracker.reset()

    # -------------------------------------------------------------------------
    # 5. Drone simulator
    # -------------------------------------------------------------------------
    sc = cfg.simulator
    sim_params = DroneSimParams(
        dt=sc.dt,
        tau_velocity=sc.tau_velocity,
        tau_yaw=sc.tau_yaw,
        max_speed_xy=sc.max_speed_xy,
        max_speed_z=sc.max_speed_z,
        max_yaw_rate=sc.max_yaw_rate,
        collision_radius=sc.collision_radius,
        wind_enabled=sc.wind_enabled,
        wind_mean=sc.wind_mean,
        wind_std=sc.wind_std,
        wind_tau=sc.wind_tau,
        gust_enabled=sc.gust_enabled,
        gust_probability=sc.gust_probability,
        gust_magnitude=sc.gust_magnitude,
        gust_duration=sc.gust_duration,
        process_noise_std=sc.process_noise_std,
        position_noise_std=sc.position_noise_std,
        velocity_noise_std=sc.velocity_noise_std,
        yaw_noise_std=sc.yaw_noise_std,
    )

    collision_fn = lambda x, y: obs_map.is_occupied(x, y, margin=sim_params.collision_radius)
    sim = DroneSimulator(params=sim_params, obstacle_fn=collision_fn, seed=cfg.seed)
    sim.reset(x=start.x, y=start.y, z=0.0)

    # -------------------------------------------------------------------------
    # 6. Visualizer
    # -------------------------------------------------------------------------
    vis = Visualizer(
        obstacle_map=obs_map,
        trajectory=trajectory,
        raw_path=plan_result.path,
        drone_radius=sim_params.collision_radius,
        click_obstacle_radius=cfg.click_obstacles.default_radius,
        click_obstacles_enabled=cfg.click_obstacles.enabled,
    )

    print("\nRunning... (SPACE=pause, Q=quit, Click=place obstacle, C=clear)")

    dt = sim_params.dt
    t_sim = 0.0
    success = False

    # -------------------------------------------------------------------------
    # Main loop - PURE TRACKING, no algorithm code
    # -------------------------------------------------------------------------
    while t_sim < cfg.max_time:
        # Get measured state
        x, y, z, vx, vy, vz, yaw = sim.get_measured_state()
        state = State3D(
            pose=Pose3D(x=x, y=y, z=z, yaw=yaw),
            twist=Twist3D(vx=vx, vy=vy, vz=vz, yaw_rate=0.0),
        )

        # Track trajectory (all logic in core)
        tr = tracker.step(TrackerRequest(state=state, trajectory=trajectory, t=t_sim))

        # Step simulator
        sim_state, info = sim.step(tr.command.x, tr.command.y, tr.command.z, tr.command.yaw_rate)

        # Update visualizer
        lookahead = (tr.reference.x, tr.reference.y, tr.reference.z) if tr.reference else None
        running = vis.update(
            drone_position=(sim_state.x, sim_state.y, sim_state.z),
            drone_velocity=(sim_state.vx, sim_state.vy, sim_state.vz),
            drone_yaw=sim_state.yaw,
            time=t_sim,
            cross_track_error=tr.metadata.get("cross_track_error", 0.0),
            progress_idx=tr.metadata.get("progress_idx", 0),
            lookahead_point=lookahead,
            gust_active=info["gust_active"],
            collision=info["collision"],
            done=tr.metadata.get("done", False),
            failed=tr.metadata.get("failed", False),
            wind=info["wind"],
        )

        if not running:
            break

        # Check goal reached
        if tr.metadata.get("done"):
            success = True
            print(f"\n✓ GOAL REACHED at t={t_sim:.2f}s")
            break

        if tr.metadata.get("failed"):
            print(f"\n✗ FAILED: {tr.metadata.get('reason', 'unknown')}")
            break

        t_sim += dt

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    print(f"\nPlaced obstacles: {len(obs_map.placed_obstacles)}")

    if running:
        vis.wait_for_close()
    else:
        vis.close()

    return success