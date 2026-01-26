"""
Main simulation loop (dynamic environment).

Notes:
- This loop uses core modules to create a single initial trajectory (planner+smoother).
- After that, the environment ONLY:
  - updates dynamic obstacles
  - provides collision callback
  - draws local interaction radius
  - computes hazard flags (geometric only)
No replanning/re-smoothing happens here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import math

from sparx_agency.core.common.types import Pose2D, Pose3D, Twist3D, State3D
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother, HermiteParams
from sparx_agency.core.planning.smoothers.minsnap import MinSnapSmoother, MinSnapParams
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker, PurePursuitParams

from .config import ScenarioConfig
from .drone_sim import DroneSimulator, DroneSimParams
from .map_dynamic import ObstacleMapDynamic
from .visualization import DroneVisualizer


def build_obstacle_map(cfg: ScenarioConfig) -> ObstacleMapDynamic:
    m = cfg.map
    obs_map = ObstacleMapDynamic(m.width, m.height, m.origin_x, m.origin_y, m.resolution)
    for o in m.obstacles:
        if o.type == "rect":
            obs_map.add_rectangle(o.x, o.y, o.w, o.h)
        else:
            obs_map.add_circle(o.x, o.y, o.r)
    return obs_map


def _distance_point_to_rect(px: float, py: float, rx: float, ry: float, rw: float, rh: float) -> float:
    cx = max(rx, min(px, rx + rw))
    cy = max(ry, min(py, ry + rh))
    return math.hypot(px - cx, py - cy)


def compute_hazards(
    obs_map: ObstacleMapDynamic,
    drone_xy: Tuple[float, float],
    local_radius_m: float,
    trajectory,
    t_now: float,
    horizon_s: float,
    sample_dt_s: float,
    path_proximity_m: float,
) -> Dict[str, Any]:
    """
    Environment-only hazard flags:
    - in_local_radius: any obstacle intersects local radius
    - near_path_ahead: any obstacle is near future trajectory points (geometric)
    """
    dx, dy = drone_xy
    r_local = float(local_radius_m)

    in_local_radius = False

    # Static circles
    for cx, cy, r in obs_map.circles:
        if math.hypot(cx - dx, cy - dy) <= (r + r_local):
            in_local_radius = True
            break

    # Static rects
    if not in_local_radius:
        for rx, ry, rw, rh in obs_map.rectangles:
            if _distance_point_to_rect(dx, dy, rx, ry, rw, rh) <= r_local:
                in_local_radius = True
                break

    # Dynamic circles
    if not in_local_radius:
        for o in obs_map.dynamic_circles:
            if math.hypot(o.cx - dx, o.cy - dy) <= (o.r + r_local):
                in_local_radius = True
                break

    # "Path-ahead" proximity: check future trajectory samples against obstacle boundaries
    near_path_ahead = False
    if trajectory is not None:
        t_end = t_now + float(horizon_s)
        t = t_now
        while t <= t_end:
            p = trajectory.sample(t)
            px, py = float(p.x), float(p.y)

            # static circles
            for cx, cy, r in obs_map.circles:
                if math.hypot(px - cx, py - cy) <= (r + path_proximity_m):
                    near_path_ahead = True
                    break
            if near_path_ahead:
                break

            # static rects
            for rx, ry, rw, rh in obs_map.rectangles:
                if _distance_point_to_rect(px, py, rx, ry, rw, rh) <= path_proximity_m:
                    near_path_ahead = True
                    break
            if near_path_ahead:
                break

            # dynamic circles
            for o in obs_map.dynamic_circles:
                if math.hypot(px - o.cx, py - o.cy) <= (o.r + path_proximity_m):
                    near_path_ahead = True
                    break
            if near_path_ahead:
                break

            t += float(sample_dt_s)

    return {
        "in_local_radius": in_local_radius,
        "near_path_ahead": near_path_ahead,
        "dynamic_count": len(obs_map.dynamic_circles),
    }


def run_simulation(cfg: ScenarioConfig) -> bool:
    print(f"\n{'=' * 60}\nSCENARIO: {cfg.name}\n{'=' * 60}")

    # Map
    obs_map = build_obstacle_map(cfg)
    costmap = obs_map.to_costmap(cfg.map.inflate_radius, include_dynamic=False)
    print(f"Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # Initial plan + smooth once (driver responsibility, not environment)
    start = Pose2D(x=cfg.start[0], y=cfg.start[1])
    goal = Pose2D(x=cfg.goal[0], y=cfg.goal[1])
    print(f"Planning (initial only): ({start.x:.2f}, {start.y:.2f}) -> ({goal.x:.2f}, {goal.y:.2f})")

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

    s = cfg.smoother
    smoother = (
        HermiteSmoother(HermiteParams(dt=s.dt, nominal_speed_xy=s.nominal_speed, tangent_scale=s.tangent_scale))
        if s.type == "hermite"
        else MinSnapSmoother(MinSnapParams(dt=s.dt, nominal_speed_xy=s.nominal_speed))
    )
    trajectory = smoother.smooth(SmootherRequest(path=plan_result.path))
    print(f"Trajectory: {trajectory.total_time:.2f}s")

    # Tracker (core algorithm under test)
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

    # Simulator
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

    # Visualizer
    li = cfg.local_interaction
    vis = DroneVisualizer(
        obstacle_map=obs_map,
        trajectory=trajectory,
        raw_path=plan_result.path,
        drone_radius=sim_params.collision_radius,
        local_radius_m=li.radius_m,
        show_local_radius=li.enabled,
    )

    print("\nRunning... (SPACE=pause, Q=quit)")
    dt = sim_params.dt
    t_sim = 0.0
    success = False

    while t_sim < cfg.max_time:
        # Environment: update dynamic obstacles
        if cfg.dynamic.enabled:
            obs_map.update_dynamic(dt, bounce_on_walls=cfg.dynamic.bounce_on_walls)

        # Core interaction
        x, y, z, vx, vy, vz, yaw = sim.get_measured_state()
        state = State3D(
            pose=Pose3D(x=x, y=y, z=z, yaw=yaw),
            twist=Twist3D(vx=vx, vy=vy, vz=vz, yaw_rate=0.0),
        )

        tr = tracker.step(TrackerRequest(state=state, trajectory=trajectory, t=t_sim))
        sim_state, info = sim.step(tr.command.x, tr.command.y, tr.command.z, tr.command.yaw_rate)

        # Environment-only hazard flags
        hazards = compute_hazards(
            obs_map=obs_map,
            drone_xy=(sim_state.x, sim_state.y),
            local_radius_m=cfg.local_interaction.radius_m if cfg.local_interaction.enabled else 0.0,
            trajectory=trajectory if cfg.local_interaction.enabled else None,
            t_now=t_sim,
            horizon_s=cfg.local_interaction.horizon_s,
            sample_dt_s=cfg.local_interaction.sample_dt_s,
            path_proximity_m=cfg.local_interaction.path_proximity_m,
        )

        lookahead = (tr.reference.x, tr.reference.y, tr.reference.z) if tr.reference else None
        running = vis.update(
            (sim_state.x, sim_state.y, sim_state.z),
            (sim_state.vx, sim_state.vy, sim_state.vz),
            sim_state.yaw,
            t_sim,
            tr.metadata.get("cross_track_error", 0.0),
            tr.metadata.get("progress_idx", 0),
            lookahead_point=lookahead,
            gust_active=info["gust_active"],
            collision=info["collision"],
            done=tr.metadata.get("done", False),
            failed=tr.metadata.get("failed", False),
            wind=info["wind"],
            hazards=hazards,
        )

        if not running:
            break
        if tr.metadata.get("done"):
            success = True
            print("\n✓ GOAL REACHED!")
            break
        if tr.metadata.get("failed"):
            print(f"\n✗ FAILED: {tr.metadata.get('reason', 'unknown')}")
            break

        t_sim += dt

    vis.wait_for_close() if running else vis.close()
    return success
