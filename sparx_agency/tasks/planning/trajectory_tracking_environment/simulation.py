"""
Main simulation loop using centralized config.
"""
from sparx_agency.core.common.types import Pose2D, Pose3D, Twist3D, State3D
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother, HermiteParams
from sparx_agency.core.planning.smoothers.minsnap import MinSnapSmoother, MinSnapParams
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker, PurePursuitParams

from drone_sim import DroneSimulator, DroneSimParams
from config import ScenarioConfig
from map_loading import ObstacleMap
from visualization import DroneVisualizer


def build_obstacle_map(cfg: ScenarioConfig) -> ObstacleMap:
    """Build ObstacleMap from config."""
    m = cfg.map
    obs_map = ObstacleMap(m.width, m.height, m.origin_x, m.origin_y, m.resolution)
    for o in m.obstacles:
        if o.type == "rect":
            obs_map.add_rectangle(o.x, o.y, o.w, o.h)
        else:
            obs_map.add_circle(o.x, o.y, o.r)
    return obs_map


def run_simulation(cfg: ScenarioConfig) -> bool:
    """Run simulation from config."""
    print(f"\n{'=' * 60}\nSCENARIO: {cfg.name}\n{'=' * 60}")

    # Build map and costmap
    obs_map = build_obstacle_map(cfg)
    costmap = obs_map.to_costmap(cfg.map.inflate_radius)
    print(f"Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # Plan path
    start, goal = Pose2D(x=cfg.start[0], y=cfg.start[1]), Pose2D(x=cfg.goal[0], y=cfg.goal[1])
    print(f"Planning: ({start.x}, {start.y}) -> ({goal.x}, {goal.y})")

    p = cfg.planner
    planner = RRTStarOmplPlanner(RRTStarOmplParams(
        timeout=p.timeout, use_clearance_objective=p.use_clearance,
        clearance_weight=p.clearance_weight, interpolation_spacing=p.interpolation_spacing))
    result = planner.plan(PlanRequest(start=start, goal=goal, frame_id="map"), costmap)

    if not result.ok:
        print(f"Planning failed: {result.message}")
        return False
    print(f"Path: {len(result.path.points)} waypoints, {result.path.length():.2f}m")

    # Smooth trajectory
    s = cfg.smoother
    smoother = (HermiteSmoother(HermiteParams(dt=s.dt, nominal_speed_xy=s.nominal_speed, tangent_scale=s.tangent_scale))
                if s.type == "hermite" else MinSnapSmoother(MinSnapParams(dt=s.dt, nominal_speed_xy=s.nominal_speed)))
    trajectory = smoother.smooth(SmootherRequest(path=result.path))
    print(f"Trajectory: {trajectory.total_time:.2f}s")

    # Tracker
    t = cfg.tracker
    tracker = PurePursuitTracker(PurePursuitParams(
        holonomic=t.holonomic, base_lookahead=t.base_lookahead, min_lookahead=t.min_lookahead,
        max_lookahead=t.max_lookahead, min_speed=t.min_speed, cruise_speed=t.cruise_speed, max_speed=t.max_speed,
        curvature_speed_factor=t.curvature_speed_factor, curvature_lookahead_factor=t.curvature_lookahead_factor,
        goal_tolerance=t.goal_tolerance, path_tolerance=t.path_tolerance))
    tracker.reset()

    # Simulator
    sc = cfg.simulator
    sim_params = DroneSimParams(
        dt=sc.dt, tau_velocity=sc.tau_velocity, tau_yaw=sc.tau_yaw,
        max_speed_xy=sc.max_speed_xy, max_speed_z=sc.max_speed_z, max_yaw_rate=sc.max_yaw_rate,
        collision_radius=sc.collision_radius, wind_enabled=sc.wind_enabled, wind_mean=sc.wind_mean,
        wind_std=sc.wind_std, wind_tau=sc.wind_tau, gust_enabled=sc.gust_enabled,
        gust_probability=sc.gust_probability, gust_magnitude=sc.gust_magnitude, gust_duration=sc.gust_duration,
        process_noise_std=sc.process_noise_std, position_noise_std=sc.position_noise_std,
        velocity_noise_std=sc.velocity_noise_std, yaw_noise_std=sc.yaw_noise_std)

    collision_fn = lambda x, y: obs_map.is_occupied(x, y, margin=sim_params.collision_radius)
    sim = DroneSimulator(params=sim_params, obstacle_fn=collision_fn, seed=cfg.seed)
    sim.reset(x=start.x, y=start.y, z=0.0)

    # Visualizer
    vis = DroneVisualizer(obstacle_map=obs_map, trajectory=trajectory, raw_path=result.path,
                          drone_radius=sim_params.collision_radius)

    # Simulation loop
    print("\nRunning... (SPACE=pause, Q=quit)")
    dt, t_sim, success = sim_params.dt, 0.0, False

    while t_sim < cfg.max_time:
        x, y, z, vx, vy, vz, yaw = sim.get_measured_state()
        state = State3D(pose=Pose3D(x=x, y=y, z=z, yaw=yaw), twist=Twist3D(vx=vx, vy=vy, vz=vz, yaw_rate=0.0))
        tr = tracker.step(TrackerRequest(state=state, trajectory=trajectory, t=t_sim))
        sim_state, info = sim.step(tr.command.x, tr.command.y, tr.command.z, tr.command.yaw_rate)

        lookahead = (tr.reference.x, tr.reference.y, tr.reference.z) if tr.reference else None
        running = vis.update(
            (sim_state.x, sim_state.y, sim_state.z), (sim_state.vx, sim_state.vy, sim_state.vz),
            sim_state.yaw, t_sim, tr.metadata.get("cross_track_error", 0.0),
            tr.metadata.get("progress_idx", 0), lookahead, info["gust_active"],
            info["collision"], tr.metadata.get("done", False), tr.metadata.get("failed", False),
            info["disturbance"])

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