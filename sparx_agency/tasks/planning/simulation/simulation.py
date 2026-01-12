"""
Main simulation loop for drone tracking.

Integrates planning, smoothing, tracking, and physics simulation.
"""
from __future__ import annotations

from typing import Optional

# Types
from sparx_agency.core.common.types import (
    Pose2D, Pose3D, Twist3D, State3D
)

# Planning interfaces
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest

# Planner (from core)
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams

# Smoothers (from core)
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother
from sparx_agency.core.planning.smoothers.minsnap import MinSnapSmoother

# Tracker (from core)
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker, PurePursuitParams

# Environment / Costmap (from core)
from sparx_agency.core.planning.environment import Costmap2D

# Simulation (physics only - not an algorithm)
from sparx_agency.tasks.planning.simulation.drone_sim import DroneSimulator, DroneSimParams

# Local imports
from map_loading import ObstacleMap
from visualization import DroneVisualizer


def run_simulation(
        obstacle_map: Optional[ObstacleMap],
        start: Pose2D,
        goal: Pose2D,
        planner_params: RRTStarOmplParams,
        smoother_type: str,  # "hermite" or "minsnap"
        smoother_params,
        tracker_params: PurePursuitParams,
        sim_params: DroneSimParams,
        max_time: float = 100.0,
        planning_margin: float = 0.15,
        seed: Optional[int] = None,
        costmap: Optional[Costmap2D] = None,
) -> bool:
    """
    Run full planning + tracking simulation.

    All algorithms from sparx_agency.core:
    1. RRTStarOmplPlanner - path planning
    2. HermiteSmoother/MinSnapSmoother - trajectory smoothing
    3. PurePursuitTracker - trajectory tracking

    Args:
        planning_margin: Safety margin for RRT path planning (inflates obstacles).
                        Collision detection uses margin=0 (actual contact only).
    """

    # ========== STEP 1: Create/Use Costmap ==========
    if costmap is not None:
        # Scenario 4: costmap provided directly (from PGM file)
        print("Using provided costmap...")
    elif obstacle_map is not None:
        print("Creating costmap from obstacles...")
        # planning_margin inflates obstacles for RRT planning (safety buffer)
        costmap = obstacle_map.to_costmap(inflate_radius=planning_margin)
    else:
        print("ERROR: No obstacle_map or costmap provided")
        return False

    print(f"  Costmap: {costmap.width}x{costmap.height}, res={costmap.resolution}m")
    print(f"  Planning margin: {planning_margin}m (RRT keeps paths this far from obstacles)")
    print(f"  Drone radius: {sim_params.collision_radius}m (collision when edge touches obstacle)")

    # ========== STEP 2: Plan path with RRT* (from CORE) ==========
    print(f"Planning path: ({start.x}, {start.y}) → ({goal.x}, {goal.y})...")
    planner = RRTStarOmplPlanner(params=planner_params)
    plan_request = PlanRequest(start=start, goal=goal, frame_id="map")
    plan_result = planner.plan(plan_request, costmap)

    print(f"  Status: {plan_result.status}")
    if not plan_result.ok:
        print(f"  Planning failed: {plan_result.message}")
        return False

    raw_path = plan_result.path
    print(f"  Path: {len(raw_path.points)} waypoints, length={raw_path.length():.2f}m")

    # ========== STEP 3: Smooth trajectory (from CORE) ==========
    print(f"Smoothing with {smoother_type}...")
    smooth_request = SmootherRequest(path=raw_path)

    if smoother_type == "hermite":
        smoother = HermiteSmoother(params=smoother_params)
    else:
        smoother = MinSnapSmoother(params=smoother_params)

    trajectory = smoother.smooth(smooth_request)
    print(f"  Trajectory duration: {trajectory.total_time:.2f}s")

    # ========== STEP 4: Create tracker (from CORE) ==========
    tracker = PurePursuitTracker(params=tracker_params)
    tracker.reset()

    # ========== STEP 5: Create drone simulator (physics only) ==========
    # Use drone's actual collision radius for accurate collision detection
    drone_radius = sim_params.collision_radius

    if obstacle_map is not None:
        # Collision when any part of drone touches obstacle (not just center)
        collision_fn = lambda x, y: obstacle_map.is_occupied(x, y, margin=drone_radius)
    else:
        # Use costmap occupancy check for scenario 4
        # Costmap is already inflated by robot radius, so check center only
        def collision_fn(x: float, y: float) -> bool:
            ix = int((x - costmap.origin_x) / costmap.resolution)
            iy = int((y - costmap.origin_y) / costmap.resolution)
            if not (0 <= ix < costmap.width and 0 <= iy < costmap.height):
                return True  # Out of bounds = collision
            if costmap.occupancy[iy, ix] > 200:  # Occupied threshold
                return True
            return False

    sim = DroneSimulator(params=sim_params, obstacle_fn=collision_fn, seed=seed)
    sim.reset(x=start.x, y=start.y, z=0.0)

    # ========== STEP 6: Create visualizer ==========
    vis = DroneVisualizer(
        obstacle_map=obstacle_map,
        trajectory=trajectory,
        raw_path=raw_path,
        costmap=costmap if obstacle_map is None else None,  # Show costmap for scenario 4
        drone_radius=drone_radius,  # Pass actual collision radius for accurate visualization
    )

    # ========== STEP 7: Run simulation loop ==========
    dt = sim.params.dt
    t = 0.0

    print("\n" + "=" * 50)
    print("SIMULATION RUNNING")
    print("  Planner: RRTStarOmplPlanner (core)")
    print(f"  Smoother: {smoother_type.capitalize()}Smoother (core)")
    print("  Tracker: PurePursuitTracker (core)")
    print("Controls: SPACE=pause, Q=quit")
    print("=" * 50 + "\n")

    running = True
    result_success = False

    while running and t < max_time:
        # Get measured state
        x, y, z, vx, vy, vz, yaw = sim.get_measured_state()

        # Build state (core types)
        state = State3D(
            pose=Pose3D(x=x, y=y, z=z, yaw=yaw),
            twist=Twist3D(vx=vx, vy=vy, vz=vz, yaw_rate=0.0),
        )

        # Run tracker (core TrackerRequest)
        request = TrackerRequest(state=state, trajectory=trajectory, t=t)
        tracker_result = tracker.step(request)

        # Extract command and step simulator
        cmd = tracker_result.command
        sim_state, info = sim.step(cmd.x, cmd.y, cmd.z, cmd.yaw_rate)

        # Get visualization data
        lookahead_pt = None
        if tracker_result.reference:
            ref = tracker_result.reference
            lookahead_pt = (ref.x, ref.y, ref.z)

        cte = tracker_result.metadata.get("cross_track_error", 0.0)
        done = tracker_result.metadata.get("done", False)
        failed = tracker_result.metadata.get("failed", False)

        # Update visualization
        running = vis.update(
            drone_position=(sim_state.x, sim_state.y, sim_state.z),
            drone_velocity=(sim_state.vx, sim_state.vy, sim_state.vz),
            drone_yaw=sim_state.yaw,
            time=t,
            cross_track_error=cte,
            progress_idx=tracker_result.metadata.get("progress_idx", 0),
            lookahead_point=lookahead_pt,
            gust_active=info["gust_active"],
            collision=info["collision"],
            done=done,
            failed=failed,
            wind=info["disturbance"],  # Total wind disturbance (wind + gust)
        )

        if done:
            result_success = True
            print("\n✓ GOAL REACHED!")
            break

        if failed:
            print(f"\n✗ TRACKING FAILED: {tracker_result.metadata.get('reason', 'unknown')}")
            break

        t += dt

    # Wait for user to close
    if running:
        print("\nSimulation complete. Press Q or close window to exit.")
        vis.wait_for_close()
    else:
        vis.close()

    return result_success