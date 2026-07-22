"""
Core planning pipeline.

This module orchestrates:
    Planner -> Path2D -> Smoother -> Trajectory -> Tracker -> ControlCommand

Design goals:
- No ROS dependencies.
- Pure orchestration only (no algorithm code).
- Works with your interfaces in core/planning/interfaces.
- Returns useful artifacts for debugging/visualization (but still ROS-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

from sparx_agency.core.common.types import (
    Pose2D,
    State3D,
    Path2D,
    Trajectory,
    PlanResult,
    PlanStatus,
    KinematicLimits,
    ControlCommand,
)

from sparx_agency.core.planning.interfaces import (
    PlanRequest,
    BasePlanner,
    SmootherRequest,
    BaseSmoother,
    TrackerRequest,
    TrackerResult,
    BaseTracker,
)


# -----------------------------------------------------------------------------
# Config / outputs
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline-level settings (not algorithm params).

    Algorithm-specific params belong in the planner/smoother/tracker implementations,
    typically passed via their constructors or via request.options.
    """
    dt: float = 0.02                 # tracker step (s) for offline stepping helpers
    timeout: float = 120.0           # default tracking timeout (s)
    frame_id: str = "map"


@dataclass
class PipelineArtifacts:
    """
    Intermediate artifacts produced by the pipeline (ROS-free).

    Use this for debugging, evaluation, and later adapters that visualize in ROS.
    """
    plan_request: Optional[PlanRequest] = None
    plan_result: Optional[PlanResult] = None
    path: Optional[Path2D] = None
    trajectory: Optional[Trajectory] = None

    planner_debug: Dict[str, Any] = field(default_factory=dict)
    smoother_debug: Dict[str, Any] = field(default_factory=dict)
    tracker_debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineOutput:
    """
    Output bundle of the pipeline.
    """
    plan_result: PlanResult
    trajectory: Optional[Trajectory]
    artifacts: PipelineArtifacts


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

class PlanningPipeline:
    """
    Orchestrates Planner -> Smoother -> Tracker.

    The pipeline owns component instances (planner/smoother/tracker), but does not
    assume how they were created (factory/registry/config is a separate layer).
    """

    def __init__(
        self,
        planner: BasePlanner,
        smoother: BaseSmoother,
        tracker: BaseTracker,
        *,
        cfg: Optional[PipelineConfig] = None,
    ) -> None:
        self.planner = planner
        self.smoother = smoother
        self.tracker = tracker
        self.cfg = cfg or PipelineConfig()

    # -------------------------------------------------------------------------
    # Plan + smooth
    # -------------------------------------------------------------------------

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        world: Any,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PlanResult, PipelineArtifacts]:
        """
        Run the planner and return (PlanResult, artifacts).
        """
        req = PlanRequest(
            start=start,
            goal=goal,
            frame_id=self.cfg.frame_id,
            options=options or {},
        )

        artifacts = PipelineArtifacts(plan_request=req)

        result = self.planner.plan(req, world)
        artifacts.plan_result = result
        artifacts.path = result.path

        # If implementation put debug data inside artifacts dict, surface it
        if result.artifacts:
            artifacts.planner_debug.update(result.artifacts)

        return result, artifacts

    def smooth(
        self,
        plan_result: PlanResult,
        world: Any,
        *,
        limits: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        artifacts: Optional[PipelineArtifacts] = None,
    ) -> Tuple[Optional[Trajectory], PipelineArtifacts]:
        """
        Smooth a successful PlanResult into a Trajectory.

        Returns:
            (trajectory or None, artifacts)
        """
        if artifacts is None:
            artifacts = PipelineArtifacts(plan_result=plan_result, path=plan_result.path)

        if plan_result.status != PlanStatus.SUCCESS or plan_result.path is None:
            artifacts.trajectory = None
            return None, artifacts

        sreq = SmootherRequest(
            path=plan_result.path,
            limits=limits,
            options=options or {},
        )

        traj = self.smoother.smooth(sreq, world=world)
        artifacts.trajectory = traj

        return traj, artifacts

    def plan_and_smooth(
        self,
        start: Pose2D,
        goal: Pose2D,
        world: Any,
        *,
        plan_options: Optional[Dict[str, Any]] = None,
        smooth_limits: Optional[Any] = None,
        smooth_options: Optional[Dict[str, Any]] = None,
    ) -> PipelineOutput:
        """
        Full pipeline: plan -> smooth.

        Returns:
            PipelineOutput containing PlanResult, optional Trajectory, and artifacts.
        """
        plan_result, artifacts = self.plan(
            start=start,
            goal=goal,
            world=world,
            options=plan_options,
        )

        traj, artifacts = self.smooth(
            plan_result=plan_result,
            world=world,
            limits=smooth_limits,
            options=smooth_options,
            artifacts=artifacts,
        )

        return PipelineOutput(plan_result=plan_result, trajectory=traj, artifacts=artifacts)

    # -------------------------------------------------------------------------
    # Tracking (step-based, ROS-free)
    # -------------------------------------------------------------------------

    def reset_tracker(self) -> None:
        """Reset any internal tracker state (filters, integrators, etc.)."""
        self.tracker.reset()

    def track_step(
        self,
        state: State3D,
        trajectory: Trajectory,
        t: float,
        *,
        limits: Optional[KinematicLimits] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> TrackerResult:
        """
        Compute a single control command for time t along the trajectory.
        """
        treq = TrackerRequest(
            state=state,
            trajectory=trajectory,
            t=t,
            limits=limits,
            options=options or {},
        )
        return self.tracker.step(treq)

    def track_open_loop(
        self,
        initial_state: State3D,
        trajectory: Trajectory,
        *,
        dt: Optional[float] = None,
        duration: Optional[float] = None,
        limits: Optional[KinematicLimits] = None,
        tracker_options: Optional[Dict[str, Any]] = None,
        state_update_fn: Optional[Any] = None,
    ) -> List[TrackerResult]:
        """
        Offline helper: step the tracker repeatedly and collect commands.

        Notes:
        - This does NOT simulate physics unless you provide `state_update_fn`.
        - `state_update_fn(cmd: ControlCommand, state: State3D, dt: float) -> State3D`

        Returns:
            List of TrackerResult (command + reference + metadata).
        """
        self.reset_tracker()

        if dt is None:
            dt = self.cfg.dt
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        if duration is None:
            duration = float(getattr(trajectory, "total_time", 0.0))
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        results: List[TrackerResult] = []
        state = initial_state
        t = 0.0

        n_steps = int(duration / dt) if duration > 0 else 0
        for _ in range(n_steps + 1):
            out = self.track_step(
                state=state,
                trajectory=trajectory,
                t=t,
                limits=limits,
                options=tracker_options,
            )
            results.append(out)

            if state_update_fn is not None:
                # Prefer passing the generic ControlCommand to the state update
                state = state_update_fn(out.command, state, dt)

            t += dt

        return results
