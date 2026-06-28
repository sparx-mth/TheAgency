from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from sparx_agency.core.common.types import (
    ControlCommand,
    Path2D,
    Pose2D,
    State3D,
    Trajectory,
    TrajectoryPoint,
)

from sparx_agency.core.planning.interfaces.planner import PlanRequest, BasePlanner
from sparx_agency.core.planning.interfaces.smoother import BaseSmoother, SmootherRequest
from sparx_agency.core.planning.interfaces.tracker import BaseTracker, TrackerRequest, TrackerResult

from sparx_agency.core.planning.behaviors.interfaces.context import BehaviorContext
from sparx_agency.core.planning.behaviors.interfaces.output import BehaviorOutput, BehaviorStatus

from sparx_agency.core.planning.behaviors.utils.path_utils import trim_path_prefix


# -----------------------------
# Public result type
# -----------------------------

@dataclass(frozen=True)
class CoordinatorResult:
    """One tick of behavior->plan->(smooth)->track."""
    command: ControlCommand
    behavior: BehaviorOutput
    tracker: Optional[TrackerResult] = None
    plan_info: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Coordinator
# -----------------------------

class BehaviorCoordinator:
    """
    Glue layer between Behavior outputs and the motion pipeline.

    Contract:
      - Behavior returns BehaviorOutput (subgoal/path/control + status).
      - Coordinator decides which pipeline to run:
          * control -> passthrough
          * path -> (smooth?) -> track
          * subgoal -> plan -> (smooth?) -> track

    Notes:
      - This module does NOT own semantics extraction (portals/rooms). Tasks provide those in ctx.
      - World type is opaque here. Planner/smoother may assume Costmap2D / OccupancyGrid2D / Voxelmap etc.
    """

    def __init__(
        self,
        *,
        planner: Optional[BasePlanner] = None,
        smoother: Optional[BaseSmoother] = None,
        tracker: Optional[BaseTracker] = None,
        # Defaults used only if smoother is None and you still want tracking:
        default_speed_mps: float = 0.35,
        # When close enough to the last point, consider trajectory done
        goal_radius_m: float = 0.10,
    ) -> None:
        self._planner = planner
        self._smoother = smoother
        self._tracker = tracker

        self._default_speed_mps = float(default_speed_mps)
        self._goal_radius_m = float(goal_radius_m)

        # Per-robot caches
        self._active_path: Dict[int, Path2D] = {}
        self._active_traj: Dict[int, Trajectory] = {}
        self._traj_t0: Dict[int, float] = {}
        self._last_subgoal: Dict[int, Pose2D] = {}

    def reset(self, robot_id: Optional[int] = None) -> None:
        """Reset cached execution state (optionally for a single robot)."""
        if robot_id is None:
            self._active_path.clear()
            self._active_traj.clear()
            self._traj_t0.clear()
            self._last_subgoal.clear()
            if self._tracker is not None:
                self._tracker.reset()
            return

        self._active_path.pop(robot_id, None)
        self._active_traj.pop(robot_id, None)
        self._traj_t0.pop(robot_id, None)
        self._last_subgoal.pop(robot_id, None)

    def step(
        self,
        *,
        behavior: Any,
        # Inputs for BehaviorContext
        robot_id: int,
        pose: Pose2D,
        world: Any,
        map_frame_id: str = "map",
        portals=(),
        forbidden_portal_ids=(),
        options: Optional[Dict[str, Any]] = None,
        # Behavior may need a goal (e.g., GoToPoseBehavior)
        goal: Optional[Pose2D] = None,
        # Tracker input
        state: Optional[State3D] = None,
        now_s: float,
        # Tracker/smoother options
        tracker_options: Optional[Dict[str, Any]] = None,
        smoother_options: Optional[Dict[str, Any]] = None,
        planner_options: Optional[Dict[str, Any]] = None,
    ) -> CoordinatorResult:
        """
        Run one coordinator tick.

        Args:
            behavior: Behavior implementation (Protocol).
            robot_id, pose, world: core inputs.
            state: required if you want tracking->ControlCommand.
            now_s: wall-clock / sim time seconds.
        """
        ctx = BehaviorContext(
            robot_id=int(robot_id),
            pose=pose,
            map_frame_id=str(map_frame_id),
            portals=tuple(portals),
            forbidden_portal_ids=set(forbidden_portal_ids),
            options=dict(options or {}),
        )

        # Many of your algorithmic behaviors currently expect extra fields
        # (ctx.goal / ctx.world / ctx.features). We DON'T want to change their files,
        # so we attach them dynamically in a minimal, non-invasive way.
        # This keeps interfaces clean while allowing legacy behavior code to run.
        setattr(ctx, "world", world)
        setattr(ctx, "goal", goal)
        # "features" is used in the snippets you pasted (ctx.features.get(...))
        if not hasattr(ctx, "features"):
            setattr(ctx, "features", {})
        # Provide portals in features too (common pattern in your behaviors)
        ctx.features.setdefault("portals", list(portals))

        # -------------------------
        # 1) Behavior step
        # -------------------------
        out = behavior.step(ctx, world) if _behavior_step_accepts_world(behavior) else behavior.step(ctx)

        # Fallback: some old behaviors may return BehaviorDecision-like object
        out = _coerce_to_behavior_output(out)

        # If behavior finished or failed, stop motion.
        if out.status in (BehaviorStatus.SUCCESS, BehaviorStatus.FAILURE):
            self._active_path.pop(robot_id, None)
            self._active_traj.pop(robot_id, None)
            self._traj_t0.pop(robot_id, None)
            self._last_subgoal.pop(robot_id, None)
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=out,
                tracker=None,
                plan_info={"terminal": out.status.value},
            )

        # -------------------------
        # 2) control passthrough
        # -------------------------
        if out.control is not None:
            # Behavior decided to output control directly (rare).
            return CoordinatorResult(command=out.control, behavior=out, tracker=None, plan_info={"mode": "control"})

        # From here: need tracker to produce a ControlCommand.
        if self._tracker is None:
            # If you're in an integration phase where task executes outside,
            # return zero command (safe) and keep out.path/out.subgoal in BehaviorOutput.info.
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=out,
                tracker=None,
                plan_info={"error": "no_tracker_configured", "mode": "decision_only"},
            )

        if state is None:
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=out,
                tracker=None,
                plan_info={"error": "state_required_for_tracking"},
            )

        tracker_options = dict(tracker_options or {})
        smoother_options = dict(smoother_options or {})
        planner_options = dict(planner_options or {})

        # -------------------------
        # 3) path present => track it
        # -------------------------
        if out.path is not None:
            path = trim_path_prefix(out.path, pose)
            traj, t0 = self._ensure_trajectory(robot_id, path, world, now_s, smoother_options)
            tr = self._tracker.step(TrackerRequest(state=state, trajectory=traj, t=now_s - t0, options=tracker_options))
            return CoordinatorResult(command=tr.command, behavior=out, tracker=tr, plan_info={"mode": "path"})

        # -------------------------
        # 4) subgoal present => plan => track
        # -------------------------
        if out.subgoal is None:
            # Running but with no actionable output: stop safely.
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=out,
                tracker=None,
                plan_info={"mode": "idle_running_no_goal"},
            )

        # Replan only if subgoal changed meaningfully or no cached plan exists.
        subgoal = out.subgoal
        last = self._last_subgoal.get(robot_id)
        if (last is None) or (pose.distance_to(last) > 1e-9 and _pose_changed(last, subgoal)):
            self._last_subgoal[robot_id] = subgoal
            # invalidate cached traj/path for this robot
            self._active_path.pop(robot_id, None)
            self._active_traj.pop(robot_id, None)
            self._traj_t0.pop(robot_id, None)

        # Plan (requires planner)
        if self._planner is None:
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=BehaviorOutput(
                    status=BehaviorStatus.FAILURE,
                    info={"error": "subgoal_provided_but_no_planner_configured"},
                ),
                tracker=None,
                plan_info={"mode": "subgoal", "error": "no_planner"},
            )

        req = PlanRequest(start=pose, goal=subgoal, frame_id=map_frame_id, options=planner_options)
        plan_res = self._planner.plan(req, world)

        if not getattr(plan_res, "ok", False) or getattr(plan_res, "path", None) is None:
            return CoordinatorResult(
                command=ControlCommand.zero(),
                behavior=BehaviorOutput(
                    status=BehaviorStatus.FAILURE,
                    info={
                        "error": "planning_failed",
                        "planner": getattr(self._planner, "name", type(self._planner).__name__),
                        "plan_status": getattr(plan_res, "status", None),
                        "message": getattr(plan_res, "message", ""),
                        "artifacts": getattr(plan_res, "artifacts", None),
                    },
                ),
                tracker=None,
                plan_info={"mode": "subgoal", "planner_failed": True},
            )

        path: Path2D = plan_res.path
        path = trim_path_prefix(path, pose)

        # Cache
        self._active_path[robot_id] = path

        traj, t0 = self._ensure_trajectory(robot_id, path, world, now_s, smoother_options)
        tr = self._tracker.step(TrackerRequest(state=state, trajectory=traj, t=now_s - t0, options=tracker_options))

        return CoordinatorResult(
            command=tr.command,
            behavior=out,
            tracker=tr,
            plan_info={
                "mode": "subgoal->plan->track",
                "planner": getattr(self._planner, "name", type(self._planner).__name__),
                "path_len": path.length() if hasattr(path, "length") else None,
            },
        )

    # -------------------------
    # Internals
    # -------------------------

    def _ensure_trajectory(
        self,
        robot_id: int,
        path: Path2D,
        world: Any,
        now_s: float,
        smoother_options: Dict[str, Any],
    ) -> Tuple[Trajectory, float]:
        # Reuse cached trajectory if path object is the same instance (or same points)
        traj = self._active_traj.get(robot_id)
        if traj is not None and _path_same(traj, path):
            t0 = self._traj_t0.get(robot_id, now_s)
            return traj, t0

        # Build new trajectory
        if self._smoother is not None:
            traj = self._smoother.smooth(SmootherRequest(path=path, options=smoother_options), world=world)
        else:
            traj = _path_to_constant_speed_trajectory(path, speed_mps=self._default_speed_mps)

        self._active_traj[robot_id] = traj
        self._traj_t0[robot_id] = now_s
        return traj, now_s


# -----------------------------
# Helpers (small + local)
# -----------------------------

def _pose_changed(a: Pose2D, b: Pose2D, *, tol_m: float = 1e-6, tol_yaw: float = 1e-6) -> bool:
    return (abs(a.x - b.x) > tol_m) or (abs(a.y - b.y) > tol_m) or (abs((a.yaw or 0.0) - (b.yaw or 0.0)) > tol_yaw)


def _behavior_step_accepts_world(behavior: Any) -> bool:
    # duck-typing: if method signature expects (ctx, world) some behaviors do that
    try:
        import inspect
        sig = inspect.signature(behavior.step)
        return len(sig.parameters) >= 2
    except Exception:
        return True


def _coerce_to_behavior_output(obj: Any) -> BehaviorOutput:
    # Already correct
    if isinstance(obj, BehaviorOutput):
        return obj

    # Legacy "BehaviorDecision" style
    goal = getattr(obj, "goal", None)
    path = getattr(obj, "path", None)
    done = getattr(obj, "done", False)
    info = getattr(obj, "info", {}) or {}

    if done:
        return BehaviorOutput(status=BehaviorStatus.SUCCESS, info=info)

    # Map legacy fields to new ones
    return BehaviorOutput(
        status=BehaviorStatus.RUNNING,
        subgoal=goal,
        path=path,
        control=None,
        info=info,
    )


def _path_to_constant_speed_trajectory(path: Path2D, *, speed_mps: float) -> Trajectory:
    """
    Minimal fallback when no smoother exists.

    Creates a Trajectory where time is proportional to cumulative arc length.
    Assumes Path2D points are in world frame and close enough for piecewise-linear tracking.
    """
    pts = list(path.points)
    if len(pts) < 2:
        # degenerate: stay
        p = pts[0] if pts else Pose2D(0.0, 0.0, 0.0)
        tp = TrajectoryPoint(x=p.x, y=p.y, yaw=p.yaw or 0.0, t=0.0)
        return Trajectory(points=(tp, tp), frame_id=path.frame_id, metadata={"source": "fallback"})

    v = max(1e-3, float(speed_mps))
    out: list[TrajectoryPoint] = []

    t = 0.0
    prev = pts[0]
    out.append(TrajectoryPoint(x=prev.x, y=prev.y, yaw=prev.yaw or 0.0, t=t))

    for p in pts[1:]:
        dx = float(p.x - prev.x)
        dy = float(p.y - prev.y)
        ds = (dx * dx + dy * dy) ** 0.5
        t += ds / v
        out.append(TrajectoryPoint(x=p.x, y=p.y, yaw=p.yaw or 0.0, t=t))
        prev = p

    return Trajectory(points=tuple(out), frame_id=path.frame_id, metadata={"source": "fallback_constant_speed"})


def _path_same(traj: Trajectory, path: Path2D) -> bool:
    # cheap check: same frame and same end points => assume same for caching
    try:
        if traj.frame_id != path.frame_id:
            return False
        tp0 = traj.points[0]
        tpl = traj.points[-1]
        p0 = path.points[0]
        pl = path.points[-1]
        return (abs(tp0.x - p0.x) < 1e-9 and abs(tp0.y - p0.y) < 1e-9 and
                abs(tpl.x - pl.x) < 1e-9 and abs(tpl.y - pl.y) < 1e-9)
    except Exception:
        return False
