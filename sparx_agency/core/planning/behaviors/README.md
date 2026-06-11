# Behaviors Module

**Path:** `sparx_agency/core/planning/behaviors`

A modular behavior framework for autonomous robot navigation and exploration.

## Overview

This module provides a composable behavior system where each behavior encapsulates a specific navigation strategy. Behaviors produce outputs (paths, subgoals, or control commands) that a coordinator can execute.

## Architecture

```
behaviors/
├── interfaces/          # Core abstractions
│   ├── behavior.py      # Behavior protocol and BehaviorDecision
│   ├── context.py       # BehaviorContext input container
│   ├── features.py      # Semantic features (Portal2D)
│   └── output.py        # BehaviorOutput and BehaviorStatus
├── algorithmic/         # Concrete behavior implementations
│   ├── go_to_pose.py    # Goal-directed navigation
│   ├── explore_room.py  # Frontier-based room exploration
│   ├── enter_portal.py  # Doorway/threshold traversal
│   └── wall_follow.py   # Wall-following navigation
├── utils/               # Shared utilities
│   ├── path_utils.py    # Path manipulation helpers
│   └── world_adapters.py# World representation converters
└── registry.py          # Behavior lookup by name
```

## Behaviors

| Behavior | Purpose | Requires |
|----------|---------|----------|
| `GoToPoseBehavior` | Navigate to a specific pose | `ctx.goal` |
| `ExploreRoomBehavior` | Frontier exploration within room bounds | `OccupancyGrid2D` |
| `EnterPortalBehavior` | Cross doorways/thresholds | `ctx.features["portals"]` |
| `WallFollowBehavior` | Follow walls at set clearance | `Costmap2D` or `OccupancyGrid2D` |

## Usage

```python
from sparx_agency.core.planning.behaviors import GoToPoseBehavior, BehaviorContext

behavior = GoToPoseBehavior()
ctx = BehaviorContext(robot_id=1, pose=current_pose, goal=target_pose, world=grid)

output = behavior.step(ctx, planner=my_planner)
if output.ok:
    execute(output.path or output.subgoal)
```

## Output Contract

Each `step()` returns a `BehaviorOutput` containing:
- `status`: `RUNNING`, `SUCCESS`, or `FAILURE`
- `path`: Optional planned path (if planner provided)
- `subgoal`: Optional intermediate goal pose
- `control`: Optional direct control command (rare)
- `info`: Diagnostic metadata

## Design Principles

1. **Stateless inputs**: All state passed via `BehaviorContext`
2. **Planner-agnostic**: Behaviors work with or without a planner
3. **Composable**: Coordinator chains behaviors as needed
4. **Minimal coupling**: Behaviors don't depend on each other