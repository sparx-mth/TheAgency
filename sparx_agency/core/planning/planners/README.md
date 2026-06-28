# Path Planning Algorithms

This module provides three path planning algorithms using OMPL (Open Motion Planning Library).

## Architecture

```
planners/
├── __init__.py                 # Unified exports from all planners
├── README.md
├── common/                     # Shared utilities (no duplication)
│   ├── __init__.py
│   ├── ompl_imports.py         # OMPL availability check
│   ├── utils_2d.py             # 2D path utilities
│   └── utils_3d.py             # 3D path utilities + OMPL space setup
├── rrtstar/                    # RRT* (2D + 3D)
│   ├── __init__.py
│   ├── algorithm.py            # Planning logic with debug support
│   ├── params.py               # RRTStarOmplParams, RRTStarOmpl3DParams
│   └── planner.py              # Planner classes
├── bitstar/                    # BIT* (3D only)
│   ├── __init__.py
│   ├── algorithm.py
│   ├── params.py               # BITStarParams
│   └── planner.py
└── informed_rrtstar/           # Informed RRT* (3D only)
    ├── __init__.py
    ├── algorithm.py
    ├── params.py               # InformedRRTStarParams
    └── planner.py

# Interface types come from:
# sparx_agency/core/planning/interfaces/
#   └── PlanRequest, PlanRequest3D, BasePlanner, BasePlanner3D
```

## Quick Start

### RRT* (2D)
```python
from planners import RRTStarOmplPlanner, PlanRequest
from sparx_agency.core.common.types import Pose2D

planner = RRTStarOmplPlanner()
request = PlanRequest(
    start=Pose2D(0, 0),
    goal=Pose2D(10, 10)
)
result = planner.plan(request, costmap)
```

### RRT* (3D)
```python
from planners import RRTStarOmpl3DPlanner, PlanRequest3D
from sparx_agency.core.common.types import Pose3D

planner = RRTStarOmpl3DPlanner()
request = PlanRequest3D(
    start=Pose3D(0, 0, 1),
    goal=Pose3D(10, 10, 2)
)
result = planner.plan(request, voxelmap)
```

### BIT* (3D) - Recommended for complex environments
```python
from planners import BITStarPlanner
from planners.bitstar import PlanRequest3D
from sparx_agency.core.common.types import Pose3D

planner = BITStarPlanner()
request = PlanRequest3D(
    start=Pose3D(0, 0, 1),
    goal=Pose3D(10, 10, 2)
)
result = planner.plan(request, voxelmap)
```

### Informed RRT* (3D)
```python
from planners import InformedRRTStarPlanner
from planners.informed_rrtstar import PlanRequest3D
from sparx_agency.core.common.types import Pose3D

planner = InformedRRTStarPlanner()
request = PlanRequest3D(
    start=Pose3D(0, 0, 1),
    goal=Pose3D(10, 10, 2)
)
result = planner.plan(request, voxelmap)
```

## Algorithm Comparison

| Algorithm | Dimension | Best For |
|-----------|-----------|----------|
| **RRT*** | 2D, 3D | General use, extensive debugging |
| **BIT*** | 3D | Complex environments, fast convergence |
| **Informed RRT*** | 3D | Clear paths, simpler than BIT* |

### When to Use Each

- **RRT* 2D**: Standard 2D navigation with costmaps
- **RRT* 3D**: 3D planning with extensive debug output for troubleshooting
- **BIT* 3D**: Best for complex 3D environments; uses batch processing for faster convergence
- **Informed RRT* 3D**: Good middle ground; samples in ellipsoidal region after finding initial solution

## Shared Utilities

The `common/` module contains utilities shared across all planners:

- **OMPL imports**: Centralized availability check
- **2D utilities**: `interpolate_path_2d`, `reduce_path_2d`, `make_clearance_objective_2d`
- **3D utilities**: `interpolate_path_3d`, `reduce_path_3d`, `make_clearance_objective_3d`, `setup_ompl_space_3d`

Interface types (`PlanRequest`, `PlanRequest3D`, `BasePlanner`, `BasePlanner3D`) are imported from `sparx_agency.core.planning.interfaces`.

## Parameters

All planners share common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | 3.0-5.0s | Maximum planning time |
| `use_clearance_objective` | True | Optimize for obstacle clearance |
| `clearance_weight` | 10.0 | Weight for clearance cost |
| `min_clearance_for_keep` | 0.3m | Min clearance for waypoint removal |
| `interpolation_spacing` | 0.2m | Target spacing between waypoints |
| `collision_check_resolution` | 0.005 | OMPL validity check resolution |

### Algorithm-Specific Parameters

**BIT***:
- `samples_per_batch`: Number of samples per batch (default: 100)
- `use_k_nearest`: Use k-nearest instead of r-disc (default: True)
- `rewire_factor`: Rewiring factor (default: 1.1)

**Informed RRT***:
- `range_m`: Max edge length (default: auto)

**RRT* 3D**:
- `rrt_range_m`: RRT step/extension length
- `debug_enabled`: Enable detailed debug output
