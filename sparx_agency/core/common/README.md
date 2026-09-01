# Common Package

The `common` package provides shared utilities, base classes, and common functionality used throughout the Agency
project. This package serves as a foundation for standardizing core operations and reducing code duplication across
different modules.

## Key Components

- **Base Classes**: Abstract implementations and interfaces that define common behavior
- **Utilities**: Helper functions and tools used across different modules
- **Data Structures**: Shared data structures and models
- **Constants**: Project-wide constants and configuration values

## Main Functionality

- Standardized logging and error handling
- Common mathematical operations
- Shared configuration management
- Basic data type conversions
- Core interfaces for system components

## What belongs here, and what only looks like it does

This is the **widest** ring of shared code: something lands here when it is
unambiguous and universal — arithmetic and vocabulary that mapping, localization,
planning and control all mean identically. A thing shared by only one subsystem
belongs in *that* subsystem's own `common` (e.g. `core/planning/vlas/common/`
holds the policy-server HTTP client, which nothing outside the VLAs speaks).
Widening a domain-specific contract to here is not generosity; it makes every
consumer depend on a concept it does not use.

### The geometry modules, and which to reach for

- **`math/se2.py`** — the planar body↔world pair (`body_to_world_2d`,
  `world_to_body_2d`, `body_to_world_xy`, `rotate_2d`). Reach for this instead of
  spelling out `cos`/`sin`: it existed as three separate copies before it lived
  here.
- **`math/se3.py`** — quaternions and 4×4 transforms.
  `yaw_from_quaternion(q)` is **the** implementation of quaternion→yaw.
- **`spatial_math.py`** — the older, broader module (Euler, intrinsics, YAML,
  pose conversions). Its `quat_to_yaw(qx, qy, qz, qw)` is a scalar-argument
  spelling that delegates to `se3.yaw_from_quaternion`; both are kept because 25
  files import one or the other, but there is only one copy of the arithmetic.
  `robots/common/spatial_math.py` is a re-export shim for older import paths.

### The vocabulary module

- **`label_match.py`** — `label_matches(target, label)` is **the** offline rule
  for deciding whether a detector class counts as the requested target (exact,
  substring, or shared whitespace/underscore token). It is here rather than in
  either caller because two subsystems mean it identically: the visual-servo
  acquisition gate (`core/planning/visual_servo/confirmation_gate.py`, which
  re-exports it under its published name) and the scene-graph match ladder's
  offline rung (`core/mapping/topology/target_matcher.py::fallback_match`).

## Usage Examples

The common package can be imported and used in other modules:
