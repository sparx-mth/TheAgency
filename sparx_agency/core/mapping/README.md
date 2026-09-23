# Mapping Package

The mapping package provides core functionality for environmental data processing and map building in multi-agent
systems.

## Key Features

- **Map Generation**: Creates and maintains consistent 2D/3D representations of the environment
- **Map Merging**: Combines individual agent maps into a unified global map
- **Data Processing**: Handles sensor data integration and filtering
- **Map Updates**: Manages real-time updates and corrections to environmental data

## Components

### Map Representation

- Occupancy grid maps for 2D environments
- Point cloud and mesh representations for 3D environments
- Support for semantic layer annotations

### Data Integration

- Sensor fusion from multiple data sources
- Noise filtering and outlier rejection
- Continuous map refinement

### Geometry Rasterisation (`geometry_raster/`)

Turns 3D triangle geometry into a 2D occupancy slice. The caller brings vertices
and faces already placed in world coordinates, and the package answers which
grid cells the geometry occupies between two heights. Pure numpy — no mesh
library, no scipy, no ROS.

- `rasterise_mesh_slab()` is the entry point. `GridSpec` carries the five
  numbers that make a raster a map: resolution, origin x/y, width, height.
- Triangles are culled, clipped to the height band, and the surviving convex
  polygons are both **filled** (cell centres inside the polygon) and
  **edge-stamped** (4-connected cell chains along the boundary). Filling alone
  leaves holes in thin walls, and a vertical wall sliced by a horizontal slab
  projects to zero area, so its edges are the entire answer.
- Row 0 is **minimum y**, as everywhere else in `core/` — deliberately not the
  top-down row order of a nav2 PGM.

Its first consumer is `tasks/mapping/gazebo_world_occupancy`, which computes the
ground-truth maps committed under `robots/SJTU/maps/`. Conventions and the
reasoning behind the two-pass raster: `geometry_raster/README.md`.

### Multi-Agent Support

- Distributed map building
- Map alignment and registration
- Conflict resolution for overlapping regions

## Usage

The mapping package integrates closely with the localization and planning modules to provide:

- Real-time environment understanding
- Navigation support
- Obstacle detection and avoidance
- Mission planning assistance

## Depth-to-Potential-Field Pipeline

A specialized pipeline for real-time obstacle avoidance using monocular depth and repulsive potential fields.

### Components

- **DepthEngineTRT**: TensorRT-accelerated DepthAnything V3 inference.
- **PotentialMapper**: Orchestrates depth back-projection, EMA-decay occupancy mapping, and potential field computation.
- **PotentialFieldLayer**: Logic for distance transforms and Gaussian repulsive potentials.

### Usage Demo

Run the side-by-side visualization demo:

```bash
# Using HuggingFace fallback (no GPU/Engine required)
python sparx_agency/tasks/mapping/demo_depth_potential_field.py --source 0

# Using TensorRT Engine (Performance mode)
python sparx_agency/tasks/mapping/demo_depth_potential_field.py --source 0 --engine path/to/model.engine
```

For more details, see the [Implementation Plan](file:///home/daphnaa/.gemini/antigravity/brain/fb6b3545-dba2-456c-a5e8-1d7054d6b702/implementation_plan.md).
