# SJTU maps — ground-truth occupancy

Surveyed reference maps of the Gazebo worlds the SJTU drone flies in. They are
computed from the worlds' own SDF and collision meshes, not recorded from a
flight, so they are right by construction and never change unless the world
does.

## `hospital`

The whole aws-robomaker hospital floor, in the Gazebo **world frame** — the
same frame `/simple_drone/odom` reports, since the sjtu_drone plugin publishes
`model->WorldPose()` straight onto that topic. A pose read off odom indexes
into this map with no transform.

| | |
|---|---|
| world | `aws-robomaker-hospital-world/worlds/hospital.world` (SDF 1.6, 182 includes) |
| source repo | `sjtu_project` @ `942996a` (2026-08-12) |
| world sha256 | `4dec25f2bd02745a084482e9843b4a59196854531f17949e0e574d8e2db3f284` |
| height band | **0.30 m to 2.00 m** — what a drone cruising at 1.2 m can hit |
| resolution | 0.05 m |
| grid | 544 x 1182 cells |
| origin | `(-13.60, -36.10)` m, lower-left corner |
| geometry | 174 model instances in the band, 167 733 triangles |
| occupied | 77 083 cells, 11.99 % of the map |
| occupied extent | x `[-12.60, 12.55]` m, y `[-35.10, 22.00]` m |

The drone itself is excluded, and so are Gazebo's `sun` and `ground_plane`
(a light and an infinite plane; neither is mappable).

### Files

| file | what it is |
|---|---|
| `hospital.pgm` + `hospital.yaml` | nav2 `map_server` format — 0 occupied, 254 free, 205 unknown. **PGM row 0 is maximum y.** |
| `hospital.npz` | An `OccupancyGrid2D` via `core.planning.environment.occupancy_io`, plus provenance metadata (world path and hash, search paths, band, triangle count). **Grid row 0 is minimum y**, as everywhere else in `core/`. |

Load the `.npz` with
`sparx_agency.core.planning.environment.occupancy_io.load_occupancy_grid`.

`hospital.yaml` sets `free_thresh: 0.196`, not the customary `0.25`. nav2
decodes a pixel as `occ = 1 - pixel/255` and calls anything *below*
`free_thresh` free — and the unknown grey, 205, is `occ = 0.196`. At 0.25 an
unknown pixel therefore reads back as **free**, which is a map server inventing
open space wherever nothing was surveyed. This map has no unknown cells for it
to get wrong, but the writer is shared with maps that do.

### Every cell is known

Occupied means geometry passes through the cell somewhere inside the band.
Everything else is free — including outside the building, and including the
inside of a closed cupboard. **There are no unknown cells**, because nothing
here was observed; it was computed. Do not hand this to an exploration
algorithm that reads "free" as "already seen".

The world's models are surfaces, so a large closed object slices to a
watertight hollow ring rather than a solid blob. Its interior is enclosed and
unreachable, but it does read as free — so do not sample a goal uniformly from
free space and assume every draw is reachable.

Both traps have one answer, and it is written down:
`core.planning.environment.grid_regions.largest_enclosed_region` returns the
largest free component that does not touch the edge of the grid — the hospital's
own floor, **1139.9 m² of the map's 1414.8 m² of free space**. The 197 m² outside
the wall runs off the border and is dropped; the seventy sealed voids inside
cupboards, wardrobes and the elevator cars are dropped with it.
`core.planning.exploration.visibility_coverage` divides by exactly that, and
maintains its own seen-mask on top of this grid rather than mistaking `free` for
`observed`.

### Cross-check

Against the independently produced SLAM map at
`tasks/planning/rrt_smoothing_check/maps/hospital_map_cropped.pgm`
(0.04 m, origin `(-12.85, -35.00)`), whose occupied extent is
x `[-12.49, 12.43]` y `[-34.96, 20.96]`:

| edge | delta |
|---|---|
| x min | −0.11 m |
| x max | +0.12 m |
| y min | −0.14 m |
| y max | +1.04 m |

The +y edge is the only real difference and it is expected: this map includes
the two **elevator car interiors**, which sit behind the elevator doors from
y ≈ 19.2 to 22.0 m. A map recorded from inside the corridor cannot see them.
Ignore everything above the wall line at y = 20.95 m and the y max delta falls
to −0.06 m, i.e. all four edges agree within 0.14 m.

### Regenerate

```bash
cd /path/to/TheAgency
.venv/bin/python -m sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map \
    --world  /path/to/sjtu_project/aws-robomaker-hospital-world/worlds/hospital.world \
    --search-path /path/to/sjtu_project/aws-robomaker-hospital-world/models \
    --search-path /path/to/sjtu_project/aws-robomaker-hospital-world/fuel_models \
    --search-path /path/to/sjtu_project/sjtu_drone/sjtu_drone_description \
    --search-path /path/to/sjtu_project/sjtu_drone/models \
    --output-dir sparx_agency/robots/SJTU/maps \
    --resolution 0.05 --z-min 0.30 --z-max 2.00
```

About 16 seconds. Add `--preview /tmp/hospital.png` to eyeball the result. The
tool, its options and the COLLADA unit trap that will bite anyone who rewrites
the mesh loading are documented in
`tasks/mapping/gazebo_world_occupancy/README.md`.
