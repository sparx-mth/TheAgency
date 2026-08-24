# `gazebo_world_occupancy` — ground-truth maps from a Gazebo world

Compute a 2D occupancy map of a whole simulated building **from the world's own
SDF and collision meshes**, sliced over the height band the robot flies through.
Three artifacts come out: a nav2 `map_server` PGM + YAML for ROS tooling, and an
`.npz` carrying an `OccupancyGrid2D` for this repo's planners.

## Why not just fly the robot and record what it maps?

Because then you are measuring the mapper, not the building. A flown map
inherits every depth error, every pose drift, every corridor the robot never
went down. It is exactly the thing you want a *reference* for.

This map is computed from the geometry the simulator itself collides against,
so it is right by construction. That makes it the yardstick for three jobs:

* **scoring a live map** — how much of the real building did the run actually
  discover, and did it hallucinate anything?
* **planning ground truth** — an optimal path to compare a planner against, or
  a reachability check before spending a run;
* **choosing goals** — sampling start/goal pairs that are genuinely in free
  space and genuinely connected, without flying the world first.

It is *not* an exploration map. Every cell here is known: geometry or free
space, nothing else.

## Run it

The hospital, exactly as committed under `robots/SJTU/maps/`:

```bash
cd /path/to/TheAgency
.venv/bin/python -m sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map \
    --world  /path/to/aws-robomaker-hospital-world/worlds/hospital.world \
    --search-path /path/to/aws-robomaker-hospital-world/models \
    --search-path /path/to/aws-robomaker-hospital-world/fuel_models \
    --output-dir sparx_agency/robots/SJTU/maps \
    --resolution 0.05 --z-min 0.30 --z-max 2.00
```

About 16 seconds for 180 model instances and 168k triangles.

| flag | default | what it is for |
|---|---|---|
| `--world` | required | A `.world` path, or a bare name resolved against the search paths |
| `--output-dir` | required | Where the three artifacts land |
| `--name` | the world's stem | Basename of the artifacts |
| `--resolution` | `0.05` | Metres per cell |
| `--z-min` / `--z-max` | `0.30` / `2.00` | The band the robot's body sweeps |
| `--margin` | `1.0` | Free space added around the geometry |
| `--search-path` | — | Repeatable; where `model://` resolves. Highest priority |
| `--skip` | `drone` | Repeatable substring; excludes matching models |
| `--strict` | off | Fail rather than report an unresolved `model://` include |
| `--preview` | — | Also write a PNG to eyeball |

Search paths are tried in order: `--search-path` flags first, then the world's
sibling `models/`, `fuel_models/` and `meshes/` directories (the aws-robomaker
layout, found automatically), then `GAZEBO_MODEL_PATH`. Nothing is hardcoded.

## What counts as a missing model, and what is always fatal

Three different things end up unresolved, and they are not equally serious:

* **Gazebo's own `sun` and `ground_plane`** live in Gazebo's model database
  rather than on the model path, so *every* world fails to resolve them here.
  Neither holds mappable geometry — a light and an infinite plane — so they are
  reported as built-ins and `--strict` passes over them. Without that exemption
  `--strict` refuses every real world, which makes it useless, which is how a
  genuinely missing model went on reading as routine.
* **Any other unresolved `model://` include** is a warning, and an error under
  `--strict`.
* **An unresolved mesh *file*** aborts the build, `--strict` or not. There is no
  benign version of it: the link exists, it declares that it is shaped like that
  file, and the file is gone — so a wall, a lift shaft or a whole ward is absent
  from a map that still looks complete.

## The height band, and why it is not zero

`--z-min 0.30 --z-max 2.00` is what a drone cruising at 1.2 m can hit. Slicing
matters in both directions: below 0.30 m sit the floor, the skirting and the
ramp; above 2.00 m sit the ceiling, the light fittings and the door lintels. A
naive top-down projection turns all of it into walls and closes every doorway.

## The COLLADA unit trap

**Read this before changing `mesh_cache.py`.** A `.dae` declares its own unit in
`<asset><unit meter="0.01"/></asset>`. Gazebo honours it. `trimesh.load(...,
force="mesh")` applies the file's Y_UP-to-Z_UP axis conversion but **not** that
scale. The hospital's wall mesh therefore loads spanning
`x[-1255.8, 1255.8] y[-3505.8, 2105.8]` — centimetres read as metres.

Nothing downstream can catch it. The map's extent is measured from the same
geometry, so a 100x error produces a map that looks entirely plausible and is
100x too big, in a frame that no longer matches the robot's odometry. So the
unit is read out of the XML by hand and applied in `mesh_cache.py`.

For the same reason a `.dae` whose unit cannot be *read* — unparseable XML, a
`meter` that is not a number, a `meter` that is not positive — aborts the build
instead of quietly assuming 1.0. Assuming 1.0 is the failure, not the recovery.
A COLLADA file that declares no `<unit>` at all is a different case and does
mean metres: that is COLLADA's own default.

The correct answer for the hospital walls, and a good thing to re-check after
any trimesh upgrade:

```
x[-12.56, 12.56]  y[-35.06, 21.06]  z[-3.00, 3.00]   metres
```

`.obj` files carry no unit of their own; they get an explicit `<scale>` in the
SDF (the Chair is `0.00817`), which is read from the SDF and applied instead.

## Frames and row order

* **World frame.** The map is in the Gazebo world frame, unrotated and
  untranslated — the same frame the sjtu_drone plugin publishes on
  `/simple_drone/odom`, which reports `model->WorldPose()` directly. A pose
  read off that topic indexes straight into this map with no transform.
* **The `.npz` grid** is indexed `[row, col]` with **row 0 at minimum y**,
  matching `OccupancyGrid2D` and everything else in `core/`.
* **The `.pgm`** is indexed with **row 0 at maximum y**, because that is what
  nav2 expects. `nav2_map.py` is the only place that flip happens.

## What "occupied" means here

A cell is occupied when scene geometry passes through it somewhere inside the
band. Everything else is free. **There are no unknown cells** — not outside the
building, not inside a cupboard. That is the honest reading of a ground-truth
map: nothing was unobserved, because nothing was observed in the first place.
It is also why this map must not be handed to an exploration algorithm that
reads "free" as "already seen".

One consequence worth knowing: the world's models are *surfaces*, so a large
closed object slices to a hollow ring rather than a solid blob. The ring is
watertight — the rasteriser guarantees 4-connected edges — so the interior is
enclosed and unreachable, and a planner can never route through it. But it does
read as free, so do not sample a goal uniformly from free space and expect
every draw to be reachable.

## Layout

| file | what it does |
|---|---|
| `sdf_scene.py` | Walk the world tree, compose poses, flatten it to placed shapes |
| `geometry_instance.py` | The flat vocabulary: a placed shape, and what was skipped |
| `sdf_elements.py` | Read a `<pose>` (extrinsic XYZ) and a child element's text |
| `model_lookup.py` | Resolve `model://` and `file://`, find a model's SDF |
| `resource_paths.py` | Find the world and its model directories, no hardcoded paths |
| `mesh_cache.py` | Load a mesh to `(vertices, faces)` in metres — unit trap and LRU cache |
| `primitives.py` | Tessellate `<box>`, `<cylinder>`, `<sphere>` |
| `scene_raster.py` | Place the geometry, measure the extent, rasterise into one grid |
| `nav2_map.py` | Write and read the nav2 PGM + YAML, including the row flip |
| `build_map.py` | The CLI |

The rasterisation itself is not here. It is
`core/mapping/geometry_raster` — pure numpy, no mesh library, no ROS — so the
algorithm stays testable and reusable while this package owns the messy job of
turning a simulator's file formats into triangles. Read that package's README
for why polygon edges are stamped as well as interiors filled.

`<collision>` geometry wins over `<visual>`; `<visual>` is used only for links
that declare no collision at all. `<plane>` is ignored — it is the infinite
ground.

## Tests

```bash
.venv/bin/python -m pytest sparx_agency/tasks/mapping/gazebo_world_occupancy -q
```

The suites build worlds in a `tmp_path` and check the answers exactly: pose
composition through include/link/collision, the first-search-path-wins rule,
the collision-over-visual preference, a centimetre `.dae` loading as metres
(alongside the raw trimesh call that does not), and a two-box world producing
occupied cells where the boxes are, free cells between them, the PGM flipped
and the YAML fields nav2 wants.
