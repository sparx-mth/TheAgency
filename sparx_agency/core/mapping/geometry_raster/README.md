# `geometry_raster` — 3D triangle geometry to a 2D occupancy slice

Given the triangles of a scene, already placed in world coordinates, answer one
question: **which grid cells does the geometry occupy between two heights?**

This is how a ground-truth map is made from the geometry a simulator is already
using for collision, rather than by flying a robot around and accumulating what
its sensors happened to see. The result has no unknown cells and no sensor
noise, so it is the reference a live map is judged against — not a replacement
for one.

Pure numpy. No scipy, no mesh library, no OpenCV: loading a `.dae` and working
out its unit scale is the caller's problem (see
`tasks/mapping/gazebo_world_occupancy/`), and this package only ever sees
vertices and faces. That keeps it importable inside the Python 3.8 / numpy 1.17
Noetic container like the rest of `core/`.

## The approach: clip to the slab, then rasterise

```
triangles ──cull──► slab clip ──► convex polygons ──► fill + edge stamp ──► bool grid
```

1. **Cull** (`mesh_occupancy.py`) — drop every triangle whose bounding box
   cannot reach the height band or the grid. In a building that is most of
   them: floors, ceilings, and everything off the side of the map.
2. **Clip** (`slab_clip.py`) — intersect each surviving triangle with the slab
   `z_min <= z <= z_max` by Sutherland-Hodgman against the two half-spaces
   (`halfspace_clip.py`). A triangle clipped by two planes is a convex polygon
   of at most five vertices, so results are carried as a padded `(N, 5, 3)`
   array plus a per-triangle vertex count. A triangle that misses the slab
   comes back with a count of zero.
3. **Rasterise** (`polygon_raster.py`) — draw each polygon's XY projection.

Taking the slice *before* projecting is the point. A wall is only an obstacle
where the robot's body would hit it; a doorway's lintel, a desk's underside and
the floor itself are all geometry a naive top-down projection would turn into
walls.

Everything is batched over triangles, because a building's collision meshes run
to millions of them and a Python loop turns a two-minute job into an hour. The
batches are then walked in slices sized by how much intermediate memory each
item generates (`work_chunks.py`), so a mesh of any size stays inside a fixed
memory budget.

## Why the edges are stamped as well as the interior filled

`polygon_raster.py` does two different things, answering two different failure
modes.

**Fill** (`polygon_fill.py`) tests cell centres against every directed edge —
the classic edge function, evaluated only over the polygon's own bounding box.
That is what draws anything with area: a table top sliced at 0.75 m, a cabinet's
shelf, a ramp.

**Edge stamping** (`edge_raster.py`) draws the polygon's boundary as an unbroken
chain of cells. This is not belt-and-braces, it is the main event:

* A **vertical wall** clipped to a horizontal slab projects to a segment of
  *exactly zero area*. Fill finds nothing. The edges are the entire answer, and
  walls are the geometry a navigation map exists to represent.
* An **11.6 cm wall at 5 cm resolution, at a grazing angle**, passes between
  cell centres for stretches of its length. Centre testing alone rasterises it
  with holes, and a hole in a wall makes the map lie in the most dangerous
  possible direction: a planner will happily route a robot through it.

The chain is **4-connected**. The segment is sampled at half-cell steps, so
consecutive samples always land in the same or an adjacent cell, and wherever
the chain takes a diagonal step both corner cells are added. An 8-connected
chain would still be leaky, because a planner allowing diagonal moves can cut
through the corner between two diagonally adjacent cells.

Two details in the fill are load-bearing rather than decorative:

* **Winding comes from the signed area**, not from an assumption. Mesh
  triangles stop being consistently wound the moment a model is placed with a
  mirroring scale.
* **Degenerate polygons are never filled.** For a zero-area polygon every edge
  function is zero, so *every* cell in its bounding box passes an
  orientation-agnostic test — and a wall running diagonally across a room would
  fill the entire room. Anything below an area threshold is left to the edge
  stamp, which draws it correctly.

## Conventions

**Row 0 is minimum y.** The returned array is indexed `grid[row, col]` with
`row 0` at the *minimum* y and `col 0` at the minimum x — plain grid indexing,
matching `OccupancyGrid2D` in `core/planning/environment/`. This is
deliberately **not** the top-down convention of a nav2 PGM image, where row 0 is
maximum y. Whatever writes an image flips; nothing in this package does.

`GridSpec` (`grid_spec.py`) carries the five numbers that define the raster —
resolution, the world coordinate of the lower-left corner of cell `(0, 0)`, and
the extent — as one value, because a grid separated from its origin is not a
map.

Frames are the caller's: this package never rotates anything. Feed it vertices
already in the frame you want the map in.

## Use

```python
from sparx_agency.core.mapping.geometry_raster import rasterise_mesh_slab

grid = rasterise_mesh_slab(
    vertices, faces,             # (V, 3) float, (F, 3) int, world metres
    z_min=0.30, z_max=2.00,      # the band the robot's body sweeps
    resolution=0.05,
    origin_x=-13.6, origin_y=-36.1,
    width=530, height=1150,
)
```

Pass `out=grid` to accumulate several meshes into one map — that is how a scene
of hundreds of models becomes a single grid. A supplied grid must match the
requested `width` and `height`; every entry point refuses one that does not,
because the alternative is drawing a correct map into the corner of a different
one and returning it without complaint.

## Tests

```
pytest sparx_agency/core/mapping/geometry_raster
```

The three suites test the three stages: a triangle inside, outside and
straddling the slab; a square rasterising to exactly the cells it covers and a
thin diagonal wall having no holes and staying 4-connected; and a closed box
sliced mid-height producing exactly its four walls as a hollow ring, which is
the correct answer for a *surface* mesh and a good check that nothing is
quietly flooding interiors.
