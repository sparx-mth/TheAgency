"""Duplicate a scene's own obstacle assets into new places, for denser clutter.

A VLA training campaign wants realistic obstacle density; the stock Isaac Sim
indoor scenes are sparse (``warehouse``'s bare shell is the extreme case, see
``scene.py``'s ``INDOOR_SCENES`` comment). Rather than importing foreign assets
-- a materials/scale/style mismatch against the rest of the scene -- this
copies prims that are *already in the loaded scene* to new positions, so a
duplicate has the same textures, collision geometry and visual style as the
original.

Where new positions come from is deliberately not here: :mod:`obstacle_placement`
picks them, purely against the scene's own surveyed map, with no Isaac import,
so that choice is unit-tested. This module only knows how to find a
duplicatable prim and copy it.

**There is no USD export here on purpose.** An earlier version exported the
augmented stage's root layer to a new ``.usd`` and re-referenced *that* through
the normal ``load_indoor_scene(name, prim_path="/World/Scene")`` path -- but
that file's root layer already contains a prim literally named
``/World/Scene`` (holding the original CDN reference), so referencing it again
under a target also named ``/World/Scene`` nests it one level deeper
(``/World/Scene/World/Scene/...``) instead of replacing it. The building's own
geometry still swept fine (PhysX resolves world-space transforms regardless of
nesting), but the indoor/outdoor ceiling test in
``voxel_grid_3d.restrict_to_indoor`` assumes the ordinary single-nesting
layout and came back with the entire building reclassified as outdoors -- a
survey of 0 occupied, 0 free, all-unknown. The fix is to never bake a new USD
file at all: :func:`scene.load_augmented_scene` instead loads the base scene
normally (one clean reference, no nesting) and replays the same
:func:`duplicate_prim` calls against it every time, from a small JSON recipe
of ``(source_path, delta_xyz, rotation_z_deg)`` tuples.

Must run inside a live Isaac Sim process (needs ``omni``/``pxr``), after the
scene has been loaded and the timeline reset -- see
``tasks/planning/sim_flight_recording/augment_scene.py``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Structural prims to never duplicate -- floor/walls/ceiling/lighting/sky make
# a corrupted-looking scene rather than a denser one, and are usually a very
# different size from a real freestanding obstacle anyway.
EXCLUDE_KEYWORDS = (
    "floor", "wall", "ceiling", "roof", "ground", "sky", "dome", "backdrop",
    "light", "environment", "groundplane",
    # Fixtures that are structurally installed (plumbed, hinged, bolted to a
    # wall) -- duplicating one loose into open floor reads as a rendering
    # glitch rather than a denser room, which is the opposite of realistic
    # training data. Doors and clocks matter more once tall obstacles are
    # selected for (see min_reach_height_m): a door slab floating mid-room
    # with no frame is the single most obvious case of this.
    "toilet", "urinal", "sink", "bathtub", "door", "clock", "lamp", "elevator",
)
# A candidate's world-space bounding box must fall in this range, metres, on
# its longest axis. Filters out both small fixtures (a door handle, a few cm)
# and structural elements too large to read as "one obstacle" (a whole rack
# row) -- those get skipped in favour of descending into their own children.
MIN_EXTENT_M = 0.15
MAX_EXTENT_M = 3.0


def list_obstacle_prims(
    root: str = "/World/Scene",
    min_extent_m: float = MIN_EXTENT_M,
    max_extent_m: float = MAX_EXTENT_M,
    exclude_keywords: Sequence[str] = EXCLUDE_KEYWORDS,
    min_reach_height_m: float = 0.0,
    floor_z_m: float = 0.0,
) -> List[Dict]:
    """Find prims under ``root`` that look like duplicatable obstacles.

    Walks the stage depth-first and accepts the first prim on each branch
    whose world bounding box falls in ``[min_extent_m, max_extent_m]`` on its
    longest axis, then stops descending into it -- so a shelf is one
    candidate, not the shelf plus every board on it.

    Args:
        root: Stage path the indoor scene was referenced at, see
            :func:`~sparx_agency.robots.PEGASUS.adapters.scene.load_indoor_scene`.
        min_extent_m: Smallest allowed longest-axis world bounding-box size.
        max_extent_m: Largest allowed longest-axis world bounding-box size.
        exclude_keywords: Case-insensitive substrings that disqualify a prim
            (and its whole subtree) by path -- structural elements, not
            obstacles.
        min_reach_height_m: Keep only prims whose bounding box *top* reaches at
            least this height above ``floor_z_m``. ``min_extent_m`` alone does
            not do this: a desk lying flat can have a 1 m diagonal extent
            without standing more than 75 cm off the floor. A drone cruising
            at ~1-1.5 m only cares about obstacles that actually reach that
            band -- columns, partitions, door frames, tall racks -- not desk
            clutter that a scan at that altitude would never touch. 0
            (default) disables the filter.
        floor_z_m: World z of the floor, for ``min_reach_height_m``.

    Returns:
        One dict per candidate: ``path``, ``extent_m`` (longest bbox axis),
        ``top_m`` (world z of the bounding box top, height above the floor
        this object actually reaches).

    Raises:
        RuntimeError: If ``root`` does not exist on the current stage.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        raise RuntimeError(f"no prim at {root!r} -- load the scene first")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    candidates: List[Dict] = []
    # Usd.PrimRange itself has no PruneChildren -- only the iterator it
    # produces does, so the range must be turned into one explicitly.
    prim_iter = iter(Usd.PrimRange(root_prim))
    for prim in prim_iter:
        if prim == root_prim:
            continue
        path_lower = prim.GetPath().pathString.lower()
        if any(keyword in path_lower for keyword in exclude_keywords):
            prim_iter.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue

        box_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box_range.IsEmpty():
            continue
        extent = float(max(box_range.GetSize()))
        if extent < min_extent_m:
            continue  # too small alone -- the real obstacle may be a child
        if extent > max_extent_m:
            continue  # too big for one obstacle -- descend into its parts

        top_m = float(box_range.GetMax()[2]) - floor_z_m
        if top_m < min_reach_height_m:
            continue  # tall enough on its longest axis, but lying flat

        candidates.append({
            "path": prim.GetPath().pathString, "extent_m": extent, "top_m": top_m,
        })
        prim_iter.PruneChildren()

    return candidates


def duplicate_prim(
    dest_path: str,
    source_path: str,
    delta_xyz: Tuple[float, float, float],
    rotation_z_deg: float = 0.0,
) -> str:
    """Copy a prim to a new stage path, offset by a world-frame delta.

    Computes the copy's new *world* transform directly (old world position
    plus ``delta_xyz``, an extra yaw about world Z, same rotation/scale
    otherwise) and converts that back into the copy's local transform. Working
    in world space is what makes ``delta_xyz`` mean what it says regardless of
    whatever local rotation the source prim already carries.

    **The source's world transform is read before the copy, not after.** The
    copy is created at ``dest_path``, under a different parent than the
    source's own (typically a flat ``AugmentedObstacles`` group) -- but it
    keeps the source's raw local xform ops, which were authored relative to
    the *source's* parent. Computing "world transform" from the already-
    reparented copy silently reinterprets those ops under the wrong parent,
    which is only ever invisible when that parent happens to be identity.
    Object copies with a source in a translated/rotated parent group (a
    furniture cluster, a rack row) instead came out at wildly wrong world
    positions -- confirmed by a preview PNG showing stray obstacle marks
    outside the building envelope after a large batch.

    Args:
        dest_path: Stage path for the new copy. Must not already exist.
        source_path: Prim to copy, typically from :func:`list_obstacle_prims`.
        delta_xyz: World-frame ``(dx, dy, dz)`` added to the source's existing
            world position -- relative rather than absolute, so the copy keeps
            whatever height off the floor the original had.
        rotation_z_deg: Extra yaw about world Z applied on top of the source's
            own rotation, for visual variety between copies of the same asset.

    Returns:
        ``dest_path``.

    Raises:
        RuntimeError: If the copy command fails.
    """
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    time = Usd.TimeCode.Default()

    source_prim = stage.GetPrimAtPath(source_path)
    source_world = UsdGeom.Xformable(source_prim).ComputeLocalToWorldTransform(time)

    success, _ = omni.kit.commands.execute(
        "CopyPrim", path_from=source_path, path_to=dest_path, exclusive_select=False,
    )
    if not success:
        raise RuntimeError(f"CopyPrim {source_path!r} -> {dest_path!r} failed")

    dest_prim = stage.GetPrimAtPath(dest_path)
    xformable = UsdGeom.Xformable(dest_prim)
    parent_world = UsdGeom.Xformable(dest_prim.GetParent()).ComputeLocalToWorldTransform(time)

    # Order matters and got this wrong once already: Gf.Matrix4d is a row-vector
    # (p' = p * M) convention, so a matrix placed to the RIGHT of source_world
    # acts on an already-world-space point -- a "yaw" there rotates the
    # object's *position* around the world origin, not its own facing, which
    # for an object tens of metres from the origin threw it tens of metres
    # across the map (confirmed: every duplicate in an 80-object batch landed
    # 3-92 m from its intended target, each error roughly matching a circular
    # displacement for that object's distance from the origin). ``yaw`` must
    # compose on the *local* side (spins the object about its own pivot before
    # ``source_world`` places it); ``translation`` still belongs on the world
    # side (added to whatever the point already is, so it means the same
    # world-frame delta regardless of the object's own rotation).
    translation = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*delta_xyz))
    yaw = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_z_deg))
    new_world = yaw * source_world * translation
    new_local = new_world * parent_world.GetInverse()

    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(new_local)
    return dest_path


def augment_with_duplicates(
    placements,
    candidates: List[Dict],
    dest_root: str = "/World/AugmentedObstacles",
    rng=None,
) -> List[Dict]:
    """Place one duplicated obstacle per placement, picked randomly from ``candidates``.

    Args:
        placements: :class:`~obstacle_placement.Placement` instances -- world
            ``(x, y, rotation_deg)`` for each new obstacle.
        candidates: Source prims to duplicate from, as returned by
            :func:`list_obstacle_prims`. Must be non-empty.
        dest_root: Stage path duplicates are created under. Created if
            missing.
        rng: Source of randomness for which candidate is copied to each
            placement. A fresh default generator if omitted.

    Returns:
        One dict per placement actually created: ``source``, ``dest_path``,
        ``x``, ``y``, ``rotation_deg``, and the ``(dx, dy, dz)`` delta applied
        -- the last is what a caller needs to replay this same placement later
        (see ``scene.load_augmented_scene``).

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom

    if not candidates:
        raise ValueError("no candidate obstacle prims to duplicate from")

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(dest_root).IsValid():
        UsdGeom.Xform.Define(stage, dest_root)

    rng = rng if rng is not None else np.random.default_rng()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    placed = []
    for i, placement in enumerate(placements):
        source = candidates[int(rng.integers(0, len(candidates)))]
        source_path = source["path"]
        source_prim = stage.GetPrimAtPath(source_path)
        origin = cache.ComputeWorldBound(source_prim).ComputeAlignedRange().GetMidpoint()

        dest_path = f"{dest_root}/dup_{i:03d}"
        delta = (placement.x - origin[0], placement.y - origin[1], 0.0)
        duplicate_prim(dest_path, source_path, delta, placement.rotation_deg)
        placed.append({
            "source": source_path, "dest_path": dest_path,
            "x": placement.x, "y": placement.y,
            "rotation_deg": placement.rotation_deg,
            "dx": delta[0], "dy": delta[1], "dz": delta[2],
        })
    return placed
