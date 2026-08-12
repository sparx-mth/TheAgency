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

from typing import Dict, List, Optional, Sequence, Tuple

# How far a finished duplicate's centre may sit from its intended target
# before it is treated as a failed placement and dropped rather than left in
# the scene at an arbitrary position. Set well above ordinary placement
# rounding noise (centimetres) and well below the tens-of-metres errors a real
# failure produces. Used by both :func:`duplicate_prim_verified` call sites --
# the interactive generation pass in ``augment_scene.py`` *and* every replay
# of the recipe (``scene.load_augmented_scene``). The two are not
# interchangeable: a placement verified once at generation time is not
# guaranteed to reproduce identically on a fresh replay (confirmed on a
# warehouse scene with heavily shared/referenced box and pallet geometry --
# some placements that passed verification during generation still landed
# tens of metres off on a later, separately-booted replay), so the check has
# to run again every time the recipe is applied, not just once when it is
# written.
VERIFY_TOLERANCE_M = 2.0

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


def vertical_placement(
    source_base_m: float,
    source_top_m: float,
    floor_z_m: float = 0.0,
    stretch_top_m: float = None,
    span_m: float = None,
    centre_m: float = None,
) -> Tuple[float, float]:
    """How to scale and lift a copy so it occupies the height you meant.

    Pure arithmetic, deliberately separated from the USD work so the one part
    of a placement that is easy to get silently wrong can be tested without a
    simulator. Both scaling modes anchor at the object's own base, matching
    :func:`duplicate_prim`'s local-space stretch.

    Two modes, and the difference is what a drone can do about the result:

    * ``stretch_top_m`` grows a floor-standing prop until its top reaches that
      height. Set it *past* the cruise altitude, not to it. Set to exactly the
      altitude the obstacle merely reaches the aircraft's own plane, which is
      the case the aircraft slips over -- measured on the first augmented
      office and warehouse, where added occupancy peaked well below 1.5 m and
      fell away immediately above it.
    * ``span_m`` + ``centre_m`` make an obstacle at least that tall and hang it
      centred on that height, clear of the floor. This is the case a route has
      to go *around*: there is nothing under it to land the aircraft on and
      nothing over it within the flight band, so no altitude change helps.

    Args:
        source_base_m: World z of the source prim's lowest point.
        source_top_m: World z of its highest point.
        floor_z_m: World z of the floor, which ``stretch_top_m`` measures from.
        stretch_top_m: Grow the copy until its top is this high above the
            floor. Never shrinks (scale is clamped to >= 1).
        span_m: Make the copy at least this tall. Never shrinks.
        centre_m: Put the copy's vertical midpoint here, lifting it off the
            floor. None leaves the copy at the source's own height.

    Returns:
        ``(scale_z, extra_dz)`` -- the local Z scale and the world-frame z
        offset to add to the placement delta, in that order of application.
    """
    height = max(source_top_m - source_base_m, 1e-6)
    scale_z = 1.0
    if stretch_top_m is not None and source_top_m - floor_z_m > 1e-6:
        scale_z = max(scale_z, stretch_top_m / (source_top_m - floor_z_m))
    if span_m is not None:
        scale_z = max(scale_z, span_m / height)

    extra_dz = 0.0
    if centre_m is not None:
        # Scaling is anchored at the base, so after it the copy spans
        # [base, base + scale*height] and its midpoint has moved.
        extra_dz = centre_m - (source_base_m + scale_z * height / 2.0)
    return scale_z, extra_dz


def duplicate_prim(
    dest_path: str,
    source_path: str,
    delta_xyz: Tuple[float, float, float],
    rotation_z_deg: float = 0.0,
    stretch_top_m: float = None,
    floor_z_m: float = 0.0,
    float_span_m: float = None,
    float_centre_m: float = None,
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
        stretch_top_m: If given, scale the copy along Z (anchored at its own
            base, not at the world origin -- see below) so its top reaches at
            least this height above ``floor_z_m``. A drone cruising through a
            band it merely *pokes into* can still slip over or under it;
            stretching guarantees the obstacle actually spans the band. Never
            shrinks a copy that already reaches this height (scale factor is
            clamped to >= 1). See :func:`vertical_placement` for why this
            wants to be set past the cruise altitude rather than to it.
        floor_z_m: World z of the floor, for ``stretch_top_m``.
        float_span_m: Make the copy at least this tall, as
            :func:`vertical_placement`.
        float_centre_m: Hang the copy with its vertical midpoint at this
            height, clear of the floor -- an obstacle a route has to go around
            rather than over or under.

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

    # Scale factor computed from the source's own bounding box, before the
    # copy: assumes the object's local origin sits at its own floor contact
    # point (true of the floor-standing props this is meant for -- columns,
    # racks, boxes -- which is also why min_reach_height_m/EXCLUDE_KEYWORDS
    # steer away from wall/ceiling-mounted fixtures). Scaling about that local
    # origin, in the object's own local frame, stretches it upward from where
    # it already touches the floor rather than growing it from the middle or
    # dragging its base off the ground.
    scale_z, extra_dz = 1.0, 0.0
    if stretch_top_m is not None or float_span_m is not None or float_centre_m is not None:
        cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_])
        source_range = cache.ComputeWorldBound(source_prim).ComputeAlignedRange()
        scale_z, extra_dz = vertical_placement(
            float(source_range.GetMin()[2]), float(source_range.GetMax()[2]),
            floor_z_m=floor_z_m, stretch_top_m=stretch_top_m,
            span_m=float_span_m, centre_m=float_centre_m)
    delta_xyz = (delta_xyz[0], delta_xyz[1], delta_xyz[2] + extra_dz)

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
    # displacement for that object's distance from the origin). ``yaw`` and
    # ``stretch`` must both compose on the *local* side (act on the object
    # about its own pivot before ``source_world`` places it) for the same
    # reason a world-side scale would grow the object away from the world
    # origin instead of from its own base; ``translation`` still belongs on
    # the world side (added to whatever the point already is, so it means the
    # same world-frame delta regardless of the object's own rotation/scale).
    stretch = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(1.0, 1.0, scale_z))
    yaw = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_z_deg))
    translation = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*delta_xyz))
    new_world = stretch * yaw * source_world * translation
    new_local = new_world * parent_world.GetInverse()

    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(new_local)
    return dest_path


def duplicate_prim_verified(
    dest_path: str,
    source_path: str,
    target_xy: Tuple[float, float],
    delta_xyz: Tuple[float, float, float],
    rotation_z_deg: float = 0.0,
    stretch_top_m: float = None,
    floor_z_m: float = 0.0,
    float_span_m: float = None,
    float_centre_m: float = None,
    tolerance_m: float = VERIFY_TOLERANCE_M,
) -> Optional[str]:
    """:func:`duplicate_prim`, then confirm it landed near ``target_xy`` or undo it.

    Trusting a single placement computation has twice been wrong on this
    codebase already (see :func:`duplicate_prim`'s docstring, and
    :func:`augment_with_duplicates`'s note on a reused ``BBoxCache`` corrupting
    the delta before it was even stored) -- and a placement verified once is
    not guaranteed to reproduce identically on a later, separately-booted
    replay of the same recipe. So every call site checks, including replay
    (:func:`~sparx_agency.robots.PEGASUS.adapters.scene.load_augmented_scene`),
    not just the one-off generation pass.

    Args:
        dest_path: Stage path for the new copy.
        source_path: Prim to copy.
        target_xy: World-frame ``(x, y)`` the duplicate's centre was meant to
            land at.
        delta_xyz: As :func:`duplicate_prim`.
        rotation_z_deg: As :func:`duplicate_prim`.
        stretch_top_m: As :func:`duplicate_prim`.
        floor_z_m: As :func:`duplicate_prim`.
        float_span_m: As :func:`duplicate_prim`.
        float_centre_m: As :func:`duplicate_prim`.
        tolerance_m: How far from ``target_xy`` is still considered a good
            placement.

    Returns:
        ``dest_path`` if the duplicate landed within ``tolerance_m`` of
        ``target_xy``; ``None`` if it did not (the failed copy is removed from
        the stage before returning).
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    duplicate_prim(dest_path, source_path, delta_xyz, rotation_z_deg,
                   stretch_top_m=stretch_top_m, floor_z_m=floor_z_m,
                   float_span_m=float_span_m, float_centre_m=float_centre_m)

    stage = omni.usd.get_context().get_stage()
    dest_prim = stage.GetPrimAtPath(dest_path)
    # A fresh cache for every verification -- see augment_with_duplicates'
    # note on why one reused across stage edits returned stale data.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    dest_range = cache.ComputeWorldBound(dest_prim).ComputeAlignedRange()
    if dest_range.IsEmpty():
        stage.RemovePrim(dest_prim.GetPath())
        return None

    mid = dest_range.GetMidpoint()
    error_m = ((mid[0] - target_xy[0]) ** 2 + (mid[1] - target_xy[1]) ** 2) ** 0.5
    if error_m > tolerance_m:
        stage.RemovePrim(dest_prim.GetPath())
        return None
    return dest_path


def augment_with_duplicates(
    placements,
    candidates: List[Dict],
    dest_root: str = "/World/AugmentedObstacles",
    rng=None,
    stretch_top_m: float = None,
    float_span_m: float = None,
    float_centre_m: float = None,
    start_index: int = 0,
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
        stretch_top_m: Forwarded to :func:`duplicate_prim` -- stretch every
            copy that doesn't already reach this height, so a flight band it
            merely brushed becomes one it cannot slip over or under.
        float_span_m: Forwarded to :func:`duplicate_prim`.
        float_centre_m: Forwarded to :func:`duplicate_prim` -- hang this whole
            batch at that height instead of standing it on the floor.
        start_index: First number for the ``dup_NNN`` destination paths. A
            scene is built from more than one batch (floor-standing clutter and
            obstacles hung in the flight band are separate calls), and they
            share one destination group, so a second batch starting at zero
            would collide with the first batch's paths.

    Returns:
        One dict per placement actually created: ``source``, ``dest_path``,
        ``x``, ``y``, ``rotation_deg``, the ``(dx, dy, dz)`` delta applied, and
        the vertical treatment (``stretch_top_m``, ``float_span_m``,
        ``float_centre_m``) it was placed with -- together, everything a caller
        needs to replay this same placement later, batch by batch (see
        ``scene.load_augmented_scene``).

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

    placed = []
    dropped = 0
    for i, placement in enumerate(placements):
        source = candidates[int(rng.integers(0, len(candidates)))]
        source_path = source["path"]
        source_prim = stage.GetPrimAtPath(source_path)
        # A fresh cache every iteration, not one reused across the loop.
        # UsdGeomBBoxCache is documented as unsafe to reuse across stage
        # edits -- reusing one here (each iteration's duplicate_prim call
        # mutates the stage) returned a stale extent for some sources on a
        # warehouse scene heavy with shared/referenced box and pallet
        # geometry, corrupting the delta computed below *before* it was even
        # stored: a batch of 300 came back with 22 duplicates 10-126 m from
        # where they were meant to be.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        origin = cache.ComputeWorldBound(source_prim).ComputeAlignedRange().GetMidpoint()

        dest_path = f"{dest_root}/dup_{start_index + i:03d}"
        delta = (placement.x - origin[0], placement.y - origin[1], 0.0)
        kept = duplicate_prim_verified(
            dest_path, source_path, (placement.x, placement.y), delta,
            placement.rotation_deg, stretch_top_m=stretch_top_m,
            float_span_m=float_span_m, float_centre_m=float_centre_m)
        if kept is None:
            dropped += 1
            continue

        placed.append({
            "source": source_path, "dest_path": dest_path,
            "x": placement.x, "y": placement.y,
            "rotation_deg": placement.rotation_deg,
            "dx": delta[0], "dy": delta[1], "dz": delta[2],
            "stretch_top_m": stretch_top_m,
            "float_span_m": float_span_m, "float_centre_m": float_centre_m,
        })

    if dropped:
        print(f"augment_with_duplicates: dropped {dropped}/{len(placements)} "
              f"duplicates that landed >{VERIFY_TOLERANCE_M:.1f} m from target",
              flush=True)
    return placed
