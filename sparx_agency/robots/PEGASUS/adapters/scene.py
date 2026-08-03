"""Load one of NVIDIA's stock Isaac Sim indoor environments.

Confirmed reachable (HTTP 200) from the ``isaac-sim`` container against the
Isaac Sim 4.5 asset CDN on 2026-07-26; ``House`` and ``Library`` do not exist in
this pack (404) -- there is no bundled home/library scene. See
``robots/PEGASUS/README.md`` for the full survey.

**Where a flight is safe to go is not here.** That is a surveyed map, measured
per scene by ``tasks/planning/sim_flight_recording/survey_scene.py`` and read
back through :mod:`scene_map`. The hand-measured spawn points below predate that
and survive only because the no-autopilot debugging script
(``fly_direct.py``) needs somewhere to start and does no planning at all.
"""
from __future__ import annotations

import json
from pathlib import Path

_ASSET_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.5/Isaac/Environments"
)

INDOOR_SCENES = {
    "simple_room": f"{_ASSET_BASE}/Simple_Room/simple_room.usd",
    # NOTE: hospital's USD loads fine on its own, but any run that also enables
    # the pegasus.simulator extension (i.e. every real flight or survey) reliably
    # crashes Kit ~2-3s in with a std::out_of_range in
    # libomni.anim.behavior.core.plugin.so -- before scene loading even starts.
    # Disabling that extension avoids the crash but takes pegasus.simulator
    # down with it (a hard dependency), so it is not a usable workaround.
    # Currently unusable for real flights. See
    # tasks/planning/sim_flight_recording/README.md and robots/PEGASUS/README.md.
    "hospital": f"{_ASSET_BASE}/Hospital/hospital.usd",
    "office": f"{_ASSET_BASE}/Office/office.usd",
    # The bare shell. Almost nothing to avoid -- four walls and a floor -- so it
    # is a poor source of obstacle-avoidance data even though it flies fine.
    # Prefer the two furnished variants below.
    "warehouse": f"{_ASSET_BASE}/Simple_Warehouse/warehouse.usd",
    "warehouse_shelves": f"{_ASSET_BASE}/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_forklifts": f"{_ASSET_BASE}/Simple_Warehouse/warehouse_with_forklifts.usd",
    "full_warehouse": f"{_ASSET_BASE}/Simple_Warehouse/full_warehouse.usd",
}

# Hand-measured open spots, one per scene, from the raycast survey that preceded
# the occupancy maps. Only ``fly_direct.py`` still uses these: it applies forces
# from a scripted pattern and needs a start point but has no map to draw one
# from. Anything that plans a route samples its start out of the surveyed map
# instead, which is measured rather than remembered.
SCENE_SPAWNS = {
    "simple_room": (-0.5, 1.0),
    "office": (-4.0, 3.5),
    # Both warehouse assets are centred on the origin with open floor there,
    # confirmed by surveying from it: 535 m2 and 1446 m2 of contiguous flyable
    # space came back, which a spawn inside a rack or outside the building
    # could not produce.
    "warehouse": (0.0, 0.0),
    "warehouse_shelves": (0.0, 0.0),
    "warehouse_forklifts": (0.0, 0.0),
    "full_warehouse": (0.0, 0.0),
}

SCENE_SWEEP_CEILING_M = {
    "warehouse": 14.0,
    "warehouse_shelves": 14.0,
    "warehouse_forklifts": 14.0,
    "full_warehouse": 14.0,
}
"""How high to sweep, for scenes whose roof is above the 6 m default.

``restrict_to_indoor`` calls a column with no ceiling above it outdoors, so a
warehouse swept only to 6 m comes back as *no* indoor space at all -- the survey
then dies in ``largest_region`` with "no cell in the map has 0.35 m of
clearance", which reads like an empty scene rather than a too-short sweep.
"""

LOCAL_SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"
"""Where ``augment_scene.py`` writes locally generated obstacle-duplication recipes.

Never committed (see ``.gitignore``) -- regenerated per device, same as the
``.ply`` point clouds under ``maps/``.
"""

AUGMENTED_SCENES: dict = {}
"""Recipe-name -> ``.json`` path, populated by :func:`_register_local_scenes`.

An augmented scene is not a second USD file on the CDN pattern of
:data:`INDOOR_SCENES` -- see :func:`load_augmented_scene` for why -- so it
gets its own registry rather than being folded into that one.
"""


def _register_local_scenes() -> None:
    """Add any ``scenes/*.json`` recipe found on disk to :data:`AUGMENTED_SCENES`.

    A recipe names the stock scene it was derived from (``base_scene``), so
    the augmented variant can inherit that scene's :data:`SCENE_SPAWNS` /
    :data:`SCENE_SWEEP_CEILING_M` entries rather than needing its own -- the
    building layout, and therefore the known-open spawn point, did not change;
    only the clutter did.
    """
    if not LOCAL_SCENES_DIR.is_dir():
        return
    for recipe_path in sorted(LOCAL_SCENES_DIR.glob("*.json")):
        name = recipe_path.stem
        AUGMENTED_SCENES[name] = recipe_path

        base = json.loads(recipe_path.read_text()).get("base_scene")
        if base in SCENE_SPAWNS:
            SCENE_SPAWNS.setdefault(name, SCENE_SPAWNS[base])
        if base in SCENE_SWEEP_CEILING_M:
            SCENE_SWEEP_CEILING_M.setdefault(name, SCENE_SWEEP_CEILING_M[base])


_register_local_scenes()

SPAWN_HEIGHT_M = 0.15  # just above the floor -- PX4 needs to detect it is landed at boot

# App ticks between referencing a scene and duplicating one of its children --
# see load_augmented_scene. Matches flight_session.STAGE_SETTLE_STEPS, which
# exists for the same reason (async reference composition).
_AUGMENT_SETTLE_STEPS = 20


def scene_spawn(name: str, z: float = SPAWN_HEIGHT_M) -> tuple:
    """A known-open spot to drop the aircraft into a scene at.

    Args:
        name: A key of :data:`SCENE_SPAWNS`.
        z: Spawn height above the floor, metres.

    Returns:
        World-frame ``(x, y, z)``.

    Raises:
        KeyError: If the scene is unknown or has no recorded spawn.
    """
    if name not in INDOOR_SCENES and name not in AUGMENTED_SCENES:
        raise KeyError(
            f"Unknown indoor scene {name!r}; choose from "
            f"{sorted(set(INDOOR_SCENES) | set(AUGMENTED_SCENES))}"
        )
    if name not in SCENE_SPAWNS:
        raise KeyError(
            f"No recorded spawn point for scene {name!r}. Either add one to "
            f"SCENE_SPAWNS, or use the surveyed map instead -- "
            f"tasks/planning/sim_flight_recording/survey_scene.py --scene {name}"
        )
    x, y = SCENE_SPAWNS[name]
    return x, y, z


def load_indoor_scene(name: str, prim_path: str = "/World/Scene") -> str:
    """Reference one of :data:`INDOOR_SCENES` onto the current stage.

    An ``AUGMENTED_SCENES`` name is dispatched to :func:`load_augmented_scene`
    instead -- it is a duplication recipe over a base scene, not a second CDN
    reference.

    Args:
        name: A key of :data:`INDOOR_SCENES` or :data:`AUGMENTED_SCENES`.
        prim_path: Stage path to spawn the environment reference at.

    Returns:
        The USD path that was referenced (the *base* scene's, for an
        augmented one).

    Raises:
        KeyError: If ``name`` is not a known scene.
    """
    if name in AUGMENTED_SCENES:
        return load_augmented_scene(name, prim_path)
    if name not in INDOOR_SCENES:
        raise KeyError(f"Unknown indoor scene {name!r}; choose from {sorted(INDOOR_SCENES)}")

    from isaacsim.core.utils.stage import add_reference_to_stage

    usd_path = INDOOR_SCENES[name]
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return usd_path


def load_augmented_scene(name: str, prim_path: str = "/World/Scene") -> str:
    """Load a base scene, then replay its recorded obstacle duplications onto it.

    Deliberately not a second USD file referenced the normal way: an earlier
    version exported the augmented stage to its own ``.usd`` and referenced
    that through the same ``add_reference_to_stage`` path as a stock scene,
    which double-nests the building (the exported layer's root layer already
    contains a prim path named ``/World/Scene``, so referencing it again under
    a target also named ``/World/Scene`` composes it one level deeper). PhysX
    still swept the geometry fine -- world transforms are unaffected by extra
    nesting -- but the indoor/outdoor ceiling test in
    ``voxel_grid_3d.restrict_to_indoor`` assumes the ordinary single-nesting
    layout and silently reclassified the whole building as outdoors. Loading
    the base scene fresh and replaying the same duplications every time avoids
    the composition entirely.

    Args:
        name: A key of :data:`AUGMENTED_SCENES`.
        prim_path: Stage path the base scene is referenced at -- the
            duplicated obstacles' source paths are recorded relative to this
            and re-anchored here.

    Returns:
        The base scene's USD path (see :func:`load_indoor_scene`).

    Raises:
        KeyError: If ``name`` is not a known augmented scene.
    """
    if name not in AUGMENTED_SCENES:
        raise KeyError(f"Unknown augmented scene {name!r}; choose from {sorted(AUGMENTED_SCENES)}")

    recipe = json.loads(AUGMENTED_SCENES[name].read_text())
    usd_path = load_indoor_scene(recipe["base_scene"], prim_path)

    # The CDN reference just added composes asynchronously; ticking the app
    # gives it time to finish before anything below reads the source prims.
    import omni.kit.app

    kit_app = omni.kit.app.get_app()
    for _ in range(_AUGMENT_SETTLE_STEPS):
        kit_app.update()

    import omni.usd
    from pxr import UsdGeom

    # Duplicating straight into "<prim_path>/AugmentedObstacles/dup_000" with
    # no prim ever explicitly Defined at the parent path lets CopyPrim
    # auto-vivify it as a typeless Sdf "over" (Usd's implicit behaviour for an
    # unauthored ancestor). An "over" is not IsDefined(), and neither is
    # anything under it, in the *whole* subtree -- not just at that one prim.
    # The duplicates exist on the stage (GetPrimAtPath finds them, CopyPrim
    # reports success) but read as though they do not: GetChildren()'s default
    # predicate excludes them, and so, critically, does the PhysX/omap sweep
    # that surveys occupancy -- an augmented scene surveyed this way came back
    # with an occupied-voxel count byte-identical to the unaugmented one, as
    # if none of the duplicates were ever placed. Defining the group explicitly
    # first is what the interactive augment_scene.py run already did
    # (`augment_with_duplicates`'s ``UsdGeom.Xform.Define``) -- this mirrors it
    # for the replay path.
    obstacles_root = f"{prim_path}/AugmentedObstacles"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(obstacles_root).IsValid():
        UsdGeom.Xform.Define(stage, obstacles_root)

    from sparx_agency.robots.PEGASUS.adapters import scene_augment

    stretch_top_m = recipe.get("stretch_top_m")
    dropped = 0
    for i, obstacle in enumerate(recipe["obstacles"]):
        source_path = f"{prim_path}{obstacle['source_rel']}"
        dest_path = f"{obstacles_root}/dup_{i:03d}"
        # Verified, not just replayed: a placement that passed verification
        # when the recipe was written is not guaranteed to reproduce
        # identically on this fresh boot -- see
        # scene_augment.duplicate_prim_verified's docstring.
        kept = scene_augment.duplicate_prim_verified(
            dest_path, source_path,
            (obstacle["target_x"], obstacle["target_y"]),
            (obstacle["dx"], obstacle["dy"], obstacle["dz"]),
            obstacle["rotation_deg"],
            stretch_top_m=stretch_top_m,
        )
        if kept is None:
            dropped += 1

    if dropped:
        print(f"load_augmented_scene({name!r}): dropped {dropped}/"
              f"{len(recipe['obstacles'])} obstacles that failed to reproduce "
              f"their recorded placement on this replay", flush=True)
    return usd_path
