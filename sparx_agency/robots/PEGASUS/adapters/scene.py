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
    "warehouse": f"{_ASSET_BASE}/Simple_Warehouse/warehouse.usd",
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
}

SPAWN_HEIGHT_M = 0.15  # just above the floor -- PX4 needs to detect it is landed at boot


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
    if name not in INDOOR_SCENES:
        raise KeyError(f"Unknown indoor scene {name!r}; choose from {sorted(INDOOR_SCENES)}")
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

    Args:
        name: A key of :data:`INDOOR_SCENES`.
        prim_path: Stage path to spawn the environment reference at.

    Returns:
        The USD path that was referenced.

    Raises:
        KeyError: If ``name`` is not a known scene.
    """
    if name not in INDOOR_SCENES:
        raise KeyError(f"Unknown indoor scene {name!r}; choose from {sorted(INDOOR_SCENES)}")

    from isaacsim.core.utils.stage import add_reference_to_stage

    usd_path = INDOOR_SCENES[name]
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return usd_path
