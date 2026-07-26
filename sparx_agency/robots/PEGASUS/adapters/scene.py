"""Load one of NVIDIA's stock Isaac Sim indoor environments onto the stage.

Confirmed reachable (HTTP 200) from the ``isaac-sim`` container against the
Isaac Sim 4.5 asset CDN on 2026-07-26; ``House`` and ``Library`` do not exist in
this pack (404) -- there is no bundled home/library scene. See
``robots/PEGASUS/README.md`` for the full survey.
"""
from __future__ import annotations

_ASSET_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.5/Isaac/Environments"
)

INDOOR_SCENES = {
    "simple_room": f"{_ASSET_BASE}/Simple_Room/simple_room.usd",
    "hospital": f"{_ASSET_BASE}/Hospital/hospital.usd",
    "office": f"{_ASSET_BASE}/Office/office.usd",
    "warehouse": f"{_ASSET_BASE}/Simple_Warehouse/warehouse.usd",
    "full_warehouse": f"{_ASSET_BASE}/Simple_Warehouse/full_warehouse.usd",
}


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
