"""Load one of NVIDIA's stock Isaac Sim indoor environments, and fly it safely.

Confirmed reachable (HTTP 200) from the ``isaac-sim`` container against the
Isaac Sim 4.5 asset CDN on 2026-07-26; ``House`` and ``Library`` do not exist in
this pack (404) -- there is no bundled home/library scene. See
``robots/PEGASUS/README.md`` for the full survey.

Beyond loading the USD, this module records **where in each scene it is safe to
fly**. These are furnished buildings with no machine-readable floor plan, so
spawning at the origin and flying a fixed pattern does not work -- the first
``office`` run wedged the drone against a wall 1.7 m behind its spawn point.
:data:`SCENE_SURVEYS` holds the measured answer per scene, produced by
``tasks/planning/sim_flight_recording/probe_scene.py``.
"""
from __future__ import annotations

import math

_ASSET_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.5/Isaac/Environments"
)

INDOOR_SCENES = {
    "simple_room": f"{_ASSET_BASE}/Simple_Room/simple_room.usd",
    # NOTE: hospital crashes Kit ~25 s into loading in Isaac Sim 6.0.1
    # (libomni.anim.behavior.core, std::out_of_range) and is currently unusable.
    # See tasks/planning/sim_flight_recording/README.md.
    "hospital": f"{_ASSET_BASE}/Hospital/hospital.usd",
    "office": f"{_ASSET_BASE}/Office/office.usd",
    "warehouse": f"{_ASSET_BASE}/Simple_Warehouse/warehouse.usd",
    "full_warehouse": f"{_ASSET_BASE}/Simple_Warehouse/full_warehouse.usd",
}

# Free-space surveys, measured at 1.5 m altitude by
# ``tasks/planning/sim_flight_recording/probe_scene.py`` on 2026-07-26.
#
#   spawn     -- the most open indoor cell found; ``clearance`` is how far the
#                nearest obstacle is, in the *worst* of eight directions
#   route     -- waypoints spread across the building, each leg raycast-verified
#                clear of obstacles over a 1 m-wide corridor
#
# Re-run the probe after changing altitude: clearance at head height and
# clearance at desk height are not the same thing.
SCENE_SURVEYS = {
    "simple_room": {
        "spawn": (-0.5, 1.0),
        "clearance": 3.85,
        "route": [(-1.5, -1.0), (0.5, -0.5), (3.0, -0.5), (1.5, -2.0), (2.0, 0.5), (3.0, -2.0)],
    },
    "office": {
        "spawn": (-4.0, 3.5),
        "clearance": 8.62,
        "route": [(4.0, -1.0), (-9.0, -10.5), (-12.5, -1.5), (4.5, -10.0)],
    },
}

SPAWN_HEIGHT_M = 0.15  # just above the floor -- PX4 needs to detect it is landed at boot


def _require_survey(name: str) -> dict:
    if name not in INDOOR_SCENES:
        raise KeyError(f"Unknown indoor scene {name!r}; choose from {sorted(INDOOR_SCENES)}")
    if name not in SCENE_SURVEYS:
        raise KeyError(
            f"No free-space survey for scene {name!r}. Run "
            f"tasks/planning/sim_flight_recording/probe_scene.py --scene {name} "
            f"and add its SPAWN/ROUTE output to SCENE_SURVEYS."
        )
    return SCENE_SURVEYS[name]


def scene_spawn(name: str, z: float = SPAWN_HEIGHT_M) -> tuple:
    """The surveyed, verified-open spawn position for a scene.

    Args:
        name: A key of :data:`SCENE_SURVEYS`.
        z: Spawn height above the floor, metres.

    Returns:
        World-frame ``(x, y, z)``.

    Raises:
        KeyError: If the scene is unknown or has not been surveyed.
    """
    x, y = _require_survey(name)["spawn"]
    return x, y, z


def scene_route(name: str, altitude: float) -> list:
    """The surveyed flight route for a scene, at ``altitude``.

    Each waypoint's yaw points along the leg that reaches it, so the onboard
    camera looks where the drone is going -- which is what makes the recording
    useful as navigation training data rather than a sequence of sideways
    drifts.

    Args:
        name: A key of :data:`SCENE_SURVEYS`.
        altitude: Flight altitude, metres.

    Returns:
        A list of world-frame ``(x, y, z, yaw)`` waypoints, yaw in radians CCW
        from +X (FLU, matching the repo-wide convention).

    Raises:
        KeyError: If the scene is unknown or has not been surveyed.
    """
    survey = _require_survey(name)
    points = [survey["spawn"]] + [tuple(p) for p in survey["route"]]

    waypoints = []
    for previous, (x, y) in zip(points, points[1:]):
        yaw = math.atan2(y - previous[1], x - previous[0])
        waypoints.append((x, y, altitude, yaw))
    return waypoints


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
