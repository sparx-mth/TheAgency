"""The flat vocabulary a world is reduced to: placed shapes, and what was skipped.

Nothing downstream of the parser cares about SDF's tree of models, links and
collisions. It wants "this shape, at this 4x4 transform" -- which is what
:class:`GeometryInstance` is -- plus an honest account of what did *not* make
it into the list, which is what :class:`SdfScene` carries alongside.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

MESH_KIND = "mesh"
BOX_KIND = "box"
CYLINDER_KIND = "cylinder"
SPHERE_KIND = "sphere"
SUPPORTED_KINDS = (MESH_KIND, BOX_KIND, CYLINDER_KIND, SPHERE_KIND)
IGNORED_KINDS = ("plane", "heightmap", "polyline", "empty")

COLLISION_SOURCE = "collision"
VISUAL_SOURCE = "visual"


# eq=False, so the instance keeps object identity for == and hash(). The
# synthesised versions compare and hash a tuple holding ndarrays, and both
# blow up on contact: == returns an array whose truth value is ambiguous, and
# hash() rejects an unhashable ndarray. Frozen buys immutability here, not
# value semantics.
@dataclass(frozen=True, eq=False)
class GeometryInstance:
    """One placed shape: what it is, and where it is in the world.

    Attributes:
        model_name: Name of the world-level model instance it belongs to.
        link_name: Name of the link inside that model.
        source: :data:`COLLISION_SOURCE` or :data:`VISUAL_SOURCE`.
        kind: One of :data:`SUPPORTED_KINDS`.
        transform: ``(4, 4)`` world transform, applied after ``scale``.
        scale: ``(3,)`` mesh scale from the SDF, applied in the mesh's own
            frame. Always ``(1, 1, 1)`` for primitives.
        mesh_path: Resolved mesh file, for ``kind == "mesh"``.
        size: ``(3,)`` box extents, for ``kind == "box"``.
        radius: Radius, for cylinders and spheres.
        length: Axial length, for cylinders.
    """

    model_name: str
    link_name: str
    source: str
    kind: str
    transform: np.ndarray
    scale: np.ndarray
    mesh_path: Optional[Path] = None
    size: Optional[np.ndarray] = None
    radius: Optional[float] = None
    length: Optional[float] = None


@dataclass
class SdfScene:
    """Everything a rasteriser needs from one world file.

    Attributes:
        world_path: The ``.world`` that was parsed.
        instances: Flattened, world-placed geometry.
        missing_models: ``model://`` names that did not resolve. Gazebo's own
            built-ins (``model_lookup.GAZEBO_BUILTIN_MODELS``) land here on a
            machine without the Gazebo model database, which is harmless --
            but it is reported rather than swallowed, because a missing
            *building* would look exactly the same.
        missing_meshes: Mesh URIs that did not resolve, kept apart from
            ``missing_models`` because they are never harmless: a link exists,
            says it is shaped like that file, and the file is gone. There is
            no built-in that lands here, so this list is fatal on sight.
        skipped_models: Model names deliberately excluded, such as the robot.
        ignored_geometry: ``(model, kind)`` pairs skipped as unmappable, such
            as the infinite ground ``<plane>``.
    """

    world_path: Path
    instances: List[GeometryInstance] = field(default_factory=list)
    missing_models: List[str] = field(default_factory=list)
    missing_meshes: List[str] = field(default_factory=list)
    skipped_models: List[str] = field(default_factory=list)
    ignored_geometry: List[Tuple[str, str]] = field(default_factory=list)
