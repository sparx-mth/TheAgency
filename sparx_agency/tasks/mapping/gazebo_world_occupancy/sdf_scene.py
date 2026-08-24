"""Flatten a Gazebo ``.world`` into world-placed geometry instances.

An SDF world is a tree: the world includes models, a model's ``model.sdf``
declares links, a link declares collisions, and every level may add its own
``<pose>``. Nothing downstream cares about that tree -- a rasteriser wants a
flat list of "this shape, at this 4x4 transform". This module walks the tree,
resolves every ``model://`` URI, composes the poses, and yields that flat list.

Three decisions worth stating:

* **Collision geometry wins.** A link's ``<collision>`` is what the simulator
  itself would stop a robot against, so it is what a ground-truth map should
  contain. ``<visual>`` is used only for links with no collision at all, where
  it is the only evidence the object exists.
* **Poses compose, they do not override.** ``include -> model -> link ->
  collision`` each contribute a transform. The one exception is SDF's own rule
  that a ``<pose>`` on an ``<include>`` replaces the included model's own
  ``<pose>`` rather than stacking with it.
* **What is left out is recorded, not swallowed.** An unresolved ``model://``,
  a model skipped by name, an unmappable ``<plane>``: each lands in a list on
  the returned scene so the caller can report it. A missing chair and a missing
  building look identical at this level, and only the human can tell them apart.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from sparx_agency.tasks.mapping.gazebo_world_occupancy.geometry_instance import (
    BOX_KIND,
    COLLISION_SOURCE,
    CYLINDER_KIND,
    GeometryInstance,
    IGNORED_KINDS,
    MESH_KIND,
    SPHERE_KIND,
    SUPPORTED_KINDS,
    SdfScene,
    VISUAL_SOURCE,
)
from sparx_agency.tasks.mapping.gazebo_world_occupancy.model_lookup import (
    ModelNotFoundError,
    resolve_mesh_uri,
    resolve_model_uri,
    model_sdf_path,
)
from sparx_agency.tasks.mapping.gazebo_world_occupancy.sdf_elements import (
    child_text,
    element_pose,
    pose_to_matrix,
)

__all__ = [
    "BOX_KIND",
    "CYLINDER_KIND",
    "GeometryInstance",
    "MESH_KIND",
    "ModelNotFoundError",
    "SPHERE_KIND",
    "SUPPORTED_KINDS",
    "SceneWalker",
    "SdfScene",
    "load_scene",
    "pose_to_matrix",
    "resolve_model_uri",
]


def load_scene(
    world_path,
    search_paths: Sequence[Path],
    skip_substrings: Iterable[str] = (),
) -> SdfScene:
    """Parse a world file into flattened, world-placed geometry.

    Args:
        world_path: Path to the ``.world`` (or any SDF file with a ``<world>``
            or a top-level ``<model>``).
        search_paths: Directories that ``model://`` URIs resolve against, in
            priority order.
        skip_substrings: Case-insensitive substrings; any model whose instance
            name or URI contains one is excluded. This is how the robot is
            kept out of its own map.

    Returns:
        The populated :class:`SdfScene`.

    Raises:
        FileNotFoundError: If ``world_path`` does not exist.
    """
    world_path = Path(world_path)
    if not world_path.exists():
        raise FileNotFoundError("no world file at %s" % (world_path,))

    scene = SdfScene(world_path=world_path)
    walker = SceneWalker(
        scene=scene,
        search_paths=[Path(p) for p in search_paths],
        skip_substrings=tuple(s.lower() for s in skip_substrings),
    )
    walker.walk_root(ET.parse(str(world_path)).getroot(), world_path.parent)
    return scene


class SceneWalker:
    """Recursive descent over the SDF tree, accumulating into an SdfScene."""

    def __init__(
        self,
        scene: SdfScene,
        search_paths: Sequence[Path],
        skip_substrings: Sequence[str],
    ) -> None:
        """Args are the scene to fill, the model path, and the skip list."""
        self._scene = scene
        self._search_paths = list(search_paths)
        self._skip = tuple(skip_substrings)

    def walk_root(self, root: ET.Element, base_dir: Path) -> None:
        """Walk a parsed SDF document's top level.

        Args:
            root: The ``<sdf>`` element, or a ``<world>``/``<model>`` element.
            base_dir: Directory relative mesh URIs resolve against.
        """
        for container in root.findall("world") or [root]:
            self._walk_container(container, np.eye(4), base_dir, "")

    def _walk_container(self, element, parent, base_dir, prefix) -> None:
        """Walk the ``<include>`` and ``<model>`` children of one element."""
        for include in element.findall("include"):
            self._walk_include(include, parent, prefix)
        for model in element.findall("model"):
            self._walk_model(model, parent, base_dir, prefix)

    def _walk_include(self, include, parent, prefix) -> None:
        """Resolve one ``<include>`` and walk the model it names."""
        uri = child_text(include, "uri")
        name = child_text(include, "name") or uri.rsplit("/", 1)[-1]
        qualified = "%s%s" % (prefix, name)
        if self._is_skipped(qualified) or self._is_skipped(uri):
            self._scene.skipped_models.append(qualified)
            return
        try:
            model_dir = resolve_model_uri(uri, self._search_paths)
        except ModelNotFoundError:
            self._scene.missing_models.append(uri)
            return

        sdf_path = model_sdf_path(model_dir)
        if sdf_path is None:
            self._scene.missing_models.append("%s (no model.sdf)" % (uri,))
            return
        root = ET.parse(str(sdf_path)).getroot()
        # SDF's own rule: a pose on the include replaces the model's own pose.
        override = element_pose(include) if include.find("pose") is not None else None
        for model in root.findall("model") or [root]:
            self._walk_model(model, parent, model_dir, prefix, override, qualified)

    def _walk_model(
        self, model, parent, base_dir, prefix, override_pose=None, name=None
    ) -> None:
        """Walk one ``<model>`` element: its links, then any nested models."""
        own_name = name or "%s%s" % (prefix, model.get("name", "model"))
        if self._is_skipped(own_name):
            self._scene.skipped_models.append(own_name)
            return
        local = override_pose if override_pose is not None else element_pose(model)
        world = parent.dot(local)
        for link in model.findall("link"):
            self._walk_link(link, world, base_dir, own_name)
        self._walk_container(model, world, base_dir, own_name + "::")

    def _walk_link(self, link, model_world, base_dir, model_name) -> None:
        """Emit one link's collisions, or its visuals when it has none."""
        link_world = model_world.dot(element_pose(link))
        link_name = link.get("name", "link")
        elements = link.findall("collision")
        source = COLLISION_SOURCE
        if not elements:
            elements = link.findall("visual")
            source = VISUAL_SOURCE
        for element in elements:
            self._emit(element, link_world, base_dir, model_name, link_name, source)

    def _emit(self, element, link_world, base_dir, model_name, link_name, source):
        """Turn one ``<collision>``/``<visual>`` into GeometryInstances."""
        geometry = element.find("geometry")
        if geometry is None:
            return
        world = link_world.dot(element_pose(element))
        for shape in geometry:
            instance = self._build(
                shape, world, base_dir, model_name, link_name, source
            )
            if instance is not None:
                self._scene.instances.append(instance)

    def _build(self, shape, world, base_dir, model_name, link_name, source):
        """Build the instance for one geometry shape element, or None."""
        kind = shape.tag
        if kind in IGNORED_KINDS or kind not in SUPPORTED_KINDS:
            self._scene.ignored_geometry.append((model_name, kind))
            return None
        common = dict(
            model_name=model_name,
            link_name=link_name,
            source=source,
            transform=world,
            scale=np.ones(3, dtype=np.float64),
        )
        if kind == MESH_KIND:
            return self._build_mesh(shape, base_dir, common)
        if kind == BOX_KIND:
            size = np.array([float(v) for v in child_text(shape, "size").split()])
            return GeometryInstance(kind=BOX_KIND, size=size, **common)
        if kind == CYLINDER_KIND:
            return GeometryInstance(
                kind=CYLINDER_KIND,
                radius=float(child_text(shape, "radius", "0")),
                length=float(child_text(shape, "length", "0")),
                **common
            )
        return GeometryInstance(
            kind=SPHERE_KIND, radius=float(child_text(shape, "radius", "0")), **common
        )

    def _build_mesh(self, shape, base_dir, common) -> Optional[GeometryInstance]:
        """Resolve a ``<mesh>``'s URI and scale."""
        uri = child_text(shape, "uri")
        scale_text = child_text(shape, "scale")
        if scale_text:
            common["scale"] = np.array([float(v) for v in scale_text.split()])
        try:
            path = resolve_mesh_uri(uri, self._search_paths, base_dir)
        except ModelNotFoundError:
            # Kept apart from missing_models: an unresolved include may be
            # Gazebo's sun, but an unresolved *mesh file* is a link that
            # exists, declares a shape, and has nothing to draw -- a wall
            # silently absent from a map that still looks complete.
            self._scene.missing_meshes.append(uri)
            return None
        return GeometryInstance(kind=MESH_KIND, mesh_path=path, **common)

    def _is_skipped(self, name: str) -> bool:
        """True when a model name matches any skip substring."""
        lowered = name.lower()
        return any(token in lowered for token in self._skip)
