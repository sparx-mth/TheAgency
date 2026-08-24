"""Find the world file and the directories its ``model://`` URIs resolve against.

Gazebo resolves ``model://NAME`` by scanning a list of directories in order.
Reproducing that here needs no hardcoded machine paths: a world's model
directories sit beside it (``worlds/`` next to ``models/`` and ``fuel_models/``
is the aws-robomaker layout), and Gazebo's own ``GAZEBO_MODEL_PATH`` covers the
rest. Explicit ``--search-path`` flags come first so an override always wins.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

WORLD_SUFFIX = ".world"
SIBLING_DIRS = ("models", "fuel_models", "meshes")
MODEL_PATH_ENV = "GAZEBO_MODEL_PATH"
RESOURCE_PATH_ENV = "GAZEBO_RESOURCE_PATH"


def search_paths_for(world_path: Path, extra: Sequence[Path] = ()) -> List[Path]:
    """Build the model search path for a world, in priority order.

    Args:
        world_path: The world file. Its parent and grandparent are scanned for
            the conventional sibling model directories.
        extra: Explicitly requested directories, searched first.

    Returns:
        Existing directories, de-duplicated, highest priority first.
    """
    ordered = [Path(p).expanduser().resolve() for p in extra]
    world_path = Path(world_path).expanduser().resolve()
    for root in (world_path.parent, world_path.parent.parent):
        for name in SIBLING_DIRS:
            ordered.append(root / name)
    for entry in os.environ.get(MODEL_PATH_ENV, "").split(os.pathsep):
        if entry.strip():
            ordered.append(Path(entry.strip()).expanduser())
    return _existing_unique(ordered)


def resolve_world(spec: str, extra: Sequence[Path] = ()) -> Path:
    """Resolve a world given as a path or as a bare name.

    Args:
        spec: A path to a ``.world`` file, or a bare name such as
            ``hospital``.
        extra: Directories to search for a bare name, alongside
            ``GAZEBO_RESOURCE_PATH``. Each is checked itself and for a
            ``worlds/`` subdirectory, at that level and one above.

    Returns:
        The resolved world file.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    direct = Path(spec).expanduser()
    if direct.is_file():
        return direct.resolve()

    name = direct.name
    if not name.endswith(WORLD_SUFFIX):
        name += WORLD_SUFFIX

    roots: List[Path] = []
    for entry in extra:
        base = Path(entry).expanduser()
        roots.extend([base, base / "worlds", base.parent, base.parent / "worlds"])
    for entry in os.environ.get(RESOURCE_PATH_ENV, "").split(os.pathsep):
        if entry.strip():
            base = Path(entry.strip()).expanduser()
            roots.extend([base, base / "worlds"])

    for root in _existing_unique(roots):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "no world %r; pass a path, or a --search-path whose tree holds it" % (spec,)
    )


def _existing_unique(paths: Sequence[Path]) -> List[Path]:
    """Keep existing directories, in order, without repeats."""
    seen = set()
    kept = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        kept.append(resolved)
    return kept
