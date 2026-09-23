"""Resolve ``model://`` URIs and find a model directory's SDF file.

Gazebo resolves ``model://NAME/rest`` by trying ``<dir>/NAME/rest`` for each
directory on the model path, in order, and taking the first hit -- which is how
a locally edited model beats a downloaded Fuel copy of the same name. That
ordering is the whole contract, so it is kept explicit here rather than hidden
in a dict.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence

MODEL_SCHEME = "model://"
FILE_SCHEME = "file://"
GAZEBO_BUILTIN_MODELS = ("sun", "ground_plane")


class ModelNotFoundError(FileNotFoundError):
    """A ``model://`` URI did not resolve against any search path."""


def resolve_model_uri(uri: str, search_paths: Sequence[Path]) -> Path:
    """Resolve a ``model://NAME/rest`` URI against the model search path.

    Args:
        uri: The URI as written in the SDF.
        search_paths: Directories to search, in priority order.

    Returns:
        The resolved path.

    Raises:
        ValueError: If ``uri`` is not a ``model://`` URI.
        ModelNotFoundError: If no search path contains it.
    """
    if not uri.startswith(MODEL_SCHEME):
        raise ValueError("not a model:// uri: %r" % (uri,))
    relative = uri[len(MODEL_SCHEME):].strip("/")
    for root in search_paths:
        candidate = Path(root) / relative
        if candidate.exists():
            return candidate
    raise ModelNotFoundError(
        "%r not found under %s" % (uri, ", ".join(str(p) for p in search_paths))
    )


def is_gazebo_builtin(uri: str) -> bool:
    """True when a ``model://`` URI names one of Gazebo's own built-in models.

    Gazebo keeps ``sun`` and ``ground_plane`` in its own database rather than
    on ``GAZEBO_MODEL_PATH``, so on a machine that only has the world's models
    they never resolve -- and every world includes them. Neither holds
    mappable geometry (a light and an infinite plane), so this is the one
    unresolved include that is routine, and the only one ``--strict`` may pass
    over. Everything else it refuses, which is what makes it usable at all.

    Args:
        uri: The URI as written in the SDF.

    Returns:
        True for a built-in, False for anything else, including a URI that is
        not ``model://`` at all.
    """
    if not uri.startswith(MODEL_SCHEME):
        return False
    name = uri[len(MODEL_SCHEME):].strip("/").split("/")[0]
    return name in GAZEBO_BUILTIN_MODELS


def resolve_mesh_uri(uri: str, search_paths: Sequence[Path], base_dir: Path) -> Path:
    """Resolve a mesh URI, which may be ``model://``, ``file://`` or relative.

    Args:
        uri: The URI as written in the SDF.
        search_paths: Directories ``model://`` resolves against.
        base_dir: Directory a relative URI is taken from -- the model's own
            directory, which is how Gazebo reads them.

    Returns:
        The resolved mesh file.

    Raises:
        ModelNotFoundError: If the file does not exist.
    """
    if uri.startswith(MODEL_SCHEME):
        return resolve_model_uri(uri, search_paths)
    raw = uri[len(FILE_SCHEME):] if uri.startswith(FILE_SCHEME) else uri
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(base_dir) / raw
    if not candidate.exists():
        raise ModelNotFoundError("mesh %r not found (tried %s)" % (uri, candidate))
    return candidate


def model_sdf_path(model_dir: Path) -> Optional[Path]:
    """Find a model directory's SDF file, preferring what ``model.config`` names.

    Args:
        model_dir: The resolved model directory.

    Returns:
        The SDF file, or None when the directory holds none.
    """
    config = model_dir / "model.config"
    if config.exists():
        try:
            root = ET.parse(str(config)).getroot()
        except ET.ParseError:
            root = None
        if root is not None:
            for entry in root.findall("sdf"):
                named = (entry.text or "").strip()
                if named and (model_dir / named).exists():
                    return model_dir / named
    default = model_dir / "model.sdf"
    return default if default.exists() else None
