"""Load mesh files to ``(vertices, faces)`` in metres, with a cache.

Two things make this more than a call to ``trimesh.load``.

**The COLLADA unit trap.** A ``.dae`` carries its own unit in
``<asset><unit meter="0.01"/></asset>``. Gazebo honours it; ``trimesh.load(...,
force="mesh")`` applies the file's Y_UP-to-Z_UP axis conversion but **not** that
scale. The hospital's wall mesh therefore loads spanning x[-1255.8, 1255.8] --
centimetres read as metres, a silent factor of 100. Nothing downstream can
detect it: the map simply comes out 100x too big, or, once the bounds are
computed from the same wrong geometry, correct-looking and wrong. So the unit
is read out of the XML here and applied by hand. ``.obj`` files carry no unit
and instead get an explicit ``<scale>`` in the SDF, which the caller passes in.

**The same chair sixteen times.** A furnished world instantiates a handful of
models over and over. Loading and parsing each one once, keyed by path *and*
scale, turns a minutes-long parse into a one-off.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh

COLLADA_SUFFIXES = (".dae",)
CACHE_SIZE = 512


def load_mesh(path, scale=(1.0, 1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Load a mesh in metres, with its file unit and SDF scale applied.

    Args:
        path: Mesh file. ``.dae``, ``.obj`` and anything else trimesh reads.
        scale: The SDF ``<scale>`` for this instance, applied in the mesh's own
            frame after the file's own unit.

    Returns:
        ``(vertices, faces)`` -- ``(V, 3)`` float64 in metres and ``(F, 3)``
        int32. Both are read-only, because they are shared with every other
        instance of the same model.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file holds no triangles.
    """
    factors = tuple(float(v) for v in scale)
    if len(factors) != 3:
        raise ValueError("scale must have 3 components, got %r" % (scale,))
    return _load_cached(str(Path(path)), factors)


@lru_cache(maxsize=CACHE_SIZE)
def _load_cached(path_text: str, scale: Tuple[float, float, float]):
    """Cached body of :func:`load_mesh`, keyed by path and scale."""
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError("no mesh at %s" % (path,))

    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.size == 0:
        raise ValueError("mesh %s has no triangles" % (path,))

    vertices = vertices * collada_unit_metres(path)
    vertices = vertices * np.asarray(scale, dtype=np.float64)
    vertices.flags.writeable = False
    faces.flags.writeable = False
    return vertices, faces


def collada_unit_metres(path) -> float:
    """Return the ``<asset><unit meter="...">`` of a COLLADA file.

    Args:
        path: Mesh file. Anything that is not COLLADA returns 1.0.

    Returns:
        Metres per file unit. 1.0 when the file is not COLLADA, or when it
        declares no ``<unit>`` at all -- which is COLLADA's own default.

    Raises:
        ValueError: If the file is COLLADA but its unit cannot be trusted:
            unparseable XML, a ``meter`` that is not a number, or one that is
            not positive.
    """
    path = Path(path)
    if path.suffix.lower() not in COLLADA_SUFFIXES:
        return 1.0
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as error:
        # Guessing 1.0 here *is* the 100x failure this module exists to
        # prevent: a centimetre mesh read as metres builds a map that looks
        # entirely plausible and is 100x too big, and nothing downstream can
        # tell. A .dae whose unit cannot be read is a .dae we cannot map.
        raise ValueError("cannot parse COLLADA %s: %s" % (path, error))
    for element in root.iter():
        if _local_name(element.tag) != "unit":
            continue
        raw = element.get("meter")
        if raw is None:
            # A <unit> that names no scale declares nothing; 1.0 is the
            # COLLADA default and the only silent answer that is safe.
            continue
        try:
            metres = float(raw)
        except ValueError:
            raise ValueError(
                "COLLADA %s declares meter=%r, which is not a number" % (path, raw)
            )
        if not metres > 0.0:
            raise ValueError(
                "COLLADA %s declares meter=%r, which is not a positive scale"
                % (path, raw)
            )
        return metres
    return 1.0


def cache_info():
    """Return the underlying LRU cache statistics, for the CLI summary."""
    return _load_cached.cache_info()


def clear_cache() -> None:
    """Drop every cached mesh. Used by tests and long-running tools."""
    _load_cached.cache_clear()


def _local_name(tag: str) -> str:
    """Strip an XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]
