"""The method stays in ``core``: no simulator, no dataset, no service transport, no ``tasks`` import.

``core/planning/object_search`` is the algorithm; everything that knows a
benchmark -- Habitat, the Gibson files, the HTTP services -- lives under
``tasks/planning/objnav_benchmark_runtime``. The rule is enforced by scanning
the source with :mod:`ast`: a lazy import inside a function counts too, since
it still ties the method to the adapter, only later and further from the
cause. The modules extracted into ``core/mapping`` for this method must in
turn import nothing from ``core/planning`` -- mapping sits below planning in
the repository's layering, and today no ``core/mapping`` module reaches up.

Standard library only.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

#: .../core/planning/object_search  (this file is <object_search>/tests/<name>.py)
PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
#: .../sparx_agency
SPARX_ROOT = PACKAGE_DIR.parents[2]

#: Top-level packages the method may never import, even lazily.
FORBIDDEN_ROOTS = frozenset({
    "habitat", "habitat_sim", "magnum", "quaternion",   # the simulator and its maths
    "requests", "urllib3", "http", "socket",            # service transport is the clients' business
    "rclpy", "rospy", "ros2",                            # ROS lives at the edges
    "torch", "tensorrt", "pycuda", "ultralytics",        # weights never run inside the method
    "skfmm", "skimage",                                  # the Gibson scorer's stack
})

#: ``sparx_agency`` prefixes the method may never import.
FORBIDDEN_SPARX_PREFIXES = ("sparx_agency.tasks.", "sparx_agency.robots.", "sparx_agency.demos.", "sparx_agency.tools.")

#: The modules this method placed in ``core/mapping``; each must stay below ``core/planning``.
MAPPING_EXTRACTIONS = (
    "core/mapping/objects/box_occluder.py",
    "core/mapping/objects/box_extent.py",
    "core/mapping/objects/detection_dedup.py",
    "core/mapping/objects/target_evidence.py",
    "core/mapping/objects/support_surface.py",
    "core/mapping/tracking/target_memory_3d.py",
    "core/mapping/topology/room_priors.py",
    "core/mapping/topology/revisable_room_labels.py",
)

#: The scan must find at least this many modules; fewer means it is looking in the wrong place.
MIN_MODULES = 50


def method_modules():
    """Every library module of the package (tests excluded), sorted."""
    return sorted(p for p in PACKAGE_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts and "tests" not in p.relative_to(PACKAGE_DIR).parts)


def imported_names(tree):
    """Every dotted module name an import statement in ``tree`` pulls, with its line."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.lineno


def where(path, line=None):
    name = str(path.relative_to(SPARX_ROOT))
    return name if line is None else "%s:%d" % (name, line)


def test_the_scan_finds_the_whole_package():
    files = method_modules()
    names = {where(p) for p in files}
    assert len(files) >= MIN_MODULES, sorted(names)
    for expected in ("core/planning/object_search/policy.py", "core/planning/object_search/interfaces.py",
                     "core/planning/object_search/closing/target_closing.py",
                     "core/planning/object_search/rooms/room_search_loop.py"):
        assert expected in names, "the scan missed %s" % expected


def test_the_method_imports_no_adapter_simulator_or_transport():
    offenders = []
    for path in method_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for name, line in imported_names(tree):
            root = name.split(".")[0]
            if root in FORBIDDEN_ROOTS or name.startswith(FORBIDDEN_SPARX_PREFIXES):
                offenders.append("%s imports %r" % (where(path, line), name))
    assert not offenders, ("the method must stay simulator-, dataset- and transport-free; "
                           "put the adapter under tasks/planning/objnav_benchmark_runtime:\n  " + "\n  ".join(offenders))


def test_every_module_has_a_docstring():
    missing = [where(p) for p in method_modules()
               if ast.get_docstring(ast.parse(p.read_text(encoding="utf-8"), str(p))) is None]
    assert not missing, "modules without a docstring: %r" % missing


@pytest.mark.parametrize("relative", MAPPING_EXTRACTIONS)
def test_the_mapping_extractions_stay_below_planning(relative):
    path = SPARX_ROOT / relative
    assert path.is_file(), "%s was moved or renamed without updating this list" % relative
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    upward = ["%s imports %r" % (where(path, line), name) for name, line in imported_names(tree)
              if name.startswith("sparx_agency.core.planning") or name.startswith(FORBIDDEN_SPARX_PREFIXES)]
    assert not upward, "core/mapping must not import core/planning (or tasks):\n  " + "\n  ".join(upward)


def test_the_package_root_imports_nothing_heavy_at_import_time():
    """``import sparx_agency.core.planning.object_search`` is a docstring: the policy is imported by module path."""
    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    assert not list(imported_names(tree)), "the package root must stay import-free"
