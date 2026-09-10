"""Importing any part of the ObjectNav layer pulls numpy and nothing heavier.

The layer is imported by the FALCON Noetic container (Python 3.8, numpy, no
scipy, no torch) and by every benchmark adapter, and it is imported at node
start-up -- so a heavy dependency does not fail a test, it kills a process on
the aircraft or in the Habitat env. The quiet ones are those installed on every
developer machine: a module-scope ``import yaml`` or ``from scipy ...`` passes
every local test. The repo's heavy packages are quieter still: importing
``core.common.spatial_math`` pulls yaml, and the planners pull OMPL, whose
bindings also corrupt the heap at interpreter exit.

Each module is imported in a FRESH interpreter: an in-process check passes
trivially once any earlier test has loaded the dependency into
``sys.modules``. The module list is the spec's, plus every module found on
disk, so a module added later is checked without anyone editing this file. A
static twin checks the source for module-scope imports of the heavy
``sparx_agency`` packages, which catches the mistake even where the heavy
package happens to import cleanly.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

#: .../core/planning/objnav  (this file is <objnav>/tests/<name>.py)
OBJNAV_DIR = pathlib.Path(__file__).resolve().parents[1]
#: The directory that holds ``sparx_agency/`` -- what the child imports from.
REPO_ROOT = OBJNAV_DIR.parents[3]
#: The package under test.
PACKAGE = "sparx_agency.core.planning.objnav"

#: What must never be in ``sys.modules`` after importing a module here.
HEAVY = ("torch", "tensorrt", "pycuda", "cv2", "requests", "PIL", "scipy",
         "yaml", "rclpy", "rospy", "ompl", "networkx", "skimage")

#: ``sparx_agency`` packages that pull OMPL, scipy, networkx, cv2, torch or
#: yaml. Only a lazy import, inside a function, is allowed.
HEAVY_SPARX = (
    "sparx_agency.core.planning.planners",
    "sparx_agency.core.planning.replanning",
    "sparx_agency.core.planning.exploration",
    "sparx_agency.core.mapping.topology",
    "sparx_agency.core.mapping.costmap",
    "sparx_agency.core.common.utils",
    "sparx_agency.core.common.spatial_math",
)

#: The modules the spec names. A module another package has not written yet
#: fails here until it exists.
SPEC_MODULES = tuple([PACKAGE] + [PACKAGE + suffix for suffix in (
    ".types", ".interfaces", ".errors", ".camera_geometry",
    ".action_converter", ".action_converter.converter",
    ".labels", ".labels.table_mapper", ".labels.registry",
    ".agent", ".agent.headless_agent",
)])

#: Outside the package, but imported by its target labels.
EXTRA_MODULES = ("sparx_agency.core.common.label_match",)

#: Keeps each child's BLAS from starting a thread per core just to import numpy.
SINGLE_THREADED = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1"}

#: The line the child prints its findings on.
MARKER = "LEAKED="


def library_files():
    """Every non-test module file under the package."""
    return sorted(p for p in OBJNAV_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts and "tests" not in p.parts)


def discovered_modules():
    """The dotted name of every non-test module on disk."""
    names = []
    for path in library_files():
        parts = path.relative_to(REPO_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


#: What the runtime check imports, one fresh interpreter each.
MODULES = tuple(sorted(set(SPEC_MODULES) | set(EXTRA_MODULES)
                       | set(discovered_modules())))


def probe(module, heavy=HEAVY):
    """Import ``module`` in a fresh interpreter and name what of ``heavy`` it loaded.

    Args:
        module: Dotted module name.
        heavy: Top-level package names to look for.

    Returns:
        ``(returncode, leaked names, stderr)``.
    """
    code = ("import importlib, sys\n"
            "importlib.import_module(%r)\n"
            "found = {name.split('.')[0] for name in sys.modules} & set(%r)\n"
            "print(%r + ','.join(sorted(found)))\n"
            % (module, tuple(heavy), MARKER))
    env = dict(os.environ, **SINGLE_THREADED)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(REPO_ROOT), env=env, timeout=120)
    lines = [l for l in result.stdout.splitlines() if l.startswith(MARKER)]
    leaked = [n for n in lines[-1][len(MARKER):].split(",") if n] if lines else []
    return result.returncode, leaked, result.stderr


def module_scope_imports(tree):
    """Import statements that run when the module is imported.

    Everything outside a function body: top level, and inside top-level
    ``if`` / ``try`` / ``with`` blocks and class bodies, which all execute at
    import. A function body runs only when called, so a lazy import there is
    the sanctioned way to use a heavy package.
    """
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            continue
        stack.extend(ast.iter_child_nodes(node))


def imported_names(node):
    """Every dotted name an import statement may bind or load."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [module] + ["%s.%s" % (module, alias.name) for alias in node.names]


def is_heavy_sparx(name) -> bool:
    """Whether a dotted name is one of :data:`HEAVY_SPARX` or inside one."""
    return any(name == heavy or name.startswith(heavy + ".")
               for heavy in HEAVY_SPARX)


# -- the runtime check --------------------------------------------------------

@pytest.mark.parametrize("module", MODULES)
def test_importing_the_module_pulls_no_heavy_dependency(module):
    """A heavy import here kills the FALCON node or the adapter at start-up."""
    returncode, leaked, stderr = probe(module)
    assert returncode == 0, "importing %s failed:\n%s" % (module, stderr)
    assert not leaked, (
        "%s imported %r. Import heavy packages inside the function that needs "
        "them: the Noetic container has none of them." % (module, leaked))


def test_the_module_list_covers_the_spec_and_the_disk():
    """A module missing from the list is a module nobody checks."""
    discovered = discovered_modules()
    assert len(discovered) >= 20, discovered
    assert PACKAGE + ".types.actions" in discovered
    assert set(SPEC_MODULES) | set(discovered) <= set(MODULES)


def test_the_probe_reports_a_module_it_is_told_to_look_for():
    """Guard the guard: a probe that always reports nothing proves nothing."""
    returncode, leaked, stderr = probe("json", heavy=("json",))
    assert returncode == 0, stderr
    assert leaked == ["json"]


def test_the_probe_runs_in_a_fresh_interpreter():
    """The parent has pytest loaded; a probe that inherited it would see it."""
    assert "pytest" in sys.modules
    returncode, leaked, stderr = probe(PACKAGE + ".errors", heavy=("pytest",))
    assert returncode == 0, stderr
    assert leaked == []


# -- the static twin ----------------------------------------------------------

def test_no_module_imports_a_heavy_sparx_package_at_module_scope():
    """OMPL, scipy and yaml arrive through these packages; only a lazy import may reach them."""
    offenders = []
    files = sorted(p for p in OBJNAV_DIR.rglob("*.py")
                   if "__pycache__" not in p.parts)
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in module_scope_imports(tree):
            for name in imported_names(node):
                if is_heavy_sparx(name):
                    offenders.append("%s:%d imports %s" % (
                        path.relative_to(OBJNAV_DIR), node.lineno, name))
    assert not offenders, (
        "module-scope imports of heavy sparx_agency packages; move them into "
        "the function that needs them: %r" % offenders)


def test_the_module_scope_scan_ignores_only_function_bodies():
    """Guard the guard: top-level, guarded and class-body imports count; lazy ones do not."""
    tree = ast.parse(
        "from sparx_agency.core.common import spatial_math\n"
        "try:\n    import sparx_agency.core.planning.planners.astar\n"
        "except ImportError:\n    pass\n"
        "class K:\n    from sparx_agency.core.common.utils import x\n"
        "def lazy():\n    from sparx_agency.core.mapping.costmap import y\n")
    names = [name for node in module_scope_imports(tree)
             for name in imported_names(node)]
    flagged = sorted(name for name in names if is_heavy_sparx(name))
    assert flagged == ["sparx_agency.core.common.spatial_math",
                       "sparx_agency.core.common.utils",
                       "sparx_agency.core.common.utils.x",
                       "sparx_agency.core.planning.planners.astar"]
    assert not is_heavy_sparx("sparx_agency.core.common.utilsx")
