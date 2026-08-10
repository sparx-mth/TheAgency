"""The two hard rules of ``core/planning/vlas``, as executable checks.

The FALCON ROS1 adapter imports this package inside a Noetic container that has
**Python 3.8**, numpy, and essentially nothing else -- no torch, no TensorRT, no
pycuda, and (on the Jetson build, which has no pip) only APT's requests/Pillow.
The whole XTEND flight stack goes down at node startup if either rule breaks, and
it breaks at *import* time on the drone, not in CI.

Both rules used to live only in module docstrings. They live here now.

Rule 1 -- no eager heavy imports. Importing any policy's public surface must not
pull ``torch`` / ``tensorrt`` / ``pycuda`` / ``cv2`` / ``requests`` / ``PIL``.
Rule 2 -- Python 3.8 syntax. No ``match``/``case``, no PEP 604 ``X | Y`` in a
module without ``from __future__ import annotations``, no ``dataclass(slots=)``.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

#: .../core/planning/vlas  (this file is <vlas>/common/tests/<name>.py)
VLAS_DIR = pathlib.Path(__file__).resolve().parents[2]
#: the directory that holds `sparx_agency/` -- what PYTHONPATH must point at.
REPO_ROOT = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]

#: Imported by the five FALCON ROS1 nodes; see tasks/planning/falcon/adapter/scripts.
FALCON_NAVDP_SYMBOLS = (
    "NavDPError", "NavDPPointgoalClient", "anchor_trajectory_to_world",
    "pixel_to_pointgoal", "point_to_pointgoal", "project_trajectory_to_pixels",
    "world_to_body_2d", "select_farthest_visible_waypoint",
    "NAVDP_MAX_FWD_M", "NAVDP_MAX_LAT_M",
)

HEAVY = ("torch", "tensorrt", "pycuda", "cv2", "requests", "PIL", "scipy")
"""What the Noetic container does not have. ``scipy`` is the quiet one: it is
installed on every developer machine, so a module-scope ``from scipy...``
anywhere on a FALCON import path passes every local test and kills the node at
start-up. Several packages under ``core/planning`` use scipy legitimately --
only the ones reachable from here must not."""


def _py_files():
    return sorted(p for p in VLAS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


# ── Rule 1: no eager heavy imports ───────────────────────────────────────
@pytest.mark.parametrize("module", [
    "sparx_agency.core.planning.vlas",
    "sparx_agency.core.planning.vlas.navdp",
    "sparx_agency.core.planning.vlas.navdp.trt",
    "sparx_agency.core.planning.vlas.flownav",
    "sparx_agency.core.planning.vlas.flownav.client",
    "sparx_agency.core.planning.vlas.flownav.trt",
    "sparx_agency.core.planning.vlas.common.trt",
    # The only module under vlas/ that imports from another core/planning
    # package (trackers, for the pure-pursuit lookahead), so it is the one whose
    # import chain can grow a heavy dependency without anyone here touching a
    # file. navdp_click_node imports it inside the Noetic container.
    "sparx_agency.core.planning.vlas.common.plan_commit",
])
def test_import_pulls_no_heavy_dependency(module):
    # Run in a FRESH interpreter: an in-process check passes trivially once any
    # earlier test has already imported numpy/PIL into sys.modules.
    code = (
        "import sys, importlib\n"
        "importlib.import_module(%r)\n"
        "bad = sorted({m.split('.')[0] for m in sys.modules} & set(%r))\n"
        "print(','.join(bad))\n" % (module, HEAVY)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO_ROOT))
    assert out.returncode == 0, "importing %s failed:\n%s" % (module, out.stderr)
    leaked = [x for x in out.stdout.strip().split(",") if x]
    assert not leaked, (
        "%s eagerly imported %r. These must be lazy (inside methods): the FALCON "
        "Noetic container has none of them." % (module, leaked))


def test_falcon_navdp_symbols_are_all_importable_from_the_package_root():
    # The five FALCON nodes do `from ...vlas.navdp import (...)`. Any consolidation
    # must keep every one of these re-exported from that single module path.
    import sparx_agency.core.planning.vlas.navdp as navdp
    missing = [s for s in FALCON_NAVDP_SYMBOLS if not hasattr(navdp, s)]
    assert not missing, "FALCON imports these from core.planning.vlas.navdp: %r" % missing


def test_navdp_error_is_one_class_across_client_and_trt():
    # These used to be two distinct same-named classes, so `except NavDPError`
    # imported from one half silently missed the other's.
    from sparx_agency.core.planning.vlas.navdp import NavDPError as from_pkg
    from sparx_agency.core.planning.vlas.navdp.errors import NavDPError as from_root
    from sparx_agency.core.planning.vlas.navdp.trt.errors import NavDPError as from_trt
    assert from_pkg is from_root is from_trt


# ── Rule 2: Python 3.8 syntax ────────────────────────────────────────────
def test_every_module_parses_under_python_38():
    # feature_version rejects syntax newer than 3.8 (match/case, etc.) even though
    # this interpreter would happily accept it.
    bad = []
    for path in _py_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
                      feature_version=(3, 8))
        except SyntaxError as e:
            bad.append("%s: %s" % (path.relative_to(VLAS_DIR), e))
    assert not bad, "not Python 3.8 parseable:\n  " + "\n  ".join(bad)


def _has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names)
               for n in tree.body)


def _pep604_in_annotation(node):
    """True if an annotation expression contains a PEP 604 ``X | Y`` union."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return True
    return False


def test_no_pep604_unions_without_future_annotations():
    # `X | Y` parses fine on 3.8 but raises TypeError when evaluated, which for an
    # annotation means at *import*. `from __future__ import annotations` defers it.
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _has_future_annotations(tree):
            continue
        for node in ast.walk(tree):
            annotations = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = [a.annotation for a in
                               node.args.args + node.args.kwonlyargs + node.args.posonlyargs
                               if a.annotation]
                if node.returns:
                    annotations.append(node.returns)
            elif isinstance(node, ast.AnnAssign) and node.annotation:
                annotations = [node.annotation]
            for ann in annotations:
                if _pep604_in_annotation(ann):
                    offenders.append("%s:%d" % (path.relative_to(VLAS_DIR), node.lineno))
    assert not offenders, (
        "PEP 604 unions without `from __future__ import annotations` "
        "(TypeError on Python 3.8 at import): %r" % offenders)


def test_no_dataclass_slots():
    # `@dataclass(slots=True)` is 3.10+; on 3.8 it is a TypeError at class creation.
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and any(k.arg == "slots" for k in dec.keywords):
                    offenders.append("%s:%d" % (path.relative_to(VLAS_DIR), node.lineno))
    assert not offenders, "dataclass(slots=...) is 3.10+: %r" % offenders


#: Builtins that grew ``__class_getitem__`` only in 3.9 (PEP 585).
BUILTIN_GENERICS = ("list", "dict", "tuple", "set", "frozenset", "type")


def _annotation_node_ids(tree):
    """id() of every AST node that sits inside a type annotation.

    Only meaningful for a module that defers its annotations: there the
    expression is stored as a string and never evaluated, so a builtin generic
    inside one costs nothing on 3.8. The same expression anywhere else -- and
    every annotation in a module *without* the future import -- is executed.
    """
    ids = set()
    for node in ast.walk(tree):
        annotations = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            slots = (args.args + args.kwonlyargs + args.posonlyargs
                     + [args.vararg, args.kwarg])
            annotations = [a.annotation for a in slots if a is not None and a.annotation]
            if node.returns:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign) and node.annotation:
            annotations = [node.annotation]
        for ann in annotations:
            for sub in ast.walk(ann):
                ids.add(id(sub))
    return ids


def test_no_runtime_builtin_generics():
    """Builtin generics may only appear where the future import defers them.

    ``list[int]`` / ``dict[str, Any]`` / ``tuple[float, float]`` are 3.9+ (PEP
    585): on the container's 3.8 they raise ``TypeError`` the moment the
    expression is evaluated. A module-scope type alias, a ``cast(list[int], x)``
    or any signature in a module without ``from __future__ import annotations``
    evaluates at *import*, so the node dies during roslaunch and the XTEND never
    gets that planner -- while every test here on 3.12 stays green. The future
    import defers annotations only; it does not rescue a runtime expression.
    """
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        deferred = _annotation_node_ids(tree) if _has_future_annotations(tree) else set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in BUILTIN_GENERICS and id(node) not in deferred):
                offenders.append("%s:%d %s[...]"
                                 % (path.relative_to(VLAS_DIR), node.lineno, node.value.id))
    assert not offenders, (
        "builtin generic evaluated at runtime (TypeError on Python 3.8) -- use "
        "typing.List/Dict/Tuple, or move it into an annotation in a module with "
        "`from __future__ import annotations`: %r" % offenders)
