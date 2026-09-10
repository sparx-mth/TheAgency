"""The benchmark harness parses and runs as Python 3.8, and the scan proves it.

The harness runs under the Habitat conda env's Python 3.9, beside a ``core/``
that must import in the FALCON Noetic container's 3.8, while the venv here is
3.12 -- so ``list[int]``, ``X | Y`` and ``match`` all parse and every test
stays green, and then a module dies at import in the one environment a
benchmark actually runs in. So the rule is enforced by scanning the source
with :mod:`ast`, the only check that does not depend on which interpreter
happens to run it.

The scan covers every module under ``tasks/planning/objnav_benchmark``
(tests included) and ``tasks/planning/falcon_pegasus/stub/voxel_camera.py``,
which the fake environment's renderer imports. It walks the directory rather
than a list, so a module added later is covered without anyone editing this
file.

The detectors are a deliberate copy of those in
``core/planning/objnav/tests/test_python38_contract.py`` (themselves copied
from the rpt_star and vlas contract tests). Test modules are not a library:
importing another package's tests would couple the two suites, and a change
to theirs would silently change what this one checks.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest

#: .../tasks/planning/objnav_benchmark  (this file is <harness>/tests/<name>.py)
HARNESS_DIR = pathlib.Path(__file__).resolve().parents[1]
#: .../sparx_agency
PACKAGE_ROOT = HARNESS_DIR.parents[2]
#: The one module outside the harness it imports that is not core's.
VOXEL_CAMERA = (PACKAGE_ROOT / "tasks" / "planning" / "falcon_pegasus" / "stub"
                / "voxel_camera.py")

#: The scan must find at least this many modules. It counts what existed when
#: this file was written; the harness only grows, so a lower count means the
#: scan is looking in the wrong place, not that the harness shrank.
MIN_MODULES = 58

#: Standard-library modules a module here may import. Deliberately explicit:
#: every one exists in Python 3.8, and a newer one (``zoneinfo``, ``graphlib``,
#: ``tomllib``) would parse fine and fail to import in the Habitat env.
ALLOWED_STDLIB = frozenset({
    "__future__", "argparse", "ast", "collections", "dataclasses", "datetime",
    "fcntl", "fractions", "heapq", "itertools", "json", "math", "numbers",
    "os", "pathlib", "platform", "random", "re", "statistics", "subprocess",
    "sys", "time", "types", "typing", "zlib",
})

#: The only third-party package the harness may import.
ALLOWED_THIRD_PARTY = frozenset({"numpy"})

#: Third-party packages only a test may import.
ALLOWED_IN_TESTS = frozenset({"pytest"})

#: Stdlib modules that arrived after Python 3.8.
NEWER_THAN_38 = frozenset({"graphlib", "tomllib", "zoneinfo"})

#: Builtins that grew ``__class_getitem__`` only in 3.9 (PEP 585).
BUILTIN_GENERICS = ("list", "dict", "tuple", "set", "frozenset", "type")

#: Builtin names that are types on either side of a ``|``.
BUILTIN_TYPES = frozenset(BUILTIN_GENERICS + (
    "bool", "bytes", "complex", "float", "int", "object", "str"))


def source_files():
    """Every module in the scan, tests included, sorted."""
    files = [p for p in HARNESS_DIR.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files + [VOXEL_CAMERA])


def parsed():
    """Each module as ``(path, tree)``, parsed by the running interpreter."""
    for path in source_files():
        yield path, ast.parse(path.read_text(encoding="utf-8"), str(path))


def where(path, node=None) -> str:
    """``path:line`` relative to the package root, for failure messages."""
    name = str(path.relative_to(PACKAGE_ROOT))
    return name if node is None else "%s:%d" % (name, node.lineno)


def is_test_module(path) -> bool:
    """Whether a file is test code, which may also import pytest."""
    return "tests" in path.relative_to(PACKAGE_ROOT).parts


def _has_future_annotations(tree) -> bool:
    return any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names)
               for node in tree.body)


def _annotations(node):
    """The annotation expressions a function or annotated assignment carries."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        slots = (args.posonlyargs + args.args + args.kwonlyargs
                 + [args.vararg, args.kwarg])
        found = [a.annotation for a in slots
                 if a is not None and a.annotation is not None]
        if node.returns is not None:
            found.append(node.returns)
        return found
    if isinstance(node, ast.AnnAssign) and node.annotation is not None:
        return [node.annotation]
    return []


def _annotation_node_ids(tree):
    """``id()`` of every AST node inside an annotation of ``tree``."""
    ids = set()
    for node in ast.walk(tree):
        for annotation in _annotations(node):
            ids.update(id(sub) for sub in ast.walk(annotation))
    return ids


def _is_type_name(name: str) -> bool:
    """A builtin type, or a CamelCase class -- not an ALL_CAPS constant."""
    if name in BUILTIN_TYPES:
        return True
    return name[:1].isupper() and (len(name) == 1 or not name.isupper())


def _looks_like_a_type(node) -> bool:
    """Whether one side of a ``|`` is a type rather than a number, mask or set."""
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return _is_type_name(node.id)
    if isinstance(node, ast.Attribute):
        return _is_type_name(node.attr)
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and base.id in BUILTIN_GENERICS:
            return True
        return _looks_like_a_type(base)
    return False


def _import_roots(node):
    """The top-level package names an import statement pulls, or None."""
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "").split(".")[0]]
    return None


# -- the scan itself ----------------------------------------------------------

def test_the_scan_finds_the_whole_harness():
    """A scan that looks in the wrong directory passes every check vacuously."""
    files = source_files()
    names = {where(path) for path in files}
    assert len(files) >= MIN_MODULES, (
        "the scan found only %d modules: %r" % (len(files), sorted(names)))
    for expected in ("tasks/planning/objnav_benchmark/runner.py",
                     "tasks/planning/objnav_benchmark/fake_env/env.py",
                     "tasks/planning/objnav_benchmark/tests/test_python38_contract.py",
                     "tasks/planning/falcon_pegasus/stub/voxel_camera.py"):
        assert expected in names, "the scan missed %s" % expected


def test_every_module_parses_under_python_38():
    """``feature_version`` rejects syntax newer than 3.8 that 3.12 accepts."""
    bad = []
    for path in source_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), str(path),
                      feature_version=(3, 8))
        except SyntaxError as exc:
            bad.append("%s: %s" % (where(path), exc))
    assert not bad, "not Python 3.8 parseable:\n  " + "\n  ".join(bad)


# -- syntax the 3.8 parser does not reject but 3.8 cannot run -----------------

def test_no_pep604_unions_anywhere():
    """``X | Y`` on types raises at import on 3.8 wherever it is evaluated."""
    offenders = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
                    and (_looks_like_a_type(node.left)
                         or _looks_like_a_type(node.right))):
                offenders.append(where(path, node))
    assert not offenders, (
        "PEP 604 unions, which do not run on Python 3.8 -- use "
        "typing.Optional / Union: %r" % offenders)


def test_the_union_scan_can_actually_fail():
    """Guard the guard: a detector that never fires proves nothing."""
    unions = ["x: Optional[int] | None", "x: int | str", "x: list[int] | None",
              "x: np.ndarray | None", "x: Tuple[int, int] | List[int]",
              "x: T | U", "x: EpisodeRecord | None"]
    for source in unions:
        union = ast.parse(source).body[0].annotation
        assert _looks_like_a_type(union.left) or _looks_like_a_type(union.right), (
            source)
    not_unions = ["visited | (1 << target)", "np.isnan(d) | np.isinf(d)",
                  "valid[0] | hit[0]", "mask | other",
                  "fcntl.LOCK_EX | fcntl.LOCK_NB", "ALLOWED_STDLIB | EXTRA_NAMES"]
    for source in not_unions:
        expr = ast.parse(source).body[0].value
        assert not _looks_like_a_type(expr.left), source
        assert not _looks_like_a_type(expr.right), source


def test_no_dataclass_slots():
    """``@dataclass(slots=True)`` is a TypeError at class creation on 3.8."""
    offenders = []
    for path, tree in parsed():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "dataclass" and any(k.arg == "slots" for k in node.keywords):
                offenders.append(where(path, node))
    assert not offenders, "dataclass(slots=...) needs 3.10: %r" % offenders


def test_no_walrus():
    """``:=`` is legal on 3.8 but not the house style of this code base."""
    offenders = [where(path, node) for path, tree in parsed()
                 for node in ast.walk(tree) if isinstance(node, ast.NamedExpr)]
    assert not offenders, "`:=` is not used here: %r" % offenders


def test_every_annotating_module_defers_its_annotations():
    """The future import turns an annotation 3.8 cannot evaluate into a string."""
    missing = []
    for path, tree in parsed():
        annotates = any(_annotations(node) for node in ast.walk(tree))
        if annotates and not _has_future_annotations(tree):
            missing.append(where(path))
    assert not missing, (
        "annotating modules without `from __future__ import annotations`: %r"
        % missing)


def test_no_runtime_builtin_generics():
    """``list[int]`` outside a deferred annotation is a TypeError on 3.8."""
    offenders = []
    for path, tree in parsed():
        deferred = (_annotation_node_ids(tree) if _has_future_annotations(tree)
                    else set())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in BUILTIN_GENERICS
                    and id(node) not in deferred):
                offenders.append("%s %s[...]" % (where(path, node), node.value.id))
    assert not offenders, (
        "builtin generics evaluated at runtime -- use typing.List / Dict / "
        "Tuple: %r" % offenders)


def test_the_builtin_generic_scan_can_actually_fail():
    """Guard the guard: a runtime alias is caught, a deferred annotation is not."""
    alias = ast.parse("from __future__ import annotations\nPair = tuple[int, int]\n")
    deferred = ast.parse("from __future__ import annotations\n"
                         "def f(x: list[int]) -> dict[str, int]: pass\n")
    assert not _annotation_node_ids(alias)
    ids = _annotation_node_ids(deferred)
    subscripts = [n for n in ast.walk(deferred) if isinstance(n, ast.Subscript)]
    assert len(subscripts) == 2 and all(id(n) in ids for n in subscripts)


# -- imports -----------------------------------------------------------------

def test_every_import_is_absolute():
    """A relative import breaks when a module is run by path or vendored elsewhere."""
    offenders = [where(path, node) for path, tree in parsed()
                 for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.level]
    assert not offenders, (
        "relative imports; write `from sparx_agency.... import ...`: %r"
        % offenders)


def test_imports_stay_on_the_allow_list():
    """Only listed stdlib modules, numpy, sparx_agency -- and pytest in tests.

    Function-level imports count too: a lazy import of scipy still fails in
    the Habitat env, just later and further from the cause.
    """
    offenders = []
    for path, tree in parsed():
        allowed = ALLOWED_STDLIB | ALLOWED_THIRD_PARTY | {"sparx_agency"}
        if is_test_module(path):
            allowed = allowed | ALLOWED_IN_TESTS
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                continue  # reported by the absolute-import test
            for root in _import_roots(node) or ():
                if root not in allowed:
                    offenders.append("%s imports %r" % (where(path, node), root))
    assert not offenders, (
        "imports outside the allow-list; the harness must import under "
        "Python 3.8 with the standard library and numpy only: %r" % offenders)


def test_the_stdlib_allow_list_holds_only_38_standard_library():
    """Guard the guard: the allow-list must not smuggle in a newer or foreign module."""
    assert not (ALLOWED_STDLIB & NEWER_THAN_38)
    known = getattr(sys, "stdlib_module_names", None)
    if known is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    assert ALLOWED_STDLIB <= known, sorted(ALLOWED_STDLIB - known)
