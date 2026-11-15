"""The ObjectNav layer parses and imports as Python 3.8, and the scan proves it.

``core/`` must import inside the FALCON Noetic container, which runs Python 3.8
with numpy and little else, and the benchmark adapters run it under the Habitat
conda env's Python 3.9. The venv here is 3.12, so ``list[int]``, ``X | Y`` and
``match`` all parse and every test stays green -- and then the module dies at
import time in the one place nobody is watching. So the rule is enforced by
scanning the source with :mod:`ast`, the only check that does not depend on
which interpreter happens to run it.

The scan covers every module under ``core/planning/objnav`` (tests included)
and ``core/common/label_match.py``, which the target labels import. It walks
the directory rather than a list, so a module another package adds later is
covered without anyone editing this file. Importing the package also runs
some twenty ``sparx_agency`` modules outside it (``core/common/types``,
``core/common/math``, ``core/mapping``), and any one of them can kill that
import on 3.8; so a fresh interpreter lists that closure, and its files get
the syntax checks too.

The PEP 604 detector and its guard-the-guard test are a deliberate copy of
``core/planning/routing/rpt_star/tests/test_python38_contract.py``, and the
builtin-generics check of
``core/planning/vlas/common/tests/test_core_import_contract.py``. Test modules
are not a library: importing another package's tests would couple the two
suites, and a change to theirs would silently change what this one checks. The
union detector is also tightened here -- this package does numpy mask
arithmetic, and a subscripted mask (``valid[0] | hit[0]``) is not a type.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

#: .../core/planning/objnav  (this file is <objnav>/tests/<name>.py)
OBJNAV_DIR = pathlib.Path(__file__).resolve().parents[1]
#: .../sparx_agency
PACKAGE_ROOT = OBJNAV_DIR.parents[2]
#: The one module outside the package the contract depends on for its strings.
LABEL_MATCH = PACKAGE_ROOT / "core" / "common" / "label_match.py"

#: The scan must find at least this many modules. It counts what existed when
#: this file was written; the package only grows, so a lower count means the
#: scan is looking in the wrong place, not that the package shrank.
MIN_MODULES = 44

#: Standard-library modules a module here may import. Deliberately explicit:
#: every one exists in Python 3.8, and a newer one (``zoneinfo``, ``graphlib``,
#: ``tomllib``) would parse fine and fail to import in the container.
ALLOWED_STDLIB = frozenset({
    "__future__", "abc", "ast", "collections", "copy", "dataclasses", "enum",
    "functools", "heapq", "importlib", "inspect", "itertools", "json", "math",
    "numbers", "operator", "os", "pathlib", "random", "re", "subprocess",
    "sys", "textwrap", "types", "typing", "warnings",
})

#: The only third-party package the library may import.
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
    files = [p for p in OBJNAV_DIR.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files + [LABEL_MATCH])


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
    """A builtin type, or a CamelCase class -- not an ALL_CAPS constant.

    Constants such as ``ALLOWED_STDLIB | EXTRA`` are set unions, not types; a
    single capital (``T``) is a type variable.
    """
    if name in BUILTIN_TYPES:
        return True
    return name[:1].isupper() and (len(name) == 1 or not name.isupper())


def _looks_like_a_type(node) -> bool:
    """Whether one side of a ``|`` is a type rather than a number, mask or set.

    A union always has ``None``, a type name, or a subscripted type on at
    least one side. Integer bitmasks, numpy boolean masks -- including a
    subscripted mask such as ``valid[0]`` -- and unions of ALL_CAPS constant
    sets never do.
    """
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


def union_offenders(path, tree):
    """``path:line`` of every ``|`` in ``tree`` with a type on either side."""
    return [where(path, node) for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            and (_looks_like_a_type(node.left)
                 or _looks_like_a_type(node.right))]


def runtime_generic_offenders(path, tree):
    """``path:line name[...]`` of every builtin generic outside a deferred annotation."""
    deferred = (_annotation_node_ids(tree) if _has_future_annotations(tree)
                else set())
    return ["%s %s[...]" % (where(path, node), node.value.id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id in BUILTIN_GENERICS and id(node) not in deferred]


# -- the scan itself ----------------------------------------------------------

def test_the_scan_finds_the_whole_package():
    """A scan that looks in the wrong directory passes every check vacuously."""
    files = source_files()
    names = {where(path) for path in files}
    assert len(files) >= MIN_MODULES, (
        "the scan found only %d modules: %r" % (len(files), sorted(names)))
    for expected in ("core/planning/objnav/errors.py",
                     "core/planning/objnav/types/actions.py",
                     "core/planning/objnav/tests/test_python38_contract.py",
                     "core/common/label_match.py"):
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
    """``X | Y`` on types raises at import on 3.8 wherever it is evaluated.

    The future import defers annotations only; a type alias or a ``cast``
    still evaluates the union. Rather than trust that every union sits in an
    annotation, none are allowed.
    """
    offenders = [offender for path, tree in parsed()
                 for offender in union_offenders(path, tree)]
    assert not offenders, (
        "PEP 604 unions, which do not run on Python 3.8 -- use "
        "typing.Optional / Union: %r" % offenders)


def test_the_union_scan_can_actually_fail():
    """Guard the guard: a detector that never fires proves nothing."""
    unions = ["x: Optional[int] | None", "x: int | str", "x: list[int] | None",
              "x: np.ndarray | None", "x: Tuple[int, int] | List[int]",
              "x: T | U", "x: AgentPose | Pose2D"]
    for source in unions:
        union = ast.parse(source).body[0].annotation
        assert _looks_like_a_type(union.left) or _looks_like_a_type(union.right), (
            source)
    not_unions = ["visited | (1 << target)", "np.isnan(d) | np.isinf(d)",
                  "valid[0] | hit[0]", "mask | other",
                  "ALLOWED_STDLIB | EXTRA_NAMES"]
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


def test_no_match_statement():
    """``match`` / ``case`` arrived in 3.10; named here so the message is plain."""
    offenders = [where(path, node) for path, tree in parsed()
                 for node in ast.walk(tree) if type(node).__name__ == "Match"]
    assert not offenders, "match statements need 3.10: %r" % offenders


def test_no_walrus():
    """``:=`` is legal on 3.8 but not the house style of this code base."""
    offenders = [where(path, node) for path, tree in parsed()
                 for node in ast.walk(tree) if isinstance(node, ast.NamedExpr)]
    assert not offenders, "`:=` is not used in core/: %r" % offenders


def test_every_annotating_module_defers_its_annotations():
    """The future import turns an annotation 3.8 cannot evaluate into a string.

    A module that annotates nothing -- a package ``__init__`` of re-exports --
    has nothing to defer.
    """
    missing = []
    for path, tree in parsed():
        annotates = any(_annotations(node) for node in ast.walk(tree))
        if annotates and not _has_future_annotations(tree):
            missing.append(where(path))
    assert not missing, (
        "annotating modules without `from __future__ import annotations`: %r"
        % missing)


def test_no_runtime_builtin_generics():
    """``list[int]`` outside a deferred annotation is a TypeError on 3.8.

    A module-scope alias or a ``cast(list[int], x)`` runs at import, so the
    module dies in the container while every test here on 3.12 passes.
    """
    offenders = [offender for path, tree in parsed()
                 for offender in runtime_generic_offenders(path, tree)]
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
    """A relative import breaks when a module is run or vendored elsewhere."""
    offenders = [where(path, node) for path, tree in parsed()
                 for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.level]
    assert not offenders, (
        "relative imports; write `from sparx_agency.... import ...`: %r"
        % offenders)


def test_imports_stay_on_the_allow_list():
    """Only listed stdlib modules, numpy, sparx_agency -- and pytest in tests.

    Function-level imports count too: a lazy import of scipy still fails in
    the container, just later and further from the cause.
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
        "imports outside the allow-list; core/ must import under Python 3.8 "
        "with numpy only: %r" % offenders)


def test_the_stdlib_allow_list_holds_only_38_standard_library():
    """Guard the guard: the allow-list must not smuggle in a newer or foreign module."""
    assert not (ALLOWED_STDLIB & NEWER_THAN_38)
    known = getattr(sys, "stdlib_module_names", None)
    if known is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    assert ALLOWED_STDLIB <= known, sorted(ALLOWED_STDLIB - known)


# -- the import closure --------------------------------------------------------

#: What a fresh interpreter runs: import the package, make the one call every
#: adapter makes, and print every sparx_agency source file then loaded.
CLOSURE_PROBE = (
    "import json, sys\n"
    "import sparx_agency.core.planning.objnav as objnav\n"
    "objnav.intrinsics_from_hfov(64, 48, 79.0)\n"
    "print(json.dumps(sorted(module.__file__\n"
    "    for name, module in list(sys.modules.items())\n"
    "    if name.split('.')[0] == 'sparx_agency'\n"
    "    and getattr(module, '__file__', None))))\n")


def closure_files():
    """Every ``sparx_agency`` source file the package's import and first call load."""
    repo_root = PACKAGE_ROOT.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(repo_root), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run([sys.executable, "-c", CLOSURE_PROBE],
                            capture_output=True, text=True, cwd=str(repo_root),
                            env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    return [pathlib.Path(name).resolve()
            for name in json.loads(result.stdout.splitlines()[-1])]


def test_the_import_closure_parses_and_runs_under_python_38():
    """Importing objnav runs twenty sparx modules the directory scan never reads; any one can kill the import on 3.8."""
    files = closure_files()
    elsewhere = [str(path) for path in files if PACKAGE_ROOT not in path.parents]
    assert not elsewhere, "the probe imported another checkout: %r" % elsewhere
    names = {where(path) for path in files}
    for expected in ("core/common/types/geometry.py", "core/common/math/se3.py",
                     "core/planning/objnav/camera_intrinsics.py"):
        assert expected in names, "the closure missed %s" % expected
    problems = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, str(path), feature_version=(3, 8))
        except SyntaxError as exc:
            problems.append("%s: %s" % (where(path), exc))
            continue
        tree = ast.parse(source, str(path))
        problems.extend("%s: PEP 604 union" % offender
                        for offender in union_offenders(path, tree))
        problems.extend("%s: runtime builtin generic" % offender
                        for offender in runtime_generic_offenders(path, tree))
    assert not problems, (
        "the objnav import closure breaks Python 3.8:\n  "
        + "\n  ".join(problems))
