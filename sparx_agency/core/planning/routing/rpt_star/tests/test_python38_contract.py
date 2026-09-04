"""This package has to import inside the Noetic container, on Python 3.8.

``core/`` is imported by the FALCON adapters, which run under Python 3.8 with
numpy 1.17 and no scipy. Every violation of that is invisible here -- the venv
is 3.12, so ``list[int]`` and ``X | Y`` parse perfectly and the tests all pass
-- and then the node dies at import time on the aircraft, which is the worst
place to find out.

So the constraint is enforced by a test rather than by memory. This scans the
package's own source with the ast module, which is the only check that does not
depend on which interpreter happens to be running it.

The one thing a scan cannot catch is a numpy 2 API call, and the answer to that
here is simple: this package does not import numpy at all.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: Everything a module here is allowed to import from outside the package.
#: Deliberately short. If this list needs to grow, that is a decision, not a
#: detail -- the whole point is that the search needs nothing but the standard
#: library.
ALLOWED_THIRD_PARTY = frozenset()

#: Standard-library modules the package actually uses.
ALLOWED_STDLIB = frozenset({
    "__future__", "ast", "dataclasses", "heapq", "inspect", "itertools",
    "math", "pathlib", "random", "time", "typing",
})


def source_files():
    """Every module in the package, tests included."""
    return sorted(PACKAGE.rglob("*.py"))


def parsed():
    """Each module as ``(path, tree)``."""
    for path in source_files():
        yield path, ast.parse(path.read_text(encoding="utf-8"), str(path))


def test_there_are_modules_to_check():
    """Guard against the scan silently passing because it found nothing."""
    assert len(source_files()) >= 10


def _looks_like_a_type(node):
    """Whether one side of a ``|`` is a type rather than an integer.

    The package uses ``|`` for genuine bitmask arithmetic on the visited set,
    so the scan has to tell the two apart. A union always has ``None`` or a
    capitalised name on at least one side; a bitmask never does.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return node.id[:1].isupper()
    if isinstance(node, ast.Attribute):
        return node.attr[:1].isupper()
    if isinstance(node, ast.Subscript):
        return True
    return False


def test_no_pep604_unions():
    """``int | None`` is a syntax error on 3.8 in any evaluated position.

    ``from __future__ import annotations`` defers annotations only, so a type
    alias or a ``cast`` still evaluates the expression and still dies. Rather
    than trust that every union sits in an annotation, none are allowed.
    """
    for path, tree in parsed():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.BitOr)):
                continue
            if _looks_like_a_type(node.left) or _looks_like_a_type(node.right):
                pytest.fail(
                    "%s:%d looks like a PEP 604 union, which will not parse "
                    "on Python 3.8" % (path.name, node.lineno))


def test_the_union_scan_can_actually_fail():
    """Guard the guard: a scan that never fires proves nothing.

    Two of the package's own files use ``|`` for bitmask arithmetic, so if the
    detector were merely returning False everywhere the test above would pass
    for the wrong reason.
    """
    union = ast.parse("x: Optional | None").body[0].annotation
    bitmask = ast.parse("visited | (1 << target)").body[0].value
    assert _looks_like_a_type(union.left) or _looks_like_a_type(union.right)
    assert not _looks_like_a_type(bitmask.left)
    assert not _looks_like_a_type(bitmask.right)


def test_no_dataclass_slots():
    """``@dataclass(slots=True)`` arrived in 3.10."""
    for path, tree in parsed():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None)
            if name != "dataclass":
                continue
            for keyword in node.keywords:
                assert keyword.arg != "slots", (
                    "%s:%d uses dataclass(slots=True), which needs 3.10"
                    % (path.name, node.lineno))


def test_no_match_statement():
    """``match``/``case`` arrived in 3.10."""
    for path, tree in parsed():
        for node in ast.walk(tree):
            assert type(node).__name__ != "Match", (
                "%s:%d uses a match statement, which needs 3.10"
                % (path.name, node.lineno))


def test_no_walrus_in_this_package():
    """Legal on 3.8, but the repo's older code is not written in it.

    Not a compatibility requirement -- a house-style one, kept here because
    this is where the syntax scan already lives.
    """
    for path, tree in parsed():
        for node in ast.walk(tree):
            assert not isinstance(node, ast.NamedExpr), (
                "%s:%d uses `:=`" % (path.name, node.lineno))


def test_only_the_standard_library_is_imported():
    """No numpy, no scipy, nothing that a 3.8 container might not have."""
    for path, tree in parsed():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:                  # a relative import, ours
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root in ("sparx_agency", "pytest"):
                    continue
                assert root in ALLOWED_STDLIB or root in ALLOWED_THIRD_PARTY, (
                    "%s imports %r, which is neither the standard library nor "
                    "on the allow-list. core/ must import under Python 3.8 in "
                    "the Noetic container." % (path.name, root))


def test_every_annotating_module_defers_its_annotations():
    """``from __future__ import annotations`` wherever there are annotations.

    Cheap insurance: it turns an annotation that would not parse on 3.8 into a
    string rather than an error. A module that annotates nothing -- the package
    ``__init__`` is only re-exports -- needs nothing to defer.
    """
    for path, tree in parsed():
        annotates = any(
            isinstance(node, (ast.AnnAssign, ast.arg)) and node.annotation
            for node in ast.walk(tree))
        if not annotates:
            continue
        has_future = any(
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for node in tree.body)
        assert has_future, (
            "%s annotates but is missing `from __future__ import annotations`"
            % path.name)
