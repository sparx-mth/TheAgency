"""No module in the harness shares its name with a standard-library module.

Running a script by path puts its directory first on ``sys.path``, and a
module there named like a standard-library one then shadows it for every
import in that process -- ``statistics.py``, as this package once had, broke
the ``from statistics import NormalDist`` inside it. The failure shows far
from its cause, so the names are checked here instead. The scan walks the
package, so a module added later is covered without anyone editing this file.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

#: .../tasks/planning/objnav_benchmark  (this file is <package>/tests/<name>.py)
PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]


def module_names():
    """Every module and subpackage name in the package, ``__init__`` and caches excluded."""
    names = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py":
            names.add(path.parent.name)
        else:
            names.add(path.stem)
    return names


def test_the_scan_finds_the_harness_modules_and_subpackages():
    """A scan that looks in the wrong directory passes the real check vacuously."""
    names = module_names()
    for expected in ("objnav_benchmark", "runner", "stats", "records",
                     "summaries", "checks", "fake_env", "world", "tests",
                     "test_module_names"):
        assert expected in names, "the scan missed %s" % expected


def test_no_module_shadows_a_standard_library_module():
    """Run by path, a module named like the standard library's hides it from every import in the process."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        pytest.skip("sys.stdlib_module_names needs Python 3.10")
    clashes = sorted(module_names() & set(stdlib))
    assert not clashes, (
        "each of these shadows the standard library's module of the same "
        "name for a script run by path; rename it: %r" % clashes)
