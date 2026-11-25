"""The method carries no hand-written knowledge about rooms and objects: the priors module is gone and nothing imports it."""
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOTS = (PACKAGE_ROOT / "sparx_agency" / "core", PACKAGE_ROOT / "sparx_agency" / "tasks")
REMOVED_MODULES = ("room_priors", "target_context")


def python_files():
    return [path for root in SOURCE_ROOTS for path in root.rglob("*.py") if "__pycache__" not in path.parts]


IMPORT_OF_REMOVED_MODULE = re.compile(r"^\s*(from|import)\s+[\w. ,]*\b(%s)\b" % "|".join(REMOVED_MODULES), re.MULTILINE)


def imports_removed_module(path):
    return IMPORT_OF_REMOVED_MODULE.search(path.read_text(errors="replace")) is not None


def test_the_scan_finds_the_source_files():
    assert len(python_files()) > 500


def test_the_priors_module_and_the_context_check_module_do_not_exist():
    existing = [path for path in python_files() if path.stem in REMOVED_MODULES and "tests" not in path.parts]
    assert existing == []


def test_nothing_imports_the_removed_modules():
    offenders = [str(path.relative_to(PACKAGE_ROOT)) for path in python_files() if imports_removed_module(path)]
    assert offenders == []
