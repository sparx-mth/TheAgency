"""Where a run's results go and which code made them: the layout and the revision every run records.

One layout for every run is what lets a results tree be browsed and compared,
and the revision in ``run.json`` is what says which code produced a number --
so neither may claim more than it knows.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import sparx_agency.tasks.planning.objnav_benchmark.results_io as results_io_module
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    DEFAULT_RESULTS_ROOT,
    UNKNOWN_REVISION,
    default_run_dir,
    git_revision,
    write_atomically,
)


# -- where results go ------------------------------------------------------------

def test_a_run_directory_is_root_then_benchmark_then_split_then_stamp(tmp_path):
    """One layout for every run is what lets a results tree be browsed and compared."""
    assert default_run_dir("hm3d_v2", "val", root=tmp_path,
                           stamp="20260910T120000Z") == (
        tmp_path / "hm3d_v2" / "val" / "20260910T120000Z")


def test_the_default_run_directory_is_outside_the_repository_and_stamped_in_utc():
    """Results written inside the checkout would dirty the revision they record."""
    path = default_run_dir("fake", "test")
    assert DEFAULT_RESULTS_ROOT == Path.home() / "objnav_benchmark"
    assert path.parent == DEFAULT_RESULTS_ROOT / "fake" / "test"
    assert re.fullmatch(r"\d{8}T\d{6}Z", path.name)


@pytest.mark.parametrize("name", ["", " ", "hm3d/v2", "..", "."])
def test_a_name_that_is_not_one_directory_level_is_refused(tmp_path, name):
    """``hm3d/v2`` or ``..`` would silently put the run somewhere the layout does not say."""
    with pytest.raises(HarnessError, match="one directory level"):
        default_run_dir(name, "val", root=tmp_path)


def test_an_atomic_write_replaces_the_whole_file_and_leaves_nothing_beside_it(tmp_path):
    """summary.json and run.json are replaced whole, so a crash leaves the old file or the new."""
    path = tmp_path / "run.json"
    path.write_text("old\n", encoding="utf-8")
    write_atomically(path, '{"n_episodes": 3}')
    assert path.read_text(encoding="utf-8") == '{"n_episodes": 3}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


# -- the git revision ----------------------------------------------------------

def test_git_revision_names_the_checkout_this_code_runs_from():
    """run.json must say which code produced the numbers."""
    assert re.fullmatch(r"[0-9a-f]{4,40}(-dirty)?|unknown", git_revision())


def test_git_revision_is_unknown_outside_a_repository(tmp_path, monkeypatch):
    """A directory in no repository has no revision; saying one would misattribute the run."""
    for name in ("GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_revision(tmp_path) == UNKNOWN_REVISION


def test_git_revision_is_unknown_when_git_is_not_installed(tmp_path, monkeypatch):
    """A simulator container may have no git; logging must still start, and say it does not know."""
    monkeypatch.setenv("PATH", str(tmp_path))
    assert git_revision() == UNKNOWN_REVISION


@pytest.mark.parametrize("status,expected", [
    ("", "abc1234"), (" M core/x.py\n", "abc1234-dirty"),
    (None, "abc1234-dirty")], ids=["clean", "changed", "status_failed"])
def test_a_tree_is_dirty_unless_git_confirms_it_clean(monkeypatch, status, expected):
    """A clean revision promises the commit alone reproduces the run; an unread status must not make that promise."""
    def fake_run(args, **kwargs):
        out = "abc1234\n" if args[1] == "rev-parse" else status
        return subprocess.CompletedProcess(args, 128 if out is None else 0,
                                           stdout=out or "", stderr="")
    monkeypatch.setattr(results_io_module.subprocess, "run", fake_run)
    assert git_revision() == expected
