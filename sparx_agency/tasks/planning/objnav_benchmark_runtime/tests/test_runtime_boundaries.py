"""Execution provenance and optional-native-import regressions; no GPU or services."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import frozen_eval, run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.evaluation_lock import (
    check_frozen_configuration, freeze_configuration,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_evaluation_lock import configuration


@pytest.mark.parametrize("key", ["source_sha256", "method", "dataset"])
def test_actual_execution_configuration_is_checked_before_output(tmp_path, monkeypatch, key):
    config = configuration()
    lock = tmp_path / "lock.json"
    freeze_configuration(config, lock)
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    changed = dict(config, **{key: "changed-after-preflight"})
    monkeypatch.setattr(run, "prepare", lambda args: (env, None, changed, []))
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="changed"):
        run.main(["--output", str(output)], configuration_guard=lambda current: check_frozen_configuration(current, lock))
    assert closed == [True]
    assert not output.exists()


def test_frozen_wrapper_checks_second_prepare_not_only_first(tmp_path, monkeypatch):
    config = configuration()
    lock = tmp_path / "lock.json"
    freeze_configuration(config, lock)
    monkeypatch.setattr(frozen_eval, "prepare", lambda args: (None, None, config, []))
    checked = []

    def execute(argv, *, configuration_guard):
        checked.append(True)
        configuration_guard(dict(config, source_sha256="changed-after-preflight"))
        pytest.fail("Configuration drift must prevent execution")

    monkeypatch.setattr(frozen_eval, "run_main", execute)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="changed"):
        frozen_eval.main(["run", "--lock", str(lock), "--", "--output", str(output)])
    assert checked == [True]
    assert not (output / "evaluation_role.json").exists()


def _subprocess(code):
    root = next(p for p in Path(__file__).resolve().parents if (p / "sparx_agency").is_dir())
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_astar_and_objnav_do_not_even_attempt_ompl_import():
    _subprocess('''
import importlib.abc
import sys
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ompl" or fullname.startswith("ompl."):
            attempts.append(fullname)
            raise ImportError("Native dependency forbidden in this pure-geometry process")
sys.meta_path.insert(0, Guard())
from sparx_agency.core.planning.planners import WeightedAStarPlanner2D
from sparx_agency.core.planning.planners.common import split_long_segments_2d, dist3d
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run
assert not attempts, attempts
assert not any(n == "ompl" or n.startswith("ompl.") for n in sys.modules)
''')


def test_existing_optional_exports_and_missing_dependency_errors_are_preserved():
    _subprocess('''
import sys
from types import ModuleType
from sparx_agency.core.planning import planners
from sparx_agency.core.planning.planners import common
fake = ModuleType(common.__name__ + ".ompl_imports")
fake.ob = fake.og = None
fake.OMPL_AVAILABLE = False
fake.OMPL_ERROR = "test missing backend"
sys.modules[fake.__name__] = fake
assert common.OMPL_AVAILABLE is False
assert common.OMPL_ERROR == "test missing backend"
assert common.ob is None and common.og is None
assert len(common.__all__) == 16
for builder in (common.make_clearance_objective_2d, common.make_clearance_objective_3d):
    try:
        builder(None, None, 1.0)
    except RuntimeError as exc:
        assert str(exc) == "OMPL not available"
    else:
        raise AssertionError("Missing OMPL must fail explicitly")
try:
    common.setup_ompl_space_3d(None, None)
except RuntimeError as exc:
    assert str(exc) == "OMPL not available"
else:
    raise AssertionError("Missing OMPL must fail explicitly")
rrt = ModuleType(planners.__name__ + ".rrtstar")
rrt.RRTStarOmplPlanner = object()
sys.modules[rrt.__name__] = rrt
assert planners.RRTStarOmplPlanner is rrt.RRTStarOmplPlanner
assert planners.RRTStarOmplPlanner is rrt.RRTStarOmplPlanner
try:
    planners.not_a_planner
except AttributeError:
    pass
else:
    raise AssertionError("Unknown exports must raise AttributeError")
''')

