"""The benchmark runtime hands the agent nothing of the simulator's scene geometry.

The other half of ``core/planning/object_search/tests/test_no_privileged_geometry.py``.
That one proves the agent cannot reach for the navmesh, the pathfinder or
ground-truth storeys; this one that the evaluator does not hand them over. Until
2026-11-15 it did, twice: every run script bound ``policy.target_projector`` to the
environment's navmesh snap, and both environment adapters put the navmesh's storeys and
stair connectors into ``ObjNavEpisode.metadata``.

The navmesh stays on this side of the agent interface, where the benchmark needs it --
the simulator's physics, episode setup and start validation, the geodesic a score is
measured by, the harness's kinematics check, after-the-fact analysis -- so the rules
here are about the SEAMS, not about the word:

* no ``ObjNavEpisode`` is built with metadata;
* nothing derived from an environment or a simulator is assigned onto a policy;
* the projector seam does not exist under any of its names;
* the navmesh analysis (``habitat/stair_connectors.py``) is imported by analysis only.

The source scanners are the core test's own, imported rather than copied: two copies
of a rule is how one of them goes stale.
"""
from __future__ import annotations

import ast

import pytest

from sparx_agency.core.planning.object_search.tests.test_no_privileged_geometry import (
    REPO_ROOT, SCENE_STRUCTURE, identifiers, is_test_module, parse, where)
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode

TASKS = REPO_ROOT / "sparx_agency" / "tasks" / "planning"
#: The harness (scoring, contract checks, the fake environment) and the runtime (simulator bridge, adapters, run scripts).
PACKAGES = (TASKS / "objnav_benchmark", TASKS / "objnav_benchmark_runtime")
RUNTIME = "sparx_agency/tasks/planning/objnav_benchmark_runtime/"

#: The seam the navmesh projection of a standing point went through, under each of its names.
PROJECTOR_SEAM = ("target_projector", "project_target", "target_navmesh_projection")
#: The navmesh analysis, and the only modules that may import it: they look at episodes from outside.
ANALYSIS_MODULE = "sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors"
ANALYSIS_IMPORTERS = frozenset({RUNTIME + "hm3d/episode_levels.py"})
#: Names an expression must not carry onto a policy: the environment, the simulator bridge, its mesh.
SIMULATOR_HANDLES = frozenset({"env", "environment", "simulator", "_simulator", "sim", "_sim", "bridge", "pathfinder"})


def benchmark_modules():
    """Every library module of the harness and the runtime (tests excluded), sorted."""
    return sorted(p for package in PACKAGES for p in package.rglob("*.py")
                  if "__pycache__" not in p.parts and not is_test_module(p.relative_to(package)))


def _names(node: ast.AST) -> set:
    """Every bare name and attribute name inside an expression."""
    return ({sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}
            | {sub.attr for sub in ast.walk(node) if isinstance(sub, ast.Attribute)})


def _is_policy(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "policy"


def bindings_onto_a_policy(tree: ast.AST):
    """``(line, attribute)`` of every assignment that puts something of an environment or a simulator on ``policy``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            for target in targets:
                for single in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                    if _is_policy(single) and _names(value) & SIMULATOR_HANDLES:
                        yield node.lineno, single.attr
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
              and len(node.args) == 3 and isinstance(node.args[0], ast.Name) and node.args[0].id == "policy"
              and _names(node.args[2]) & SIMULATOR_HANDLES):
            yield node.lineno, getattr(node.args[1], "value", "?")


def episodes_built_with_metadata(tree: ast.AST):
    """The line of every ``ObjNavEpisode(...)`` call that passes metadata, by keyword or as the ninth positional."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "ObjNavEpisode":
            if any(k.arg == "metadata" or k.arg is None for k in node.keywords) or len(node.args) > 8:
                yield node.lineno


def test_the_scan_covers_the_adapters_the_run_scripts_and_the_bridge():
    names = {where(p) for p in benchmark_modules()}
    assert len(names) >= 60, len(names)
    for expected in ("habitat/env.py", "habitat/simulator.py", "gibson/env.py", "hm3d/env.py", "mp3d/env.py",
                     "hm3d/run.py", "mp3d/run.py", "gibson/run.py", "gibson/run_development.py", "policy_factory.py"):
        assert RUNTIME + expected in names, "the scan missed %s" % expected


def test_no_episode_is_built_with_metadata():
    offenders = [where(p, line) for p in benchmark_modules() for line in episodes_built_with_metadata(parse(p))]
    assert not offenders, (
        "an environment hands the agent an episode with metadata. The navmesh's storeys and stair connectors went "
        "to the agent this way until 2026-11-15; a public fact about an episode gets a typed field on "
        "ObjNavEpisode instead:\n  " + "\n  ".join(offenders))


def test_nothing_of_an_environment_or_a_simulator_is_bound_to_a_policy():
    offenders = ["%s: policy.%s" % (where(p, line), attribute)
                 for p in benchmark_modules() for line, attribute in bindings_onto_a_policy(parse(p))]
    assert not offenders, (
        "a run script binds part of the environment or the simulator to the policy -- the way the navmesh "
        "projector reached the closing. The policy is given observations and nothing else:\n  " + "\n  ".join(offenders))


def test_the_projector_seam_is_gone_under_every_name():
    offenders = ["%s spells %r" % (where(p, line), name)
                 for p in benchmark_modules() for name, line in identifiers(parse(p)) if name in PROJECTOR_SEAM]
    assert not offenders, "the navmesh projector seam is back:\n  " + "\n  ".join(offenders)


def test_the_navmesh_analysis_is_imported_by_analysis_only():
    importers, readers = set(), []
    for p in benchmark_modules():
        tree = parse(p)
        for node in ast.walk(tree):
            imported = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.startswith(ANALYSIS_MODULE) for name in imported):
                importers.add(where(p))
        if where(p) != RUNTIME + "habitat/stair_connectors.py":
            readers += ["%s spells %r" % (where(p, line), name) for name, line in identifiers(tree)
                        if name in ("scene_structure", "scene_structure_from_pathfinder")]
    assert importers == set(ANALYSIS_IMPORTERS), (
        "habitat/stair_connectors.py reads storeys and stair connectors off the navmesh: ground truth, for "
        "looking at episodes from outside. An adapter, a run script or the simulator bridge importing it is "
        "how it reached the agent before. Importers now: %s" % sorted(importers))
    assert not readers, "the scene structure is read outside the analysis module:\n  " + "\n  ".join(readers)


def test_the_bridge_and_the_adapters_expose_no_seam():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.env import HabitatObjNavEnv
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.env import HM3DEnv
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
    for bridge_attribute in ("project_target", "scene_structure"):
        assert not hasattr(HabitatRGBDSimulator, bridge_attribute)
    for adapter in (HabitatObjNavEnv, GibsonEnv, HM3DEnv, MP3DEnv):
        assert not hasattr(adapter, "target_projector"), adapter.__name__
        assert not hasattr(adapter, "_scene_metadata"), adapter.__name__


def test_the_harness_refuses_an_episode_that_carries_the_scene_structure():
    """Defence in depth: an adapter that attached it again would not get past the first reset."""
    from sparx_agency.core.planning.object_search.tests.fixtures import benchmark_camera, body_actions
    from sparx_agency.tasks.planning.objnav_benchmark.env_contract import _check_metadata

    def episode(metadata):
        return ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", "chair", benchmark_camera(),
                             body_actions(), 500, metadata=metadata)

    _check_metadata(episode({}))
    with pytest.raises(EnvContractError, match="names privileged information"):
        _check_metadata(episode(SCENE_STRUCTURE))
    for key in SCENE_STRUCTURE:
        with pytest.raises(EnvContractError, match=key):
            _check_metadata(episode({key: SCENE_STRUCTURE[key]}))


@pytest.mark.parametrize("source", [
    "policy.target_projector = getattr(env, 'target_projector', None)\n",
    "policy.mesh = env._simulator.pathfinder\n",
    "policy.snap, policy.other = sim.snap, 1\n",
    "setattr(policy, 'mesh', simulator.pathfinder)\n",
])
def test_the_binding_scan_catches_a_planted_binding(source):
    assert list(bindings_onto_a_policy(ast.parse(source)))


@pytest.mark.parametrize("source", [
    "ObjNavEpisode(a, b, c, d, e, f, g, h, metadata=self._scene_metadata())\n",
    "ObjNavEpisode(a, b, c, d, e, f, g, h, structure)\n",
    "types.ObjNavEpisode(a, b, c, d, e, f, g, h, **extras)\n",
])
def test_the_episode_scan_catches_planted_metadata(source):
    assert list(episodes_built_with_metadata(ast.parse(source)))


def test_the_scans_leave_ordinary_wiring_alone():
    clean = ("policy.detector = FakeDetector()\n"
             "policy, method = _method(args, protocol, ground_truth)\n"
             "episode = ObjNavEpisode(a, b, c, d, e, f, g, h)\n"
             "agent.height = self.height_m\n")
    tree = ast.parse(clean)
    assert not list(bindings_onto_a_policy(tree)) and not list(episodes_built_with_metadata(tree))
