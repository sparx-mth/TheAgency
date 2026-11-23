"""Nothing under ``sparx_agency/core`` can reach the simulator's navmesh, its pathfinder or ground-truth storey data.

The agent is zero-shot: what it knows of a building is what its own RGB-D and
pose measured. Until 2026-11-15 two things reached it from the simulator all
the same -- a ``target_projector`` the evaluator bound to the policy, which
snapped the closing's standing points onto the navigation mesh, and the
navmesh's storeys and stair connectors in ``ObjNavEpisode.metadata``, read by a
``ground_truth`` stair source. ASCENT, ApexNav, VLFM and MFNP give their agents
neither. Both are removed; this module is what keeps them, and anything like
them, out. The navmesh may be used OUTSIDE the agent only: the simulator's own
physics, episode setup, scoring, preflight checks, after-the-fact analysis --
all of which live under ``sparx_agency/tasks``.

There are three ways privileged geometry can reach code, and each is closed
separately:

1. **By import.** The transitive closure of every import statement in every
   library module of ``core`` is followed through the repository (lazy imports
   inside functions included; a computed ``import_module`` must be anchored in
   its own package and registered here). No module of the closure may be the
   ObjectNav benchmark harness or runtime, and none may import the simulator.
2. **By name.** A handle injected at run time (as ``target_projector`` was)
   needs no import -- but the code using it must spell the service. No
   identifier in ``core`` -- name, attribute, argument, definition, or a string
   used as a key or an attribute name -- may name the navmesh, the pathfinder's
   API, the projector seam, or the storey/stair structure. Docstrings, comments
   and prose strings may: the history is worth telling.
3. **By the episode.** ``ObjNavEpisode.metadata`` is the one free-form field
   the agent is handed. No ``core`` module reads it, and a policy handed the
   navmesh's storeys and stairs there behaves exactly as one handed nothing.

Each scanner is run against planted leaks (``test_the_scanners_catch_...``),
so a rule that silently matches nothing fails here rather than passing.

Standard library plus the package's own test rig; no simulator, no ``tasks``.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import re

import pytest

#: .../core/planning/object_search  (this file is <object_search>/tests/<name>.py)
PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1]
#: .../sparx_agency
SPARX_ROOT = PACKAGE_DIR.parents[2]
REPO_ROOT = SPARX_ROOT.parent
CORE_DIR = SPARX_ROOT / "core"

#: The scan must find at least this many library modules under ``core``; fewer means it looks in the wrong place.
MIN_CORE_MODULES = 400

# -- 1. by import ------------------------------------------------------------------------------------------------
#: Third-party roots that ARE the simulator.
SIMULATOR_ROOTS = frozenset({"habitat", "habitat_sim", "habitat_baselines", "magnum"})
#: Repository packages that hold a navmesh: the benchmark harness (scoring) and its runtime (simulator bridge,
#: adapters, navmesh analysis). ``core`` must not reach either, however indirectly.
BENCHMARK_PACKAGES = ("sparx_agency/tasks/planning/objnav_benchmark", "sparx_agency/tasks/planning/objnav_benchmark_runtime")
#: Modules allowed to import by a computed name (PEP 562 lazy exports). Each must anchor the import in its own
#: package (``__name__``), so what it can load is already in the scan. A new one is a decision, not an accident.
DYNAMIC_IMPORTERS = frozenset({
    "sparx_agency/core/planning/planners/__init__.py",
    "sparx_agency/core/planning/planners/common/__init__.py",
    "sparx_agency/core/planning/safety/__init__.py",
})

# -- 2. by name --------------------------------------------------------------------------------------------------
#: The navigation mesh and the pathfinder's API (habitat-sim's ``PathFinder``/``ShortestPath``), whatever the
#: handle holding them is called. Matched case-insensitively inside an identifier.
NAVMESH_NAMES = (
    r"nav_?mesh", r"path_?finder", r"snap_point", r"find_path", r"is_navigable", r"navigable_(point|area)",
    r"(?<![a-z])try_step", r"geodesic_distance", r"closest_obstacle", r"island_radius", r"get_island", r"habitat",
)
#: The seam the navmesh projection of a standing point came through.
PROJECTOR_NAMES = (r"target_projector", r"project_target")
#: Ground-truth storeys and stairs: the structure the adapters attached, and the modules that read it.
STOREY_NAMES = (
    r"scene_structure", r"stair_connector", r"floor_levels?(?![a-z])", r"storey_levels?(?![a-z])", r"storey_position",
    r"stair_ground_truth", r"stair_sightings", r"ground_truth_traversal", r"from_metadata",
    r"(ground_?truth|(?<![a-z])gt)_?(stair|storey|stories|floor|level|connector)",
    r"(stair|storey|stories|floor|level|connector)s?_?(ground_?truth|gt(?![a-z]))",
)
FORBIDDEN_NAMES = tuple((kind, re.compile(pattern, re.IGNORECASE))
                        for kind, patterns in (("the navmesh or the pathfinder", NAVMESH_NAMES),
                                               ("the navmesh projector seam", PROJECTOR_NAMES),
                                               ("ground-truth storey or stair data", STOREY_NAMES))
                        for pattern in patterns)
#: The one declaration that may name ground-truth stairs: ``configuration()["ground_truth_stairs"]``, and only as
#: the literal ``False`` -- kept so a run of this code can be told from one recorded before 2026-11-15.
DECLARATION = "ground_truth_stairs"
#: A string constant is read as a name when it could be one (a key, an attribute name), not when it is prose.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")

# -- 3. by the episode -------------------------------------------------------------------------------------------
#: What the simulator's navmesh says about a two-storey house, in the shape the adapters attached it.
SCENE_STRUCTURE = {
    "stair_source": "navmesh",
    "floor_levels": [{"height_m": 0.0, "area_m2": 40.0}, {"height_m": 2.7, "area_m2": 40.0}],
    "stair_connectors": [{"id": 0, "bottom_xyz": [1.5, 0.0, 0.0], "top_xyz": [4.5, 0.0, 2.7], "bottom_z": 0.0,
                          "top_z": 2.7, "length_m": 4.0, "rise_m": 2.7, "traversable": True,
                          "polyline_xyz": [[1.5, 0.0, 0.0], [2.5, 0.0, 0.9], [3.5, 0.0, 1.8], [4.5, 0.0, 2.7]],
                          "surface_xyz": [[2.0, 0.0, 0.45], [3.0, 0.0, 1.35], [4.0, 0.0, 2.25]]}],
}


# ================================================================================================================
# The scanners. Pure functions of source text, so the planted-leak tests below can run them on snippets.
# ================================================================================================================
def is_test_module(path: pathlib.Path) -> bool:
    """Whether ``path`` is test code: under a ``tests`` directory, or named like a test or a rig."""
    return ("tests" in path.parts or path.name.startswith("test_") or path.name.endswith("_test.py")
            or path.name == "conftest.py")


def core_modules():
    """Every library module under ``sparx_agency/core`` (tests excluded), sorted."""
    return sorted(p for p in CORE_DIR.rglob("*.py")
                  if "__pycache__" not in p.parts and not is_test_module(p.relative_to(CORE_DIR)))


def where(path, line=None) -> str:
    name = str(pathlib.Path(path).relative_to(REPO_ROOT)) if pathlib.Path(path).is_absolute() else str(path)
    return name if line is None else "%s:%d" % (name, line)


def parse(path: pathlib.Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), str(path))


def docstring_nodes(tree: ast.AST) -> set:
    """``id`` of every docstring's constant node: prose, not code."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                found.add(id(first.value))
    return found


def false_declarations(tree: ast.AST) -> set:
    """``id`` of every ``"ground_truth_stairs"`` dict key whose value is the literal ``False``."""
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == DECLARATION
                        and isinstance(value, ast.Constant) and value.value is False):
                    allowed.add(id(key))
    return allowed


def identifiers(tree: ast.AST):
    """``(identifier, line)`` for every name the code in ``tree`` spells -- and every string that could be one."""
    prose, declared = docstring_nodes(tree), false_declarations(tree)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            yield node.id, line
        elif isinstance(node, ast.Attribute):
            yield node.attr, line
        elif isinstance(node, ast.arg):
            yield node.arg, line
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg, line
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, line
        elif isinstance(node, ast.alias):
            yield node.name, line
            if node.asname:
                yield node.asname, line
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, line
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose
              and id(node) not in declared and IDENTIFIER.match(node.value)):
            yield node.value, line


def privileged_names(tree: ast.AST):
    """``(identifier, what it names, line)`` for every identifier in ``tree`` that names privileged geometry."""
    for name, line in identifiers(tree):
        for kind, pattern in FORBIDDEN_NAMES:
            if pattern.search(name):
                yield name, kind, line
                break


def _terminal_name(node: ast.AST) -> str:
    """The last name of an expression such as ``self.policy.episode`` (``episode``); empty for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_episode(node: ast.AST) -> bool:
    name = _terminal_name(node).lower()
    return "episode" in name or name == "ep"


def episode_metadata_reads(tree: ast.AST):
    """The line of every read of an episode's ``metadata``: ``<...episode>.metadata`` or ``getattr(<...episode>, "metadata", ...)``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "metadata" and _is_episode(node.value):
            if not isinstance(node.ctx, ast.Store):
                yield node.lineno
        elif (isinstance(node, ast.Call) and _terminal_name(node.func) == "getattr" and len(node.args) >= 2
              and isinstance(node.args[1], ast.Constant) and node.args[1].value == "metadata"
              and _is_episode(node.args[0])):
            yield node.lineno


def _module_file(name: str):
    """The repository file a dotted ``sparx_agency`` module name resolves to, or None (third party, or not a module)."""
    parts = name.split(".")
    if parts[0] != "sparx_agency":
        return None
    base = REPO_ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _package_of(path: pathlib.Path):
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    return parts[:-1]                     # a module's package; for ``__init__`` the package itself


def imports_of(tree: ast.AST, package):
    """``(dotted name, line)`` of everything ``tree`` imports, relative imports resolved against ``package``.

    ``from a import b`` yields both ``a`` and ``a.b`` -- ``b`` may be a module. A computed import
    (``import_module`` / ``__import__``) yields ``("<dynamic>", line)`` followed by nothing else: what it
    loads cannot be read off the source, and :func:`dynamic_import_is_anchored` decides whether that is safe.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = list(package[:len(package) - (node.level - 1)])
                base = ".".join(anchor + ([node.module] if node.module else []))
            yield base, node.lineno
            for alias in node.names:
                yield "%s.%s" % (base, alias.name), node.lineno
        elif isinstance(node, ast.Call) and _terminal_name(node.func) in ("import_module", "__import__"):
            yield "<dynamic>", node.lineno


def dynamic_import_is_anchored(tree: ast.AST) -> bool:
    """Whether every computed import in ``tree`` names ``__name__`` among its arguments: it loads from its own package."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _terminal_name(node.func) in ("import_module", "__import__"):
            names = {sub.id for arg in list(node.args) + [k.value for k in node.keywords]
                     for sub in ast.walk(arg) if isinstance(sub, ast.Name)}
            if "__name__" not in names and "__package__" not in names:
                return False
    return True


def import_closure(start):
    """Every repository module reachable from ``start`` by import statements, with how it was reached.

    Returns:
        ``(reached, third_party, dynamic)``: ``{module file: the file that first imported it (None for a
        start module)}``; ``{third-party root: [where it is imported]}``; and the files holding a computed import.
    """
    reached = {path: None for path in start}
    queue = collections.deque(start)
    third_party = collections.defaultdict(list)
    dynamic = {}
    while queue:
        path = queue.popleft()
        tree = parse(path)
        for name, line in imports_of(tree, _package_of(path)):
            if name == "<dynamic>":
                dynamic.setdefault(path, tree)
                continue
            target = _module_file(name)
            if target is None:
                if not name.startswith("sparx_agency"):
                    third_party[name.split(".")[0]].append(where(path, line))
                continue
            parts = name.split(".")
            # Importing ``a.b.c`` runs ``a/__init__`` and ``a/b/__init__`` first.
            chain = [target] + [REPO_ROOT.joinpath(*parts[:i]) / "__init__.py" for i in range(1, len(parts))]
            for module in chain:
                if module.is_file() and module not in reached:
                    reached[module] = path
                    queue.append(module)
    return reached, third_party, dynamic


def import_chain(reached, module) -> str:
    """``core/a.py -> tasks/b.py -> ...`` : how ``module`` was reached."""
    chain = [module]
    while reached.get(chain[-1]) is not None:
        chain.append(reached[chain[-1]])
    return " <- ".join(where(p) for p in chain)


@pytest.fixture(scope="module")
def closure():
    return import_closure(core_modules())


# ================================================================================================================
# The scan itself
# ================================================================================================================
def test_the_scan_covers_the_whole_of_core():
    files = core_modules()
    names = {where(p) for p in files}
    assert len(files) >= MIN_CORE_MODULES, len(files)
    for expected in ("sparx_agency/core/planning/object_search/policy.py",
                     "sparx_agency/core/planning/object_search/baselines/falcon_yolo.py",
                     "sparx_agency/core/planning/object_search/closing/target_path.py",
                     "sparx_agency/core/planning/object_search/closing/target_backoff.py",
                     "sparx_agency/core/planning/object_search/closing/target_closing.py",
                     "sparx_agency/core/planning/object_search/world_model/standing_floor.py",
                     "sparx_agency/core/planning/object_search/multifloor/multifloor_policy.py",
                     "sparx_agency/core/localization/floor_atlas.py",
                     "sparx_agency/core/planning/objnav/types/episode.py",
                     "sparx_agency/core/planning/planners/astar/weighted_planner_2d.py",
                     "sparx_agency/core/mapping/topology/search_node_oracle.py",
                     # Where a room's size comes from, and what it is read off: the agent's own map only.
                     "sparx_agency/core/mapping/topology/room_extent.py",
                     "sparx_agency/core/planning/object_search/rooms/scene_graph.py",
                     "sparx_agency/core/planning/object_search/world_model/sightlines.py"):
        assert expected in names, "the scan missed %s" % expected
    assert not [n for n in names if "/tests/" in n], "tests are not the agent"


# -- 1. by import ------------------------------------------------------------------------------------------------
def test_core_cannot_import_its_way_to_the_simulator_or_the_benchmark(closure):
    reached, third_party, _ = closure
    simulators = ["%r, imported at %s" % (root, ", ".join(sorted(sites)[:3]))
                  for root, sites in sorted(third_party.items()) if root in SIMULATOR_ROOTS]
    assert not simulators, ("code reachable from sparx_agency/core imports the simulator; the navmesh and the "
                            "pathfinder live behind it:\n  " + "\n  ".join(simulators))
    benchmark = [import_chain(reached, module) for module in sorted(reached)
                 if str(module.relative_to(REPO_ROOT)).startswith(BENCHMARK_PACKAGES)]
    assert not benchmark, ("sparx_agency/core reaches the ObjectNav benchmark harness or runtime, which hold the "
                           "navmesh (scoring, the simulator bridge, the navmesh analysis); the agent must not:\n  "
                           + "\n  ".join(benchmark))


def test_no_computed_import_in_core_escapes_the_scan(closure):
    _, _, dynamic = closure
    found = {where(path) for path in dynamic}
    assert found == set(DYNAMIC_IMPORTERS), (
        "a computed import (import_module / __import__) cannot be followed by this scan. Use a plain import "
        "statement (inside the function, if it must be lazy), or -- for a lazy export of the module's own "
        "package -- add the file to DYNAMIC_IMPORTERS. Unregistered: %s; registered but gone: %s"
        % (sorted(found - set(DYNAMIC_IMPORTERS)), sorted(set(DYNAMIC_IMPORTERS) - found)))
    loose = [where(path) for path, tree in dynamic.items() if not dynamic_import_is_anchored(tree)]
    assert not loose, ("a computed import must be anchored in its own package (``import_module(name, __name__)``), "
                       "so everything it can load is already in the scan: %s" % loose)


# -- 2. by name --------------------------------------------------------------------------------------------------
def test_core_names_nothing_of_the_navmesh_the_pathfinder_or_ground_truth_storeys():
    offenders = []
    for path in core_modules():
        for name, kind, line in privileged_names(parse(path)):
            offenders.append("%s spells %r (%s)" % (where(path, line), name, kind))
    assert not offenders, (
        "sparx_agency/core names privileged scene geometry. The agent reads its own perception map: a standing "
        "point comes from world_model/standing_floor.py, a staircase from world_model/stair_terrain.py. The "
        "navmesh may be used only outside the agent (tasks/: physics, episode setup, scoring, preflight, "
        "analysis):\n  " + "\n  ".join(offenders))


def test_the_ground_truth_stairs_declaration_is_the_literal_false_wherever_it_is_made():
    """``configuration()["ground_truth_stairs"]`` survives so old and new runs can be told apart -- as a constant."""
    declared = []
    for path in core_modules():
        tree = parse(path)
        allowed = false_declarations(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == DECLARATION:
                assert id(node) in allowed, ("%s: %r may only be declared as the literal False"
                                             % (where(path, node.lineno), DECLARATION))
                declared.append(where(path))
    assert sorted(declared) == ["sparx_agency/core/planning/object_search/baselines/falcon_yolo.py",
                                "sparx_agency/core/planning/object_search/policy.py"]


# -- 3. by the episode -------------------------------------------------------------------------------------------
def test_core_never_reads_the_episodes_metadata():
    offenders = [where(path, line) for path in core_modules() for line in episode_metadata_reads(parse(path))]
    assert not offenders, (
        "ObjNavEpisode.metadata is the one free-form field an adapter hands the agent, and the navmesh's storeys "
        "and stair connectors came through it. Nothing in sparx_agency/core reads it; if the agent needs a new "
        "public fact about an episode, give ObjNavEpisode a typed field:\n  " + "\n  ".join(offenders))


class Untouchable:
    """Stands where an episode's metadata would. Any use of it at all is recorded, then refused."""

    def __init__(self):
        object.__setattr__(self, "touched", [])

    def _refuse(self, how):
        object.__getattribute__(self, "touched").append(how)
        raise AssertionError("the agent used episode.metadata (%s)" % how)

    def __getattribute__(self, name):
        if name in ("touched", "_refuse", "__class__"):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_refuse")("." + name)

    def __getitem__(self, key):
        self._refuse("[%r]" % (key,))

    def __contains__(self, key):
        self._refuse("%r in" % (key,))

    def __iter__(self):
        self._refuse("iter")

    def __len__(self):
        self._refuse("len")

    def __bool__(self):
        self._refuse("bool")


def _policies(episode_of):
    """The full method (stairs forbidden and allowed) and the FALCON + YOLO baseline, each reset on its own episode."""
    from sparx_agency.core.planning.object_search.baselines.falcon_yolo import FalconYoloBaseline
    from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy, LegacyRoomSearchPolicy
    from sparx_agency.core.planning.object_search.settings import RPTSettings
    from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, FakeLLM
    from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper

    def settings(**overrides):
        return RPTSettings(**dict({"map_size_m": 20.0, "doorway_peek": {"enabled": False}}, **overrides))

    builders = {
        "rpt": lambda: RPTSearchPolicy(FakeDetector("bed"), FakeLLM(), settings()),
        "rpt-stairs-allowed": lambda: LegacyRoomSearchPolicy(FakeDetector("bed"), FakeLLM(), settings(allow_stair_traversal=True)),
        "falcon-yolo": lambda: FalconYoloBaseline(FakeDetector("bed"), settings()),
    }
    for name, build in builders.items():
        policy, episode = build(), episode_of()
        policy.reset(episode, gibson_label_mapper().target_labels("chair"))
        yield name, policy, episode


def _fly(policy, episode, steps=6):
    """A few actions on synthetic frames -- a walk forward and a turn -- and the record a run keeps."""
    from sparx_agency.core.planning.object_search.tests.fixtures import frame
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    commands = []
    for step in range(steps):
        pose = AgentPose(0.25 * min(step, 3), 0.0, 0.0, 0.5 * max(0, step - 3))
        command = policy.plan(frame(episode, step, depth=3.0, pose=pose))
        commands.append((command.stop, command.final_yaw,
                         tuple(tuple(round(float(v), 6) for v in point) for point in (command.waypoints or ()))))
    return commands, policy.configuration(), policy.episode_info()


def test_a_policy_never_touches_the_metadata_of_the_episode_it_is_handed():
    from sparx_agency.core.planning.object_search.tests.fixtures import episode

    def booby_trapped():
        ep = episode()
        object.__setattr__(ep, "metadata", Untouchable())       # frozen dataclass: placed after its own checks
        return ep

    flown = []
    for name, policy, ep in _policies(booby_trapped):
        _fly(policy, ep)
        assert ep.metadata.touched == [], "%s read the episode's metadata: %s" % (name, ep.metadata.touched)
        flown.append(name)
    assert flown == ["rpt", "rpt-stairs-allowed", "falcon-yolo"]


def test_a_policy_handed_the_navmeshs_storeys_and_stairs_is_none_the_wiser():
    """The adapters attach nothing any more, and the harness refuses it at reset; were it handed over, nothing reads it."""
    from sparx_agency.core.planning.object_search.tests.fixtures import episode
    plain = {name: _fly(policy, ep) + (policy,) for name, policy, ep in _policies(episode)}
    leaked = {name: _fly(policy, ep) + (policy,)
              for name, policy, ep in _policies(lambda: episode(metadata=SCENE_STRUCTURE))}
    assert set(plain) == set(leaked) == {"rpt", "rpt-stairs-allowed", "falcon-yolo"}
    for name in plain:
        commands, configuration, _, policy = plain[name]
        told_commands, told_configuration, _, told = leaked[name]
        assert told_commands == commands, "%s acted on the scene structure" % name
        assert told_configuration == configuration
        assert told.building.portals == policy.building.portals == [], "no staircase from a file: none was seen"
        assert len(told.mapping.atlas.floors) == 1, "one storey: the one stood on, not the two the navmesh has"
        assert not hasattr(told.building, "ground_truth") and not hasattr(told.building, "sightings")


def test_the_policies_have_no_navmesh_seam_and_no_ground_truth_stair_source():
    from sparx_agency.core.localization import floor_atlas
    from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosingSettings
    from sparx_agency.core.planning.object_search.settings import RPTSettings
    from sparx_agency.core.planning.object_search.tests.fixtures import episode
    for name, policy, _ in _policies(episode):
        configuration = policy.configuration()
        assert not hasattr(policy, "target_projector"), "%s: nothing for an evaluator to bind a navmesh to" % name
        assert "target_navmesh_projection" not in configuration
        assert "observed map" in configuration["standing_points"]
        assert configuration["ground_truth_stairs"] is False and configuration["stair_source"] == "observed"
        assert policy.closing.standing_floor is not None
    assert floor_atlas.STAIR_SOURCES == ("observed",)
    with pytest.raises(ValueError, match="removed"):
        RPTSettings(multifloor={"stair_source": "ground_truth"})
    # The closing's route through unobserved space went with the projection: no setting brings it back.
    for removed in ("through_unknown", "through_unknown_cost"):
        with pytest.raises(TypeError):
            TargetClosingSettings(**{removed: True})


# ================================================================================================================
# The scanners catch what they are for (a rule that matches nothing would pass every test above)
# ================================================================================================================
LEAKS = {
    "the pathfinder by its API": "def stand(sim, p):\n    return sim.pathfinder.snap_point(p)\n",
    "a pathfinder under another name": "def stand(mesh, p):\n    return mesh.is_navigable(p)\n",
    "a path query": "def far(q):\n    return q.geodesic_distance\n",
    "a step on the mesh": "def step(nav, a, b):\n    return nav.try_step(a, b)\n",
    "the navmesh as state": "class Closing:\n    def __init__(self, mesh):\n        self.navmesh = mesh\n",
    "the projector seam": "def goal(policy, xyz):\n    return policy.target_projector(xyz)\n",
    "the projector by getattr": "def goal(env):\n    return getattr(env, 'target_projector', None)\n",
    "the projection itself": "def goal(sim, xyz):\n    return sim.project_target(xyz)\n",
    "stair connectors by key": "def stairs(structure):\n    return structure['stair_connectors']\n",
    "storey levels by get": "def storeys(structure):\n    return structure.get('floor_levels', [])\n",
    "the structure reader": "def structure(sim):\n    return sim.scene_structure()\n",
    "the ground-truth stairs class": "class GroundTruthStairs:\n    pass\n",
    "a ground-truth stair flag set true": "CONFIG = {'ground_truth_stairs': True}\n",
    "a ground-truth storey count": "def rank(building):\n    return len(building.gt_storeys)\n",
    "the simulator by import": "from habitat_sim import PathFinder\n",
}
CLEAN = {
    "the agent's own geodesic": "def walk(floor, goal):\n    return floor.geodesic_m() + goal.geodesic_m\n",
    "retry and entry steps": "def due(state):\n    return state.retry_step > state._entry_step\n",
    "the declaration, false": "CONFIG = {'ground_truth_stairs': False, 'ground_truth_semantics': False}\n",
    "history in a docstring": 'def stand():\n    """Until 2026-11-15 this asked the navmesh (pathfinder.snap_point)."""\n',
    "history in prose": "NOTE = 'standing points: the observed map only, never the navmesh or a target_projector'\n",
    "a room label's metadata": "def strength(tracker, pid):\n    return tracker.metadata.get(pid)\n",
    "observed stairs and floors": "def portals(building):\n    return [q for q in building.portals if q['floor_id'] == building.floor_id]\n",
    "a level camera": "def level(camera):\n    return camera.level_pitch\n",
}


@pytest.mark.parametrize("name", sorted(LEAKS))
def test_the_scanners_catch_a_planted_leak(name):
    assert list(privileged_names(ast.parse(LEAKS[name]))), "a %s went unnoticed" % name


@pytest.mark.parametrize("name", sorted(CLEAN))
def test_the_scanners_leave_the_agents_own_vocabulary_alone(name):
    tree = ast.parse(CLEAN[name])
    assert not list(privileged_names(tree)), list(privileged_names(tree))
    assert not list(episode_metadata_reads(tree))


@pytest.mark.parametrize("source", [
    "def stairs(policy):\n    return policy.episode.metadata.get('x')\n",
    "def stairs(self):\n    return getattr(self.policy.episode, 'metadata', None)\n",
    "def stairs(episode):\n    return dict(episode.metadata)\n",
    "def stairs(ep):\n    return ep.metadata['x']\n",
    "def stairs(self):\n    return self._episode.metadata\n",
])
def test_the_scanners_catch_a_read_of_the_episodes_metadata(source):
    assert list(episode_metadata_reads(ast.parse(source)))


def test_the_import_scan_follows_relative_lazy_and_from_imports_and_flags_a_computed_one():
    source = ("import os\n"
              "from . import sibling\n"
              "from ..upper import thing\n"
              "def lazy():\n"
              "    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat import simulator\n"
              "    import habitat_sim\n"
              "def computed(name):\n"
              "    from importlib import import_module\n"
              "    return import_module(name)\n")
    tree = ast.parse(source)
    names = {name for name, _ in imports_of(tree, ["sparx_agency", "core", "pkg", "sub"])}
    assert {"os", "sparx_agency.core.pkg.sub", "sparx_agency.core.pkg.sub.sibling", "sparx_agency.core.pkg.upper",
            "sparx_agency.core.pkg.upper.thing", "habitat_sim", "<dynamic>",
            "sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat",
            "sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator"} <= names
    assert not dynamic_import_is_anchored(tree), "import_module(name) can load anything"
    assert dynamic_import_is_anchored(ast.parse("def f(n):\n    return import_module('.' + n, __name__)\n"))
    # And the closure walker resolves a real benchmark module when one is named, so a leak would be seen.
    runtime = _module_file("sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator")
    assert runtime is not None and str(runtime.relative_to(REPO_ROOT)).startswith(BENCHMARK_PACKAGES)
    reached, third_party, _ = import_closure([runtime])
    assert "habitat_sim" in third_party, "the simulator bridge imports the simulator: the walker must see it"
