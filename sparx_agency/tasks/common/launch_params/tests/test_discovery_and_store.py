"""Layering several sources over one command, and remembering the result."""
import pytest

from sparx_agency.tasks.common.launch_params.discovery import Source, discover
from sparx_agency.tasks.common.launch_params.store import ParamStore


def test_an_unknown_source_kind_is_refused_at_declaration():
    with pytest.raises(ValueError, match="unknown parameter source kind"):
        Source("rosparam", "some/file.yaml")


def test_a_source_that_cannot_be_read_is_reported_not_swallowed():
    """One moved file must not stop the other fifteen commands from starting."""
    found = discover((Source("ros2_node", "nowhere/at/all.py"),))
    assert len(found.params) == 0
    assert len(found.problems) == 1
    assert "nowhere/at/all.py" in found.problems[0]


def test_the_real_mission_sources_layer_the_way_the_script_reads_them():
    """mission.yaml decides the defaults; the launch file adds the rest."""
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.falcon_items import (
        MISSION_SOURCES)

    found = discover(MISSION_SOURCES)
    assert not found.problems
    assert len(found.params) > 250, "the object mission's full knob set"

    # A value set in mission.yaml is what a plain run uses, so it is the default.
    assert found.params["controller"].default == "waypoint", \
        "real_drone.launch defaults to roll_assist; the mission config pins waypoint"
    # The env-mapped keys keep the shell names run_object_mission.sh reads.
    assert found.params["NAV_MODE"].syntax == "env"
    # Everything arrives explained.
    documented = sum(1 for p in found.params if p.doc)
    assert documented > len(found.params) * 0.8


def test_every_offered_mission_parameter_is_one_roslaunch_would_accept():
    """The launcher must not offer a knob that makes roslaunch refuse to start.

    mission_config.py already rejects an undeclared key in the YAML; this is the
    same guarantee for the keys the launcher itself can put on the command line.
    """
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.falcon_items import (
        MISSION_LAUNCH, MISSION_SOURCES)
    from sparx_agency.tasks.common.launch_params.discovery import REPO_ROOT
    from sparx_agency.tasks.common.launch_params.sources import roslaunch_xml

    declared = {p.name for p in roslaunch_xml.discover(REPO_ROOT / MISSION_LAUNCH)}
    offered = {p.name for p in discover(MISSION_SOURCES).params
               if p.syntax == "roslaunch"}
    assert offered <= declared, \
        "not declared by object_mission.launch: %s" % sorted(offered - declared)


def test_the_mission_environment_overrides_are_the_ones_the_script_reads():
    """An env name the script does not read would be silently ignored."""
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.falcon_items import (
        MISSION_SOURCES)

    offered = {p.name for p in discover(MISSION_SOURCES).params if p.syntax == "env"}
    assert offered == {"MAP", "SELECTION_MODE", "NAV_MODE", "SEED", "MODEL",
                       "WEIGHTS_DIR", "CONF_THRESH", "INIT_TARGET", "OBJECTS_DIR"}


def test_the_mission_offers_one_knob_per_decision_not_two():
    """The bug this guards: object_mission.launch declares nav_mode (default
    'fallback') and mission.yaml sets NAV_MODE (default 'astar'). Offering both
    let an operator select 'fallback', have it match the shown default so it was
    never emitted, and fly astar -- with no NavDP rescue -- believing otherwise.
    """
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.falcon_items import (
        MISSION_SOURCES, SCRIPT_OWNED_LAUNCH_ARGS)

    params = discover(MISSION_SOURCES).params
    for name in SCRIPT_OWNED_LAUNCH_ARGS:
        assert name not in params, \
            "%s is set by run_object_mission.sh itself and must not be offered" % name
    assert params["NAV_MODE"].default == "astar", "what a plain run really flies"


def test_a_declared_parameter_is_not_shared_between_two_commands():
    """The catalog reuses ParamSpec constants; a plan must copy, not alias."""
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.item import LaunchPlan
    from sparx_agency.demos.Demo_No4_XTEND_MapRoom.launcher.items import LAUNCH_ITEMS

    def plan(key):
        return LaunchPlan.build(next(i for i in LAUNCH_ITEMS if i.tmux_name == key))

    rviz, viewer = plan("falcon_rviz"), plan("falcon_bev_goal")
    rviz.params["container"].value = "falcon_dev"
    assert viewer.params["container"].value == "falcon"
    assert viewer.params.as_dict() == {}, "and it saves no override it never made"


def test_the_store_round_trips_and_drops_an_emptied_entry(tmp_path):
    store = ParamStore(tmp_path / "params.json")
    store.put("falcon_mission", {"NAV_MODE": "hybrid"})
    assert ParamStore(tmp_path / "params.json").get("falcon_mission") == \
        {"NAV_MODE": "hybrid"}

    store.put("falcon_mission", {})
    assert ParamStore(tmp_path / "params.json").keys() == []


def test_an_absent_store_is_simply_empty(tmp_path):
    assert ParamStore(tmp_path / "never-written.json").keys() == []


def test_a_corrupt_store_stops_rather_than_discarding_every_override(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="cannot read saved parameters"):
        ParamStore(path)
