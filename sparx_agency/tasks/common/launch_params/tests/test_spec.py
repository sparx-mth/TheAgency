"""The parameter model: how a value is written out, and how sources layer."""
import pytest

from sparx_agency.tasks.common.launch_params.spec import (CLI, ENV, FLAG, ROS2,
                                                          ROSLAUNCH, SLOT,
                                                          ParamSet, ParamSpec)


def test_each_syntax_writes_its_own_shape():
    assert ParamSpec("vel_x", "0.2", syntax=ROS2).tokens() == ["-p", "vel_x:=0.2"]
    assert ParamSpec("vel_x", "0.2", syntax=ROSLAUNCH).tokens() == ["vel_x:=0.2"]
    assert ParamSpec("out-dir", "/tmp", syntax=CLI).tokens() == ["--out-dir", "/tmp"]
    assert ParamSpec("NAV_MODE", "hybrid", syntax=ENV).tokens() == ["NAV_MODE=hybrid"]


def test_a_flag_is_spelled_by_its_presence():
    flag = ParamSpec("verbose", "off", syntax=FLAG)
    assert flag.tokens() == []
    flag.value = "on"
    assert flag.tokens() == ["--verbose"]


def test_a_slot_never_becomes_an_option():
    """A slot is substituted into the command's shape, not appended to it."""
    slot = ParamSpec("container", "falcon", syntax=SLOT)
    slot.value = "other"
    assert slot.tokens() == []


def test_an_unknown_syntax_is_refused():
    with pytest.raises(ValueError, match="unknown parameter syntax"):
        ParamSpec("x", "1", syntax="yaml-ish")


def test_only_changed_or_pinned_parameters_render():
    """The point of a 300-knob config: state the few you are changing."""
    params = ParamSet([
        ParamSpec("untouched", "1", syntax=ROSLAUNCH),
        ParamSpec("moved", "1", syntax=ROSLAUNCH),
        ParamSpec("spelled_out", "1", syntax=ROSLAUNCH, pinned=True),
    ])
    params["moved"].value = "2"
    assert [p.name for p in params.rendered()] == ["moved", "spelled_out"]


def test_a_later_source_restates_the_default_and_the_value_follows():
    """The command's own value IS the default a plain start runs with."""
    params = ParamSet([ParamSpec("alpha", "0.8", doc="node default")])
    params.add(ParamSpec("alpha", "0.2", pinned=True))
    assert (params["alpha"].default, params["alpha"].value) == ("0.2", "0.2")
    assert params["alpha"].doc == "node default", "documentation must survive"
    assert not params["alpha"].changed


def test_an_operators_edit_survives_a_later_source():
    params = ParamSet([ParamSpec("alpha", "0.8")])
    params["alpha"].value = "0.5"
    params.add(ParamSpec("alpha", "0.2"))
    assert params["alpha"].value == "0.5"


def test_a_source_that_does_not_define_defaults_only_adds():
    """A launch file read after the config that overrides it must not win.

    Otherwise "reset" returns to a built-in no run ever uses.
    """
    params = ParamSet([ParamSpec("vel_x", "0.25", doc="from mission.yaml")])
    params.extend([ParamSpec("vel_x", "0.4"), ParamSpec("only_in_launch", "7")],
                  override_default=False)
    assert params["vel_x"].default == "0.25"
    assert "only_in_launch" in params


def test_reset_returns_every_parameter_to_its_default():
    params = ParamSet([ParamSpec("a", "1"), ParamSpec("b", "2")])
    params["a"].value, params["b"].value = "9", "9"
    params.reset()
    assert [p.value for p in params] == ["1", "2"]
    assert params.changed() == []


def test_saved_values_that_match_nothing_are_reported_not_dropped():
    """A renamed parameter must not silently stop being applied."""
    params = ParamSet([ParamSpec("kept", "1")])
    unknown = params.apply({"kept": "5", "gone": "9"})
    assert unknown == ["gone"] and params["kept"].value == "5"


def test_only_changed_values_are_persisted():
    params = ParamSet([ParamSpec("a", "1"), ParamSpec("b", "2")])
    params["b"].value = "3"
    assert params.as_dict() == {"b": "3"}


def test_sections_keep_declaration_order():
    params = ParamSet([ParamSpec("a", "1", section="Two"),
                       ParamSpec("b", "1", section="One"),
                       ParamSpec("c", "1", section="Two")])
    assert params.sections() == ["Two", "One"]
