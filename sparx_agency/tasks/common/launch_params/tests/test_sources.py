"""Recovering a parameter set, with its documentation, from each source style."""
import pytest

from sparx_agency.tasks.common.launch_params.sources import (argparse_cli,
                                                             ros2_node,
                                                             roslaunch_xml,
                                                             yaml_config)
from sparx_agency.tasks.common.launch_params.spec import CLI, ENV, FLAG, ROSLAUNCH

NODE = '''
class Node:
    def __init__(self):
        self.declare_parameter("provider_type", "apriltag")
        # apriltag params
        self.declare_parameter("tag_size_m", 0.13)
        # Filter gain at full confidence. 1.0 publishes
        # the raw solve.
        self.declare_parameter("alpha", 0.8)
        self.declare_parameter("roi_rescue", True)
        self.declare_parameter("timeout", 5)  # seconds before it gives up
        self.declare_parameter(NOT_A_LITERAL, 1)
'''

SCRIPT = '''
import argparse
p = argparse.ArgumentParser()
p.add_argument("positional")
p.add_argument("-s")
p.add_argument("--frequency", type=float, default=10.0, help="frames per second")
p.add_argument("--backend", choices=["ffmpeg", "gstreamer"], default="gstreamer")
p.add_argument("--no-clear-on-start", action="store_true", help="keep old frames")
'''

LAUNCH = """<launch>
  <!-- ── Shared / nav ──────────── -->
  <arg name="map_name" default="office" />
  <arg name="real_pose_type" default="pose_stamped" /><!-- pose_stamped | pose -->
  <!-- ── The staging point ──────────
       Where the drone stands to look for the object. -->
  <arg name="stage_x" default="0.0" />
  <!-- How close counts as arrived. -->
  <arg name="arrive_radius_m" default="0.6" />
  <include file="child.launch"><arg name="map_name" value="$(arg map_name)" /></include>
</launch>
"""

CONFIG = """
# ==========================
# THE MISSION ITSELF
# ==========================
mission:
  map: office                 # maps/<map>.yaml
  # Which planner flies the route:
  #   astar : pure A* only
  nav_mode: astar
launch:
  # -- The AIM: how it turns to look -------
  aim_look_s: 4.0             # hold still this long
  viewer: false               # target-lock HUD window
"""


def written(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_ros2_node_parameters_carry_their_comments(tmp_path):
    found = {p.name: p for p in ros2_node.discover(written(tmp_path, "n.py", NODE))}
    assert found["tag_size_m"].default == "0.13"
    assert found["roi_rescue"].default == "true", "ROS2 parses values as YAML"
    assert found["roi_rescue"].choices == ("true", "false")
    assert found["alpha"].doc == "Filter gain at full confidence. 1.0 publishes the raw solve."
    assert found["timeout"].doc == "seconds before it gives up"


def test_a_heading_groups_every_parameter_after_it(tmp_path):
    """Not just the first: the rest are separated from it by lines of code."""
    found = {p.name: p for p in ros2_node.discover(written(tmp_path, "n.py", NODE))}
    assert found["provider_type"].section == ""
    assert [found[n].section for n in ("tag_size_m", "alpha", "roi_rescue")] == \
        ["apriltag params"] * 3


def test_a_non_literal_declaration_is_skipped(tmp_path):
    """The editor cannot round-trip a value it cannot read, so it must not try."""
    names = [p.name for p in ros2_node.discover(written(tmp_path, "n.py", NODE))]
    assert len(names) == 5 and "NOT_A_LITERAL" not in names


def test_argparse_options_carry_help_and_choices(tmp_path):
    found = {p.name: p for p in argparse_cli.discover(written(tmp_path, "s.py", SCRIPT))}
    assert set(found) == {"frequency", "backend", "no-clear-on-start"}, \
        "positional and short-only arguments are not editable options"
    assert (found["frequency"].default, found["frequency"].syntax) == ("10.0", CLI)
    assert found["frequency"].doc == "frames per second"
    assert found["backend"].choices == ("ffmpeg", "gstreamer")
    assert (found["no-clear-on-start"].syntax, found["no-clear-on-start"].default) == \
        (FLAG, "off")


def test_launch_arguments_carry_their_banners_and_notes(tmp_path):
    found = {p.name: p for p in roslaunch_xml.discover(written(tmp_path, "a.launch", LAUNCH))}
    assert found["map_name"].section == "Shared / nav"
    assert found["real_pose_type"].doc == "pose_stamped | pose", \
        "a comment closing on the arg's own line documents that arg"
    assert found["arrive_radius_m"].doc == "How close counts as arrived."
    assert found["stage_x"].syntax == ROSLAUNCH


def test_a_banner_titles_the_section_without_swallowing_its_prose(tmp_path):
    found = {p.name: p for p in roslaunch_xml.discover(written(tmp_path, "a.launch", LAUNCH))}
    assert found["stage_x"].section == "The staging point"
    assert "Where the drone stands" in found["stage_x"].detail


def test_an_arg_passed_to_an_include_is_not_a_knob(tmp_path):
    """It has a value, not a default: this file is passing it down, not offering it."""
    found = roslaunch_xml.discover(written(tmp_path, "a.launch", LAUNCH))
    assert len(found) == 4


def test_yaml_settings_carry_notes_sections_and_syntax(tmp_path):
    schema = {"mission": {"map": "MAP", "nav_mode": "NAV_MODE"}}
    found = {p.name: p for p in
             yaml_config.discover(written(tmp_path, "m.yaml", CONFIG), schema)}
    assert (found["MAP"].syntax, found["MAP"].default) == (ENV, "office")
    assert found["MAP"].doc == "maps/<map>.yaml"
    assert found["aim_look_s"].syntax == ROSLAUNCH
    assert found["aim_look_s"].section == "The AIM: how it turns to look"
    assert found["viewer"].default == "false"


def test_a_key_with_no_note_is_explained_by_the_paragraph_above_it(tmp_path):
    found = {p.name: p for p in yaml_config.discover(written(tmp_path, "m.yaml", CONFIG),
                                                     {"mission": {"nav_mode": "NAV_MODE"}})}
    assert found["NAV_MODE"].doc == "Which planner flies the route: astar : pure A* only"


def test_a_group_filter_keeps_a_shared_config_relevant(tmp_path):
    """The detector sidecar must not be offered 300 flight parameters."""
    found = yaml_config.discover(written(tmp_path, "m.yaml", CONFIG),
                                 None, only_groups=("mission",))
    assert [p.name for p in found] == ["map", "nav_mode"]


def test_an_unreadable_source_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(OSError):
        ros2_node.discover(tmp_path / "does_not_exist.py")
