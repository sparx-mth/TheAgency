"""Taking a command apart and putting it back together without changing it."""
import pytest

from sparx_agency.tasks.common.launch_params.command import (parse, render,
                                                             render_template)
from sparx_agency.tasks.common.launch_params.spec import (ENV, ROS2, SLOT,
                                                          ParamSet, ParamSpec)


def roundtrip(text):
    """Parse then re-render with nothing changed."""
    parsed = parse(text)
    return render(parsed, ParamSet(parsed.params))


def test_ros2_parameters_are_recovered_and_restored():
    parsed = parse("python3 node.py --ros-args -p alpha:=0.2 -p topic:=/x")
    assert [(p.name, p.default, p.syntax) for p in parsed.params] == [
        ("alpha", "0.2", ROS2), ("topic", "/x", ROS2)]
    assert parsed.head == ["python3", "node.py", "--ros-args"]
    assert roundtrip("python3 node.py --ros-args -p alpha:=0.2 -p topic:=/x") == \
        "python3 node.py --ros-args -p alpha:=0.2 -p topic:=/x"


def test_argparse_options_are_recovered():
    parsed = parse("python3 pub.py --frequency 10.0 --out-dir /tmp/frames")
    assert [(p.name, p.default) for p in parsed.params] == [
        ("frequency", "10.0"), ("out-dir", "/tmp/frames")]


def test_a_remap_stays_a_remap_and_does_not_swallow_the_next_flag():
    """`-r a:=b` rewires a name; writing it as `-p` would set a dead parameter."""
    text = "ros2 run pkg node --ros-args -r __ns:=/foo -p k:=1"
    parsed = parse(text)
    assert [(p.name, p.syntax) for p in parsed.params] == [
        ("__ns", "ros2_remap"), ("k", ROS2)]
    assert render(parsed, ParamSet(parsed.params)) == \
        "ros2 run pkg node --ros-args -p k:=1 -r __ns:=/foo"


def test_the_long_remap_spelling_is_understood_too():
    parsed = parse("ros2 run p n --ros-args --remap cloud_in:=/points")
    assert [(p.name, p.value) for p in parsed.params] == [("cloud_in", "/points")]


def test_an_option_before_ros_args_stays_before_it():
    """Re-emitting it after the marker would put it inside the ros-args region,
    where rcl rejects it -- and would leave a second --ros-args behind."""
    text = "ros2 run pkg node --my-flag v --ros-args -p k:=1"
    assert roundtrip(text) == text


def test_a_negative_number_is_a_value_not_another_option():
    """`--roll -1.5708` must not read as a bare flag followed by junk."""
    parsed = parse("ros2 run tf2_ros stp --roll -1.5708 --frame-id map")
    assert [(p.name, p.default) for p in parsed.params] == [
        ("roll", "-1.5708"), ("frame-id", "map")]


def test_a_bare_flag_becomes_an_on_off_parameter():
    parsed = parse("python3 pub.py --no-clear-on-start --frequency 5")
    flag = parsed.params[0]
    assert (flag.name, flag.default, flag.tokens()) == (
        "no-clear-on-start", "on", ["--no-clear-on-start"])


def test_a_leading_assignment_is_an_environment_override():
    parsed = parse("NAV_MODE=hybrid ./run.sh")
    assert [(p.name, p.syntax) for p in parsed.params] == [("NAV_MODE", ENV)]
    assert render(parsed, ParamSet(parsed.params)) == "NAV_MODE=hybrid ./run.sh"


def test_a_setup_line_before_the_command_is_left_alone():
    parsed = parse('LOG_PATH="/tmp/x.jsonl"\npython3 replay.py --topic /cmd_vel')
    assert parsed.preamble == ['LOG_PATH="/tmp/x.jsonl"']
    assert [p.name for p in parsed.params] == ["topic"]


def test_line_continuations_are_joined_before_parsing():
    parsed = parse("python3 node.py \\\n  --ros-args \\\n  -p a:=1")
    assert [p.name for p in parsed.params] == ["a"]


def test_remaps_are_emitted_after_the_parameters_in_the_short_form():
    """Both spellings normalise to `-r`, which is what ROS2's own docs use."""
    assert roundtrip("ros2 run p n --ros-args -p a:=1 --remap in:=/topic") == \
        "ros2 run p n --ros-args -p a:=1 -r in:=/topic"


def test_a_discovered_parameter_gets_the_ros_args_marker_it_needs():
    """Without it, ROS2 reads -p as the program's own argument."""
    parsed = parse("python3 node.py")
    params = ParamSet(parsed.params)
    params.add(ParamSpec("alpha", "0.8", syntax=ROS2)).value = "0.3"
    assert render(parsed, params) == "python3 node.py --ros-args -p alpha:=0.3"


def test_unbalanced_quoting_is_left_intact_rather_than_mangled():
    text = "echo 'unterminated"
    assert render(parse(text), ParamSet()) == text


def test_a_value_with_a_space_or_a_quote_reaches_the_program_intact():
    """Rendered commands are run by a shell, so this is a correctness question.

    Checked by actually running one: a path with a space that arrives as two
    arguments is a node that dies on a file it was never given.
    """
    import subprocess

    parsed = parse("run_me --ros-args -p tag_map:=/tmp/a.yaml")
    params = ParamSet(parsed.params)
    params["tag_map"].value = "/tmp/my maps/new map.yaml"
    params.add(ParamSpec("family", "x", syntax=ROS2)).value = "tag36h11'x"

    rendered = render(parsed, params).replace("run_me", "printf '[%s]\\n'", 1)
    seen = subprocess.run(["bash", "-c", rendered], capture_output=True,
                          text=True, check=True).stdout
    assert "[tag_map:=/tmp/my maps/new map.yaml]" in seen
    assert "[family:=tag36h11'x]" in seen


def test_an_emptied_value_still_renders_a_legal_assignment():
    """`stage_x:=` is how the mission documents switching staging off."""
    params = ParamSet([ParamSpec("stage_x", "0.0", syntax="roslaunch")])
    params["stage_x"].value = ""
    assert render_template("./run.sh {params}", params) == "./run.sh stage_x:="


def test_a_template_fills_its_named_slots_always():
    params = ParamSet([ParamSpec("container", "falcon", syntax=SLOT)])
    assert render_template("docker exec -it {container} bash", params) == \
        "docker exec -it falcon bash"


def test_a_template_splits_environment_from_the_rest():
    params = ParamSet([
        ParamSpec("MAP", "office", syntax=ENV),
        ParamSpec("NAV_MODE", "astar", syntax=ENV),
        ParamSpec("vel_x", "0.25"),
    ])
    params["NAV_MODE"].value = "hybrid"
    params["vel_x"].value = "0.3"
    params["vel_x"].syntax = "roslaunch"
    assert render_template("{env}./run.sh {MAP} {params}", params) == \
        "NAV_MODE=hybrid ./run.sh office vel_x:=0.3"


def test_a_named_slot_is_not_also_appended_to_the_pile():
    params = ParamSet([ParamSpec("MAP", "office", syntax=ENV)])
    params["MAP"].value = "hospital"
    assert render_template("{env}./run.sh {MAP} {params}", params) == \
        "./run.sh hospital"


def test_a_template_slot_with_no_parameter_says_which_one():
    with pytest.raises(KeyError, match="missing_one"):
        render_template("run {missing_one}", ParamSet())
