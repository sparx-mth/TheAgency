"""The config -> launch -> node contract: nothing set in mission.yaml may be dropped.

The mission's args funnel down three launch layers, each of which re-declares and
forwards a subset by hand::

    object_mission.launch  ->  real_drone.launch  ->  nav_stack.launch
                           ->  object_approach.launch

A parameter that is declared but never forwarded is the dangerous case: roslaunch
accepts it, the config validates, and the value is silently ignored -- the node keeps
its own default and the drone flies differently than the file says. These tests fail
instead.

Locked in here:
  * every key of mission.yaml's ``launch:`` section is declared by object_mission.launch
    AND actually used by it (forwarded to an include, or set as a node param);
  * for keys that ultimately belong to nav_stack (the A* / BEV / corrector / APF ones),
    the WHOLE chain forwards them, layer by layer;
  * a forwarded arg is always declared by the layer it is forwarded to;
  * object_mission's default matches the layer below, so the config's documented
    default is the one that actually applies.

Note the layers legitimately DISAGREE with each other: real_drone.launch deliberately
overrides some of nav_stack's defaults (los_smoothing, connectivity, the ESDF/APF shift
limits) because the real drone wants different values. So drift is asserted against the
NEAREST layer below, never against nav_stack directly.
"""
import pathlib
import re
import sys

import pytest

_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1]
_LAUNCH_DIR = _CONFIG_DIR.parent / "adapter" / "launch"
_OM = _LAUNCH_DIR / "object_mission.launch"
_RD = _LAUNCH_DIR / "real_drone.launch"
_NS = _LAUNCH_DIR / "nav_stack.launch"
_SHIPPED = _CONFIG_DIR / "mission.yaml"

sys.path.insert(0, str(_CONFIG_DIR))
import mission_config as mc  # noqa: E402


def _declared(path):
    """name -> default, first declaration winning (as roslaunch resolves it)."""
    out = {}
    for m in re.finditer(r'<arg name="([a-z_0-9]+)"\s+default="([^"]*)"', path.read_text()):
        out.setdefault(m.group(1), m.group(2))
    return out


def _effective(decl, value, _depth=0):
    """Resolve a default that is itself ``$(arg other)`` to the value that applies.

    An arg may take its default from another arg in the SAME file -- object_mission's
    bev_z_peak defaults to $(arg cruise_z), so the map's trust peak follows the flight
    altitude. The drift check below is about the value that actually reaches the drone,
    so chase that one level (and any further) before comparing. An unknown or cyclic
    reference is left as-is, which fails the comparison loudly rather than resolving to
    something invented.
    """
    m = re.fullmatch(r"\$\(arg ([a-z_0-9]+)\)", str(value).strip())
    if m is None or _depth > 4 or m.group(1) not in decl:
        return value
    return _effective(decl, decl[m.group(1)], _depth + 1)


def _forwarded(path, child):
    """The arg names ``path`` passes into its ``<include>`` of ``child``."""
    body = re.search(r'<include file="[^"]*%s".*?</include>' % child,
                     path.read_text(), re.S)
    return set(re.findall(r'<arg name="([a-z_0-9]+)"\s+value=', body.group(0))) if body else set()


def _config_launch_keys():
    _, args = mc.load(_SHIPPED, _OM)
    return [a.split(":=")[0] for a in args]


OM_DECL = _declared(_OM)
RD_DECL = _declared(_RD)
NS_DECL = _declared(_NS)
OM_TO_RD = _forwarded(_OM, "real_drone.launch")
RD_TO_NS = _forwarded(_RD, "nav_stack.launch")


@pytest.mark.parametrize("key", _config_launch_keys())
def test_every_config_key_is_declared_by_the_mission_launch(key):
    assert key in OM_DECL, "%s is in mission.yaml but object_mission.launch declares no such arg" % key


@pytest.mark.parametrize("key", _config_launch_keys())
def test_every_config_key_is_actually_used(key):
    """Declared but never referenced == silently ignored. That is the bug this catches."""
    uses = len(re.findall(r"\$\(arg %s\)" % re.escape(key), _OM.read_text()))
    assert uses > 0, "%s is declared but never used by object_mission.launch (value would be dropped)" % key


@pytest.mark.parametrize("key", [k for k in _config_launch_keys() if k in NS_DECL])
def test_nav_stack_keys_are_forwarded_the_whole_way_down(key):
    assert key in OM_TO_RD, "%s never reaches real_drone.launch" % key
    assert key in RD_DECL, "real_drone.launch does not declare %s" % key
    assert key in RD_TO_NS, "%s never reaches nav_stack.launch" % key


#: object_mission args that deliberately differ from real_drone's default, and why.
#: Anything NOT listed here must agree, so a default cannot drift in unnoticed.
INTENTIONAL_OVERRIDES = {
    # real_drone.launch made roll_assist its default in 16983c9, AFTER object_mission.launch
    # was written (a936611) pinning waypoint -- so the object mission does NOT pick up the
    # roll_assist default. Left as-is pending a decision; mission.yaml exposes `controller`,
    # so a run can select either. Remove this entry if the object mission should follow
    # real_drone's default instead.
    "controller": "predates real_drone's roll_assist default; see mission.yaml",
    # THE POINT of the GO gate: the object mission holds every command until GO, while
    # real_drone (and every launch that predates the gate) stays open so nothing that
    # used to fly silently stops flying.
    "start_go": "object mission requires an explicit GO; real_drone stays open",
}


@pytest.mark.parametrize("key", [k for k in _config_launch_keys() if k in NS_DECL])
def test_mission_default_matches_the_layer_below(key):
    """object_mission's default must equal real_drone's, so mission.yaml documents the
    value that actually applies. Compared against the NEAREST layer: real_drone
    intentionally overrides some of nav_stack's defaults. An object_mission default
    written as ``$(arg other)`` is compared by its EFFECTIVE value, so chaining one
    arg off another (bev_z_peak off cruise_z) is not drift."""
    if key in INTENTIONAL_OVERRIDES:
        pytest.skip("deliberate override: %s" % INTENTIONAL_OVERRIDES[key])
    om = _effective(OM_DECL, OM_DECL[key])
    assert om == RD_DECL[key], (
        "%s default drifted: object_mission=%r (effective %r) real_drone=%r"
        % (key, OM_DECL[key], om, RD_DECL[key]))


def test_forwarded_args_are_declared_by_the_child():
    """A forward to an arg the child never declares is dead wiring."""
    for name in sorted(OM_TO_RD):
        assert name in RD_DECL, "object_mission forwards %s, real_drone declares no such arg" % name
    for name in sorted(RD_TO_NS):
        assert name in NS_DECL, "real_drone forwards %s, nav_stack declares no such arg" % name


def test_the_nav_goal_reaches_the_planners():
    """goal_x/goal_y must be FORWARDED, not pinned to a literal.

    They used to be pinned empty, which idled the planners until an object was
    selected. They are now args, so the mission can also be a plain "fly to this
    coordinate" run. Pinning them again -- to '' or to a number -- would silently
    make the value unsettable: the config would still validate and still be ignored.
    """
    text = _OM.read_text()
    assert re.search(r'<arg name="goal_x"\s+value="\$\(arg goal_x\)" />', text)
    assert re.search(r'<arg name="goal_y"\s+value="\$\(arg goal_y\)" />', text)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_two_mission_points_are_wired_distinctly(axis):
    """The pre-selection target and the room centre are DISTINCT and correctly routed.

    Two points now, not one: goal_x/goal_y is the pre-selection target the drone flies
    to and HOLDS at before any object is picked; stage_x/stage_y is the room centre it
    works from after. The wiring that keeps them straight, locked down here:
      * the nav goal (goal_x/goal_y) is a plain literal, NOT chained to the staging
        point -- the two must be independently settable and genuinely different;
      * object_approach's arrival goal follows the STAGING point (so arriving there is
        "arrived at the room centre"), and the director publishes that same point;
      * the arm point that gates the blind visual take-over follows the PRE-SELECTION
        target, so the take-over is armed by reaching 0,-2, not 0,-3.
    Re-chaining the goal to the staging point, or the arm point to it, brings back the
    old single-point behaviour and fails here.
    """
    text = _OM.read_text()
    goal, stage = "goal_%s" % axis, "stage_%s" % axis
    keys = _config_launch_keys()
    assert goal in keys and stage in keys, \
        "mission.yaml must be able to set both %s and %s" % (goal, stage)
    # The nav goal is a decoupled literal, distinct from the staging point ...
    assert re.search(r'<arg name="%s"\s+default="-?[0-9]' % goal, text), \
        "the pre-selection target must be a literal, distinct from the staging point"
    assert not re.search(r'<arg name="%s"\s+default="\$\(arg %s\)"' % (goal, stage), text), \
        "the nav goal must NOT chain to the staging point (they are two distinct points)"
    # ... object_approach's arrival goal + the director both follow the STAGING point ...
    assert re.search(r'<arg name="%s"\s+value="\$\(arg %s\)" />' % (goal, stage), text), \
        "object_approach's arrival goal no longer follows the staging point"
    assert re.search(r'<param name="%s"\s+value="\$\(arg %s\)"' % (stage, stage), text), \
        "the mission director is not told the staging point"
    # ... and the arm point follows the PRE-SELECTION target (the room-entry point).
    assert re.search(r'<arg name="arm_point_%s"\s+value="\$\(arg %s\)" />' % (axis, goal), text), \
        "the arm point must follow the pre-selection target, not the room centre"


def test_the_go_gate_is_the_last_thing_holding_the_drone():
    """start_go must default CLOSED, and object_approach must start disabled.

    This is what the empty-goal pin used to back up. Now that goal_x/goal_y
    default to a real coordinate, A* has a route the moment the stack is up, so
    the GO gate is the ONLY thing between startup and a drone flying it -- there
    is no longer a second layer that also required an object selection first.
    Defaulting start_go true would therefore mean launching IS taking off.
    """
    assert OM_DECL["start_go"] == "false", (
        "start_go must default closed: with a numeric goal_x the planners have a "
        "route at startup, and this gate is all that holds the drone")
    assert re.search(r'<arg name="start_enabled"\s+value="false" />', _OM.read_text()), (
        "object_approach must still come up disabled, or the visual servo and the "
        "landing could fire before an object is ever selected")


# ── The launch files must actually PARSE ────────────────────────────
# Everything else in this file reads the launch XML with regexes, which happily
# match text that roslaunch itself cannot load. A launch file that does not parse
# takes the WHOLE stack down at startup -- and the easiest way to write one is a
# perfectly innocent-looking comment: XML forbids "--" inside <!-- -->, so a
# comment mentioning a command-line flag like "--foo" is a syntax error.
@pytest.mark.parametrize("launch", sorted(_LAUNCH_DIR.glob("*.launch")),
                         ids=lambda p: p.name)
def test_launch_file_is_well_formed_xml(launch):
    import xml.dom.minidom
    try:
        xml.dom.minidom.parse(str(launch))
    except Exception as e:  # noqa: BLE001 - any parse failure is the same bug
        pytest.fail(
            "%s is not well-formed XML, so roslaunch cannot load it and the whole "
            "stack fails to start:\n  %s\n"
            "If this mentions an invalid token, check for '--' inside an XML "
            "comment (e.g. writing a --flag name in prose)." % (launch.name, e))
