"""Behavioural tests for mission_director_node (select the object, then arm the mission).

The node is a ROS1 adapter, so ``rospy`` + the message packages are stubbed in
``sys.modules`` before it is imported -- but the CATALOG path is the real, unit-tested
``core.planning.mission.ObjectCatalog`` loaded from the real ``objects.json`` shipped next
to this task, so these tests exercise the actual selection -> publish wiring end to end
(without a display: the matplotlib GUI is imported lazily inside ``_run_gui`` and never
touched here).

The contract locked in here:
  * THE GATE -- at startup nothing but a disarming enable=False is published (no goal, no
    target), so nothing plans or flies until a selection;
  * selecting an object publishes its LABEL (-> YOLO + gate), its (x, y) as the coordinate
    goal (-> planners), and enable=True (-> object_approach), with x/y mapped to the exact
    chosen object (two same-labelled 'chair' rows have distinct goals);
  * ~publish_enable=false suppresses every enable publication.
"""
import pathlib
import sys
import types

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _Pub:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)

    @property
    def last(self):
        return self.msgs[-1] if self.msgs else None


_PARAMS = {}


class _Time:
    """Minimal rospy.Time: the node stamps its thinking narration off this clock."""

    t = 0.0

    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    @staticmethod
    def now():
        return _Time(_Time.t)


class _Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _String:
    def __init__(self, data=""):
        self.data = data


class _Bool:
    def __init__(self, data=False):
        self.data = data


def _install_stubs():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
    rospy.Publisher = lambda topic, *a, **k: _Pub(topic)
    rospy.Subscriber = lambda *a, **k: None
    rospy.Timer = lambda *a, **k: None
    rospy.Time = _Time
    rospy.spin = lambda: None
    rospy.sleep = lambda *a, **k: None
    rospy.signal_shutdown = lambda *a, **k: None
    rospy.on_shutdown = lambda *a, **k: None
    rospy.is_shutdown = lambda: True          # so _run_random would not loop
    for fn in ("loginfo", "logwarn", "logerr", "logfatal"):
        setattr(rospy, fn, lambda *a, **k: None)
    rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
    sys.modules["rospy"] = rospy

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    geo = _mod("geometry_msgs")
    geo.msg = _mod("geometry_msgs.msg", Point=_Point)
    std = _mod("std_msgs")
    std.msg = _mod("std_msgs.msg", Bool=_Bool, String=_String)


_install_stubs()
import mission_director_node as mdn  # noqa: E402


def _make(**params):
    _PARAMS.clear()
    _PARAMS.update({"~selection_mode": "random"})   # never opens the GUI
    _PARAMS.update({("~" + k): v for k, v in params.items()})
    return mdn.MissionDirectorNode()


# ── the gate ───────────────────────────────────────────────────────────
def test_gate_publishes_only_disarm_at_startup():
    node = _make()
    # No goal, no target published before a selection.
    assert node.goal_pub.msgs == []
    assert node.target_pub.msgs == []
    # object_approach is actively held disabled.
    assert isinstance(node.enable_pub.last, _Bool)
    assert node.enable_pub.last.data is False


def test_catalog_loads_the_shipped_objects():
    node = _make()
    assert len(node.catalog) >= 12
    assert "refrigerator" in node.catalog.unique_labels()


# ── selection -> publish mapping ───────────────────────────────────────
def test_select_publishes_label_goal_and_enable():
    node = _make()
    obj = node.catalog.by_label("refrigerator")[0]
    node._select(obj)
    assert node.target_pub.last.data == "refrigerator"
    assert node.goal_pub.last.x == obj.x
    assert node.goal_pub.last.y == obj.y
    assert node.goal_pub.last.z == 0.0
    assert node.enable_pub.last.data is True


def test_duplicate_label_selects_the_specific_object_goal():
    # objects.json has two 'chair' rows -- the goal must be the CHOSEN one's (x, y),
    # while the label sent to YOLO is the bare 'chair' for both.
    node = _make()
    chairs = node.catalog.by_label("chair")
    assert len(chairs) == 2
    for chair in chairs:
        node._select(chair)
        assert node.target_pub.last.data == "chair"
        assert (node.goal_pub.last.x, node.goal_pub.last.y) == (chair.x, chair.y)


def test_retarget_republishes_new_goal_and_label():
    node = _make()
    a, b = node.catalog[0], node.catalog[1]
    node._select(a)
    node._select(b)
    assert node.target_pub.last.data == b.label
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (b.x, b.y)
    # every selection re-publishes all three (2 selections after the startup gate).
    assert len(node.goal_pub.msgs) == 2


# ── publish_enable gate ────────────────────────────────────────────────
def test_publish_enable_false_never_touches_enable():
    node = _make(publish_enable=False)
    assert node.enable_pub.msgs == []          # no startup disarm either
    node._select(node.catalog[0])
    assert node.enable_pub.msgs == []          # and no enable on selection
    assert node.target_pub.last.data == node.catalog[0].label


# ── GUI selection path (fake matplotlib artists, no display) ───────────
class _FakeArtist:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeAx:
    def __init__(self):
        self.spans = []

    def axhspan(self, lo, hi, **kw):
        a = _FakeArtist()
        self.spans.append((lo, hi, a))
        return a


class _FakeCanvas:
    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


class _FakeFig:
    def __init__(self):
        self.canvas = _FakeCanvas()


class _FakeText:
    def __init__(self):
        self.text = ""

    def set_text(self, s):
        self.text = s


class _Event:
    def __init__(self, inaxes=None, button=1, xdata=0.0, ydata=0.0, key=None):
        self.inaxes = inaxes
        self.button = button
        self.xdata = xdata
        self.ydata = ydata
        self.key = key


def _gui_node(**params):
    """A node with the GUI artists faked in, so click/key handlers run headless."""
    node = _make(**params)
    node.ax = _FakeAx()
    node.fig = _FakeFig()
    node._highlight = None
    node._status_text = _FakeText()
    return node


def test_click_selects_the_row_under_the_cursor():
    node = _gui_node()
    # invert_yaxis does NOT change data coords: ydata ~= the row index. Row 3's band is
    # [2.5, 3.5), so a click at ydata 2.6..3.4 must select catalog[3].
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=3.4))
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (node.catalog[3].x, node.catalog[3].y)
    assert node.target_pub.last.data == node.catalog[3].label
    # a highlight band was drawn around row 3
    assert node.ax.spans[-1][:2] == (2.5, 3.5)


def test_click_rounds_to_nearest_row():
    node = _gui_node()
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=2.6))   # -> row 3
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (node.catalog[3].x, node.catalog[3].y)
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=2.4))   # -> row 2
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (node.catalog[2].x, node.catalog[2].y)


def test_click_ignores_other_buttons_axes_and_empty():
    node = _gui_node()
    node._on_click(_Event(inaxes=node.ax, button=3, ydata=1.0))       # right-click
    node._on_click(_Event(inaxes=None, button=1, ydata=1.0))          # outside axes
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=None))      # off the canvas
    assert node.goal_pub.msgs == []
    assert node.target_pub.msgs == []


def test_number_key_selects_that_object():
    node = _gui_node()
    node._on_key(_Event(key="2"))                     # 1-based -> catalog[1]
    assert node.target_pub.last.data == node.catalog[1].label
    node._on_key(_Event(key="kp_5"))                  # numeric keypad -> catalog[4]
    assert node.target_pub.last.data == node.catalog[4].label


def test_zero_and_out_of_range_keys_are_ignored():
    node = _gui_node()
    node._on_key(_Event(key="0"))                     # no object 0
    node._on_key(_Event(key="x"))                     # not a selector
    assert node.target_pub.msgs == []


def test_r_key_selects_a_valid_object():
    node = _gui_node()
    node._on_key(_Event(key="r"))
    assert node.target_pub.last.data in node.catalog.labels()
    assert node.enable_pub.last.data is True


# ── config validation ──────────────────────────────────────────────────
def test_bad_selection_mode_raises():
    _PARAMS.clear()
    _PARAMS.update({"~selection_mode": "joystick"})
    try:
        mdn.MissionDirectorNode()
    except ValueError:
        return
    raise AssertionError("expected ValueError for a bad ~selection_mode")


# ── The GO gate (the director is the button; the gate is the mechanism) ──────
def test_startup_publishes_no_go_so_the_launch_owns_the_initial_state():
    """The gate's ~start_go decides whether commands flow at boot. If the director
    published a GO/HOLD just by opening, it would override the launch."""
    node = _make()
    assert node.go_pub.msgs == []
    assert node._go is None


def test_selecting_an_object_does_not_open_the_gate():
    """Arming the mission and letting it move are deliberately separate acts."""
    node = _gui_node()
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=0.0))
    assert node.target_pub.last is not None, "selection should still arm"
    assert node.go_pub.msgs == [], "selection must NOT let commands through"


def test_g_key_opens_the_gate():
    node = _gui_node()
    node._on_key(_Event(key="g"))
    assert node.go_pub.last.data is True


def test_h_key_closes_the_gate():
    node = _gui_node()
    node._on_key(_Event(key="g"))
    node._on_key(_Event(key="h"))
    assert node.go_pub.last.data is False


def test_go_is_idempotent():
    node = _gui_node()
    node._on_key(_Event(key="g"))
    node._on_key(_Event(key="g"))
    assert len(node.go_pub.msgs) == 1


def test_go_shows_in_the_status_caption():
    node = _gui_node()
    node._on_key(_Event(key="g"))
    assert "GO" in node._status_text.text
    node._on_key(_Event(key="h"))
    assert "HOLD" in node._status_text.text


def test_go_key_does_not_disturb_the_selection():
    node = _gui_node()
    node._on_click(_Event(inaxes=node.ax, button=1, ydata=2.0))
    label = node.target_pub.last.data
    node._on_key(_Event(key="g"))
    assert node.target_pub.last.data == label


# ── the staging vantage point ──────────────────────────────────────────
# With ~stage_x/~stage_y set, the goal handed to the planners is the VANTAGE POINT,
# not the object: the drone flies there and looks at the object from a standoff
# rather than trusting the room map's coordinate enough to fly onto it. The object's
# own position always goes out separately, which is what object_approach aims at (and
# falls back to). Without them, nothing changes -- the goal is the object, as before.
def test_staging_publishes_the_vantage_point_as_the_goal():
    node = _make(stage_x=0.0, stage_y=-2.0)
    obj = node.catalog.by_label("refrigerator")[0]
    node._select(obj)
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (0.0, -2.0)
    assert (node.goal_pub.last.x, node.goal_pub.last.y) != (obj.x, obj.y)


def test_staging_still_publishes_the_object_position():
    """The object is not hidden: it is what object_approach turns to look at."""
    node = _make(stage_x=0.0, stage_y=-2.0)
    obj = node.catalog.by_label("refrigerator")[0]
    node._select(obj)
    assert (node.object_pos_pub.last.x, node.object_pos_pub.last.y) == (obj.x, obj.y)
    assert node.object_pos_pub.last.z == 0.0


def test_object_position_is_published_before_the_goal():
    """Order matters across the bridge: object_approach must know the goal is only a
    staging point BY THE TIME it learns the goal, or the first arrival could be read
    as arriving at the object -- and with land_at_goal, landed on."""
    node = _make(stage_x=0.0, stage_y=-2.0)
    order = []
    node.object_pos_pub.publish = lambda m, _o=order: _o.append("object")
    node.goal_pub.publish = lambda m, _o=order: _o.append("goal")
    node._select(node.catalog[0])
    assert order == ["object", "goal"]


def test_without_staging_the_goal_is_the_object_as_before():
    node = _make()
    obj = node.catalog.by_label("refrigerator")[0]
    node._select(obj)
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (obj.x, obj.y)
    # ...and the object position is published anyway, so aiming still applies if
    # object_approach is flying to some other goal.
    assert (node.object_pos_pub.last.x, node.object_pos_pub.last.y) == (obj.x, obj.y)


def test_staging_retarget_republishes_both():
    node = _make(stage_x=1.0, stage_y=2.0)
    a, b = node.catalog[0], node.catalog[1]
    node._select(a)
    node._select(b)
    assert (node.object_pos_pub.last.x, node.object_pos_pub.last.y) == (b.x, b.y)
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (1.0, 2.0)  # unchanged
    assert len(node.object_pos_pub.msgs) == 2


def test_empty_stage_params_mean_no_staging():
    """roslaunch's way of saying "unset" is an empty string, not a missing param."""
    node = _make(stage_x="", stage_y="")
    obj = node.catalog[0]
    node._select(obj)
    assert node.stage_xy is None
    assert (node.goal_pub.last.x, node.goal_pub.last.y) == (obj.x, obj.y)


def test_half_a_staging_point_is_refused():
    """Setting only one axis is a mistake, not half a vantage point -- flying to a
    guessed coordinate is worse than not staging at all."""
    import pytest
    with pytest.raises(ValueError):
        _make(stage_x=0.0)
    with pytest.raises(ValueError):
        _make(stage_y=-2.0)


def test_staging_point_appears_in_the_status_line():
    node = _make(stage_x=0.0, stage_y=-2.0)
    node._select(node.catalog.by_label("refrigerator")[0])
    line = node.status_pub.last.data
    assert "vantage" in line and "refrigerator" in line
