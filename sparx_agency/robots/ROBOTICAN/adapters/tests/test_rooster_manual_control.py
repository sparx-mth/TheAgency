"""The extracted Rooster actuation, driven against stub ROS types.

``rclpy`` and the Robotican interface packages exist only inside the Sphera
container, so this test installs minimal stubs and exercises the logic that used
to be copy-pasted into three VLA bridges: the hold-then-stop latch, the arm gate,
the heartbeat, and axis clamping.

Those behaviours are what actually drive a physical robot, and none of them had a
test before the consolidation.
"""
from __future__ import annotations

import sys
import types

import pytest


# ── stub ROS message packages before importing the adapter ───────────────
def _install_ros_stubs(monkeypatch):
    def _msg_module(name, cls_names):
        mod = types.ModuleType(name)
        for cls_name in cls_names:
            setattr(mod, cls_name, type(cls_name, (), {
                "__init__": lambda self: None,
            }))
        return mod

    class _Header:
        def __init__(self):
            self.stamp = None

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Header = _Header
    std_msgs.msg = std_msgs_msg

    fcu = types.ModuleType("fcu_driver_interfaces")
    fcu_msg = _msg_module("fcu_driver_interfaces.msg", ["ManualControl", "UAVState"])
    fcu.msg = fcu_msg

    handler = types.ModuleType("rooster_handler_interfaces")
    handler_msg = _msg_module("rooster_handler_interfaces.msg", ["KeepAlive"])
    handler.msg = handler_msg

    class _SetBool:
        class Request:
            def __init__(self):
                self.data = None

    srv = types.ModuleType("std_srvs")
    srv_msg = types.ModuleType("std_srvs.srv")
    srv_msg.SetBool = _SetBool
    srv.srv = srv_msg

    for name, mod in [("std_msgs", std_msgs), ("std_msgs.msg", std_msgs_msg),
                      ("fcu_driver_interfaces", fcu),
                      ("fcu_driver_interfaces.msg", fcu_msg),
                      ("rooster_handler_interfaces", handler),
                      ("rooster_handler_interfaces.msg", handler_msg),
                      ("std_srvs", srv), ("std_srvs.srv", srv_msg)]:
        monkeypatch.setitem(sys.modules, name, mod)
    # drop any cached import of the adapter so it re-binds against the stubs
    monkeypatch.delitem(
        sys.modules,
        "sparx_agency.robots.ROBOTICAN.adapters.rooster_manual_control", raising=False)


class _StubClock:
    def now(self):
        return self

    def to_msg(self):
        return "stamp"


class _StubLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(("info", msg))

    def warn(self, msg):
        self.lines.append(("warn", msg))


class _StubNode:
    """Just enough rclpy Node surface for the adapter."""

    def __init__(self):
        self.published = []          # (topic, msg)
        self.timers = []             # (period_s, callback)
        self.subscriptions = []      # (msg_type, topic, callback)
        self.clients = []
        self._logger = _StubLogger()

    def create_publisher(self, msg_type, topic, depth, **kw):
        node = self

        class _Pub:
            def publish(self, msg):
                node.published.append((topic, msg))
        return _Pub()

    def create_timer(self, period, callback, **kw):
        self.timers.append((period, callback))

    def create_subscription(self, msg_type, topic, callback, depth, **kw):
        self.subscriptions.append((msg_type, topic, callback))

    def create_client(self, srv_type, name, **kw):
        node = self

        class _Client:
            ready = True
            calls = []

            def service_is_ready(self):
                return self.ready

            def call_async(self, request):
                node.clients.append((name, request.data))
        client = _Client()
        self._client = client
        return client

    def get_clock(self):
        return _StubClock()

    def get_logger(self):
        return self._logger


@pytest.fixture()
def rmc(monkeypatch):
    """An attached adapter plus its stub node."""
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    node = _StubNode()
    adapter = mod.RoosterManualControl(node, rooster_id="R1", publish_rate_hz=80.0,
                                       keep_alive_rate_hz=10.0)
    assert adapter.attach() is True
    return adapter, node, mod


# ── wiring ───────────────────────────────────────────────────────────────
def test_attach_wires_the_documented_topics_and_rates(rmc):
    adapter, node, _ = rmc
    assert [t for _, t, _ in [(m, t, c) for m, t, c in node.subscriptions]] == \
        ["/R1/fcu/state"]
    periods = sorted(round(p, 6) for p, _ in node.timers)
    assert periods == [pytest.approx(1 / 80.0), pytest.approx(1 / 10.0)]
    assert node.clients == []          # no arm requested yet
    assert adapter.available is True


def test_without_rooster_packages_it_stays_inert(monkeypatch):
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    monkeypatch.setattr(mod, "HAS_ROOSTER", False)
    node = _StubNode()
    adapter = mod.RoosterManualControl(node)
    # Handheld / dev-box mode: no publishers, no timers, send() is a no-op.
    assert adapter.attach() is False
    assert node.timers == [] and node.subscriptions == []
    assert adapter.send(mod.ManualAxes(x=500.0)) is False
    assert any(level == "warn" for level, _ in node._logger.lines)


# ── the arm gate ─────────────────────────────────────────────────────────
def test_send_while_disarmed_requests_arming_and_refuses(rmc):
    adapter, node, mod = rmc
    assert adapter.armed is False
    assert adapter.send(mod.ManualAxes(x=500.0)) is False
    assert node.clients == [("/R1/fcu/command/force_arm", True)]


def test_arm_requests_are_rate_limited(rmc, monkeypatch):
    adapter, node, mod = rmc
    now = [1000.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])
    # A bridge calling send() at inference rate must not flood the service.
    for _ in range(20):
        adapter.send(mod.ManualAxes(x=1.0))
    assert len(node.clients) == 1
    now[0] += adapter.arm_retry_s + 0.01
    adapter.send(mod.ManualAxes(x=1.0))
    assert len(node.clients) == 2


def _arm(adapter, node):
    _, _, on_state = node.subscriptions[0]
    on_state(types.SimpleNamespace(armed=True))
    assert adapter.armed is True


def test_send_after_arming_is_accepted(rmc):
    adapter, node, mod = rmc
    _arm(adapter, node)
    assert adapter.send(mod.ManualAxes(x=500.0)) is True


def test_commands_are_refused_during_the_post_arm_settling_window(monkeypatch):
    # Arming spins the rotors up; a command issued immediately fights the
    # spin-up. InternVLA-N1 carried this as a 1.0 s window in its own node.
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    now = [1000.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])
    node = _StubNode()
    adapter = mod.RoosterManualControl(node, stabilize_s=1.0)
    adapter.attach()
    _arm(adapter, node)

    assert adapter.stabilized is False
    assert adapter.ready is False
    assert adapter.send(mod.ManualAxes(x=500.0)) is False
    assert _mc_tick(node)[1].x == 0.0            # still idling

    now[0] += 1.5                                 # window elapsed
    assert adapter.stabilized is True and adapter.ready is True
    assert adapter.send(mod.ManualAxes(x=500.0, hold_s=10.0)) is True
    assert _mc_tick(node)[1].x == 500.0


def test_no_settling_window_by_default(rmc, mod=None):
    adapter, node, mod = rmc
    assert adapter.stabilize_s == 0.0
    _arm(adapter, node)
    assert adapter.stabilized is True and adapter.ready is True


def test_disarming_clears_stabilized(rmc):
    adapter, node, mod = rmc
    _arm(adapter, node)
    _, _, on_state = node.subscriptions[0]
    on_state(types.SimpleNamespace(armed=False))
    assert adapter.armed is False and adapter.stabilized is False


# ── the hold-then-stop latch ─────────────────────────────────────────────
def _mc_tick(node):
    period, cb = min(node.timers, key=lambda t: t[0])   # the 80 Hz one
    cb()
    return node.published[-1]


def test_a_command_is_republished_while_it_is_live(rmc):
    adapter, node, mod = rmc
    _arm(adapter, node)
    adapter.send(mod.ManualAxes(x=500.0, r=-250.0, hold_s=10.0))
    for _ in range(3):
        topic, msg = _mc_tick(node)
        assert topic == "/R1/manual_control"
        assert (msg.x, msg.r) == (500.0, -250.0)


def test_an_expired_command_becomes_a_zero_frame(monkeypatch):
    # Patch the clock BEFORE arming, so the arm timestamp shares the fake epoch.
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    now = [1000.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])
    node = _StubNode()
    adapter = mod.RoosterManualControl(node)
    adapter.attach()
    _arm(adapter, node)

    assert adapter.send(mod.ManualAxes(x=500.0, z=400.0, hold_s=0.05)) is True
    assert _mc_tick(node)[1].x == 500.0
    now[0] += 1.0                                   # past the hold window
    _, msg = _mc_tick(node)
    # This is the safety property: a bridge that stops publishing must not leave
    # the drone executing its last command forever.
    assert (msg.x, msg.y, msg.z, msg.r) == (0.0, 0.0, 0.0, 0.0)


def test_a_custom_idle_frame_is_published_instead_of_zeros(monkeypatch):
    # OmniVLA idles at z = stop_tilt = -1000, which BRAKES. Publishing an
    # all-zero frame there would coast instead of stopping.
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    node = _StubNode()
    adapter = mod.RoosterManualControl(node, idle_axes=mod.ManualAxes(z=-1000.0))
    adapter.attach()
    _, msg = _mc_tick(node)
    assert (msg.x, msg.r) == (0.0, 0.0)
    assert msg.z == -1000.0


def test_stop_drops_the_held_command_immediately(rmc):
    adapter, node, mod = rmc
    _arm(adapter, node)
    adapter.send(mod.ManualAxes(x=900.0, hold_s=10.0))
    adapter.stop()
    assert _mc_tick(node)[1].x == 0.0


# ── axis clamping ────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    (0.0, 0.0), (999.0, 999.0), (1000.0, 1000.0),
    (5000.0, 1000.0), (-5000.0, -1000.0),
])
def test_axes_are_clamped_to_the_fcu_range(rmc, raw, expected):
    adapter, node, mod = rmc
    _arm(adapter, node)
    adapter.send(mod.ManualAxes(x=raw, z=raw, r=raw, hold_s=10.0))
    _, msg = _mc_tick(node)
    assert (msg.x, msg.z, msg.r) == (expected, expected, expected)


def test_clamp_axis_returns_float_for_int_input(rmc):
    _, _, mod = rmc
    assert isinstance(mod.clamp_axis(5), float)


# ── heartbeat ────────────────────────────────────────────────────────────
def test_keep_alive_publishes_the_requested_flight_mode(rmc):
    adapter, node, mod = rmc
    period, cb = max(node.timers, key=lambda t: t[0])   # the 10 Hz one
    cb()
    topic, msg = node.published[-1]
    assert topic == "/R1/keep_alive"
    assert msg.is_active is True
    assert msg.requested_flight_mode == mod.FLIGHT_MODE_GROUND_ROLL
    assert msg.command_reboot is False


def test_rooster_id_namespaces_every_topic(monkeypatch):
    _install_ros_stubs(monkeypatch)
    from sparx_agency.robots.ROBOTICAN.adapters import rooster_manual_control as mod
    node = _StubNode()
    mod.RoosterManualControl(node, rooster_id="R2").attach()
    assert node.subscriptions[0][1] == "/R2/fcu/state"
    max(node.timers, key=lambda t: t[0])[1]()
    assert node.published[-1][0] == "/R2/keep_alive"
