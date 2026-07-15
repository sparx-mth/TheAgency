"""Behavioural tests for the hybrid_planner arbiter (A* easy legs / NavDP hard ones).

The node is a ROS1 adapter, so ``rospy`` + the message packages (and ``cv2``) are
stubbed in ``sys.modules`` before it is imported -- enough to drive its callbacks
and control tick directly and assert on the mode machine, WITHOUT a NavDP server.
The contract locked in here is the one the user asked for:

  * a straight, open route stays on A* (PRIMARY, echoing A*);
  * a hard turn ahead engages NavDP after ``difficulty_confirm`` ticks (STOP first);
  * a doorway (narrow on both sides, from the BEV) engages the same way;
  * once the route ahead is easy again it returns to A* after ``recover_confirm``
    ticks (sticky), never mid-maneuver;
  * a boxed-in A* (no route) engages too, when ~engage_on_astar_fail;
  * a raising assessment never kills the tick loop (drops back to A*).
"""
import pathlib
import sys
import types

import numpy as np

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


class _Clock:
    t = 0.0


_CLOCK = _Clock()


class _Duration:
    def __init__(self, secs=0.0):
        self.secs = float(secs)

    def to_sec(self):
        return self.secs

    def __radd__(self, other):
        return other

    def __add__(self, other):
        return _Duration(self.secs + float(getattr(other, "secs", other)))


class _Time:
    def __init__(self, secs=0.0):
        self.secs = float(secs)

    @staticmethod
    def now():
        return _Time(_CLOCK.t)

    def __sub__(self, other):
        return _Duration(self.secs - other.secs)

    def __add__(self, dur):
        return _Time(self.secs + dur.secs)

    def __lt__(self, other):
        return self.secs < other.secs


class _Pub:
    def __init__(self, topic):
        self.topic = topic
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


_PARAMS = {}


def _install_stubs():
    rospy = types.ModuleType("rospy")
    rospy.init_node = lambda *a, **k: None
    rospy.get_param = lambda name, default=None: _PARAMS.get(name, default)
    rospy.Time = _Time
    rospy.Duration = _Duration
    rospy.Publisher = lambda topic, *a, **k: _Pub(topic)
    rospy.Subscriber = lambda *a, **k: None
    rospy.Timer = lambda *a, **k: None
    rospy.spin = lambda: None
    rospy.signal_shutdown = lambda *a, **k: None
    rospy.is_shutdown = lambda: False
    for fn in ("loginfo", "logwarn", "logerr", "logfatal", "loginfo_throttle",
               "logwarn_throttle", "logerr_throttle"):
        setattr(rospy, fn, lambda *a, **k: None)
    rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
    sys.modules["rospy"] = rospy

    class _Header:
        def __init__(self):
            self.stamp = None
            self.frame_id = ""

    class _Vec:
        def __init__(self):
            self.x = self.y = self.z = self.w = 0.0

    class _Pose:
        def __init__(self):
            self.position = _Vec()
            self.orientation = _Vec()

    class _PoseStamped:
        def __init__(self):
            self.header = _Header()
            self.pose = _Pose()

    class _Path:
        def __init__(self):
            self.header = _Header()
            self.poses = []

    class _Bool:
        def __init__(self, data=False):
            self.data = data

    class _Info:
        def __init__(self):
            self.width = 0
            self.height = 0
            self.resolution = 0.0
            self.origin = _Pose()

    class _OccupancyGrid:
        def __init__(self):
            self.info = _Info()
            self.data = []

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    geo = _mod("geometry_msgs")
    geo.msg = _mod("geometry_msgs.msg", Pose=_Pose, PoseStamped=_PoseStamped,
                   Point=_Vec)
    nav = _mod("nav_msgs")
    nav.msg = _mod("nav_msgs.msg", OccupancyGrid=_OccupancyGrid, Path=_Path)
    sen = _mod("sensor_msgs")
    sen.msg = _mod("sensor_msgs.msg", CameraInfo=object, Image=object)
    class _String:
        def __init__(self, data=""):
            self.data = data

    std = _mod("std_msgs")
    std.msg = _mod("std_msgs.msg", Bool=_Bool, String=_String)
    _mod("cv2", cvtColor=lambda *a, **k: None, imread=lambda *a, **k: None,
         COLOR_BGR2RGB=0, IMREAD_COLOR=1)
    return _Path, _PoseStamped, _OccupancyGrid


_Path_t, _PoseStamped_t, _Occ_t = _install_stubs()
import hybrid_planner_node as hpn  # noqa: E402

RES = 0.1
ORIGIN = -3.0
N = 60  # 6 m x 6 m grid centred on the origin


def _path(pts):
    m = _Path_t()
    for x, y in pts:
        ps = _PoseStamped_t()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        m.poses.append(ps)
    return m


def _bev(occ_fn):
    """OccupancyGrid stub whose cell (gx,gy) is occupied iff occ_fn(worldx, worldy)."""
    grid = np.zeros((N, N), dtype=np.int8)
    for gy in range(N):
        for gx in range(N):
            wx = (gx + 0.5) * RES + ORIGIN
            wy = (gy + 0.5) * RES + ORIGIN
            if occ_fn(wx, wy):
                grid[gy, gx] = 100
    m = _Occ_t()
    m.info.height, m.info.width = N, N
    m.info.resolution = RES
    m.info.origin.position.x = ORIGIN
    m.info.origin.position.y = ORIGIN
    m.data = np.ascontiguousarray(grid).ravel()
    return m


def _make_node(**params):
    _PARAMS.clear()
    _PARAMS.update({
        "~difficulty_lookahead_m": 4.0,
        "~difficulty_skip_m": 0.0,
        "~turn_thresh_deg": 45.0,
        "~passage_width_m": 1.0,
        "~difficulty_confirm": 2,
        "~recover_confirm": 3,
        "~astar_fail_confirm": 3,
        "~tick_hz": 5.0,
    })
    _PARAMS.update(params)
    node = hpn.HybridPlannerNode()
    node._reset_done = True          # short-circuit _ensure_reset (no NavDP HTTP)
    node._run_inference = lambda: hpn._PENDING   # never hit the server in a tick
    node.pose_xyyaw = (0.0, 0.0, 0.0)
    node.altitude = 1.0
    return node


def _set_route(node, pts):
    node._astar_path_cb(_path(pts))


# ─── PRIMARY: easy route stays on A* ─────────────────────────────────────────
def test_straight_open_route_stays_on_astar():
    node = _make_node()
    _set_route(node, [(0, 0), (5, 0)])
    for _ in range(6):
        node._tick(None)
    assert node.mode == hpn._PRIMARY
    assert node.difficulty_streak == 0


# ─── Hard turn engages NavDP after the confirm streak ────────────────────────
def test_hard_turn_engages_after_confirm():
    node = _make_node(**{"~difficulty_confirm": 2})
    _set_route(node, [(0, 0), (2, 0), (2, 3)])   # 90 deg corner ahead
    node._tick(None)
    assert node.mode == hpn._PRIMARY and node.difficulty_streak == 1  # not yet
    node._tick(None)
    assert node.mode == hpn._ENGAGED             # confirmed -> engaged
    assert node.leg_state == hpn._HOLD
    # Engaging STOPS the drone: a 2-point coincident hold at the current pose.
    last = node.pub_path.msgs[-1]
    assert len(last.poses) == 2
    assert last.poses[0].pose.position.x == last.poses[1].pose.position.x == 0.0


# ─── Doorway (narrow both sides, from the BEV) engages ───────────────────────
def _doorway(x, y):
    return 1.3 <= x <= 1.8 and abs(y) >= 0.4     # 0.8 m gap across the x axis


def test_doorway_engages():
    node = _make_node(**{"~difficulty_confirm": 2, "~turn_thresh_deg": 200.0})
    node._bev_cb(_bev(_doorway))                 # a real narrow gap on the route
    _set_route(node, [(0, 0), (2.5, 0)])         # straight THROUGH the doorway
    node._tick(None)
    node._tick(None)
    assert node.mode == hpn._ENGAGED


def test_open_room_with_bev_stays_on_astar():
    node = _make_node(**{"~turn_thresh_deg": 200.0})   # turn disabled: narrow only
    node._bev_cb(_bev(lambda x, y: False))       # no walls anywhere
    _set_route(node, [(0, 0), (2.5, 0)])
    for _ in range(5):
        node._tick(None)
    assert node.mode == hpn._PRIMARY


# ─── Return to A* once the hard part is behind (sticky) ──────────────────────
def test_returns_to_astar_when_cleared():
    node = _make_node(**{"~difficulty_confirm": 2, "~recover_confirm": 3})
    _set_route(node, [(0, 0), (2, 0), (2, 3)])
    node._tick(None); node._tick(None)
    assert node.mode == hpn._ENGAGED
    node.n_legs = 1                               # a NavDP leg has flown through it
    # Drone has flown through the turn: now a straight, easy route ahead.
    node.pose_xyyaw = (2.0, 3.0, 0.0)
    _set_route(node, [(2, 3), (2, 7)])
    node.astar_ok = True
    for i in range(2):                            # not yet: sticky (needs 3)
        node._tick(None)
        assert node.mode == hpn._ENGAGED, "returned too early on tick %d" % i
    node._tick(None)
    assert node.mode == hpn._PRIMARY


def test_no_disengage_before_a_navdp_leg_flies():
    # The bug: during the brake-settle (before the first inference) the difficulty
    # flickered easy and the drone bounced back to A* without ever calling NavDP.
    # With no leg flown (n_legs == 0), an easy route must NOT disengage.
    node = _make_node(**{"~difficulty_confirm": 2, "~recover_confirm": 3})
    _set_route(node, [(0, 0), (2, 0), (2, 3)])
    node._tick(None); node._tick(None)
    assert node.mode == hpn._ENGAGED and node.n_legs == 0
    node._select = lambda snap: None             # nothing visible -> no leg ever flies
    node.pose_xyyaw = (2.0, 3.0, 0.0)             # and the route ahead reads easy
    _set_route(node, [(2, 3), (2, 7)])
    for _ in range(10):
        node._tick(None)
    assert node.mode == hpn._ENGAGED             # stays committed to NavDP (n_legs still 0)
    assert node.recover_streak == 0


def test_no_disengage_until_flown_through_the_maneuver():
    # Even with a leg flown and the route ahead easy, it must NOT return to A* until
    # the drone has flown min_pass_distance_m from where it engaged (through the
    # maneuver) -- not the instant the route momentarily reads easy.
    node = _make_node(**{"~difficulty_confirm": 2, "~recover_confirm": 2,
                         "~min_pass_distance_m": 2.0})
    _set_route(node, [(0, 0), (2, 0), (2, 3)])
    node._tick(None); node._tick(None)
    assert node.mode == hpn._ENGAGED
    node.n_legs = 1
    # A leg flew and the route ahead is easy, but the drone barely moved (0.5 m).
    node.pose_xyyaw = (0.5, 0.0, 0.0)
    _set_route(node, [(0.5, 0), (0.5, 4)])
    for _ in range(6):
        node._tick(None)
    assert node.mode == hpn._ENGAGED             # travel gate holds it on NavDP
    # Now it has flown through the maneuver (>= 2 m from the engage point).
    node.pose_xyyaw = (2.5, 0.0, 0.0)
    _set_route(node, [(2.5, 0), (2.5, 4)])
    node._tick(None); node._tick(None)           # recover_confirm = 2
    assert node.mode == hpn._PRIMARY


# ─── A* boxed-in (no route) engages when enabled ─────────────────────────────
def test_astar_no_route_engages():
    node = _make_node(**{"~engage_on_astar_fail": True, "~astar_fail_confirm": 3})
    _set_route(node, [(0, 0), (5, 0)])           # geometrically easy...
    node.astar_ok = False
    node.fail_streak = 3                          # ...but A* reports no route
    node._tick(None)
    assert node.mode == hpn._ENGAGED


def test_astar_no_route_ignored_when_disabled():
    node = _make_node(**{"~engage_on_astar_fail": False})
    _set_route(node, [(0, 0), (5, 0)])
    node.fail_streak = 9
    for _ in range(4):
        node._tick(None)
    assert node.mode == hpn._PRIMARY


def test_astar_no_route_needs_a_pose():
    # No pose yet -> the rescue must NOT commit an engage (it could not STOP-hold).
    node = _make_node(**{"~engage_on_astar_fail": True, "~astar_fail_confirm": 3})
    node.pose_xyyaw = None
    _set_route(node, [(0, 0), (5, 0)])
    node.fail_streak = 5
    node._tick(None)
    assert node.mode == hpn._PRIMARY


def _snap(node, waypoints):
    return {"pose": node.pose_xyyaw, "alt": 1.0, "rgb": None, "depth": None,
            "waypoints": waypoints}


def test_rescue_aims_at_the_goal_when_no_visible_waypoint():
    # Boxed A*: waypoints are a coincident STOP at the drone -> no visible A* wp.
    # The rescue must aim NavDP at the mission goal (a real forward target), not stall.
    node = _make_node()
    node.goal_world = (5.0, 0.0)
    node._rescue = True
    node._select = lambda snap: None             # nothing visible (boxed)
    target = node._choose_target(_snap(node, [(0, 0), (0, 0)]))
    assert isinstance(target, tuple), "rescue produced no target -> would stall"
    gx, gy, is_final, tag = target
    assert gx > 0 and abs(gy) < 1e-6 and not is_final   # forward toward (5,0)


def test_no_target_without_rescue_or_goal():
    node = _make_node()
    node._select = lambda snap: None
    node._rescue = False
    assert node._choose_target(_snap(node, [(0, 0), (0, 0)])) == hpn._NO_POINT


def test_rescue_arrival_holds():
    node = _make_node(**{"~arrival_radius_m": 0.5})
    node.goal_world = (0.3, 0.0)                  # within arrival radius of pose (0,0)
    node._rescue = True
    node._select = lambda snap: None
    assert node._choose_target(_snap(node, [(0, 0), (0, 0)])) == hpn._ARRIVED


def test_resume_clears_fail_streak_no_instant_reengage():
    # After a bounded give-up (still-boxed A*), resume must reset fail_streak so the
    # rescue does not re-fire on the very next tick from the stale count.
    node = _make_node(**{"~engage_on_astar_fail": True, "~astar_fail_confirm": 3})
    _set_route(node, [(0, 0), (5, 0)])
    node.fail_streak = 5
    node._tick(None)
    assert node.mode == hpn._ENGAGED
    assert node.fail_streak == 0                  # consumed on engage
    node._resume_primary("wait_timeout")
    assert node.fail_streak == 0
    node._tick(None)                              # stale count must not re-engage
    assert node.mode == hpn._PRIMARY


# ─── Resilience: a bad assessment never kills the loop ────────────────────────
def test_tick_survives_assessment_error():
    node = _make_node()
    _set_route(node, [(0, 0), (5, 0)])

    def _boom():
        raise RuntimeError("bad frame")

    node._assess_difficulty = _boom
    node._tick(None)                              # must not raise
    assert node.mode == hpn._PRIMARY
