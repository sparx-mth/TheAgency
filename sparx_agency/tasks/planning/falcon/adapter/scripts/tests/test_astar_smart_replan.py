"""Behavioural tests for the smart (event-driven) A* replanning in astar_planner_node.

The node is a ROS1 adapter, so ``rospy`` and the message packages are stubbed in
``sys.modules`` before it is imported. The stubs give a controllable clock, a
recording ``Publisher`` and plain-attribute message objects -- enough to drive the
node's callbacks directly and assert on what it publishes. The point is to lock in
the anti-oscillation contract the user asked for:

  * a static map never re-publishes (the drone is not whipsawed),
  * discoveries off the route, or that do not shorten it, do not re-publish,
  * a discovery that opens a meaningfully shorter route IS adopted,
  * the commitment window holds the route for the slow platform,
  * a confirmed obstacle reroutes, and a boxed-in obstacle STOPS the drone once.
"""
import pathlib
import sys
import types

import numpy as np
import pytest

# ─── stub rospy + message packages BEFORE importing the node ────────────────
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


class _Time:
    def __init__(self, secs=0.0):
        self.secs = float(secs)

    @staticmethod
    def now():
        return _Time(_CLOCK.t)

    def __sub__(self, other):
        return _Duration(self.secs - other.secs)

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
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 0.0

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

    class _Point:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class _Bool:
        def __init__(self, data=False):
            self.data = data

    class _String:
        def __init__(self, data=""):
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

    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.Pose, geo_msg.PoseStamped, geo_msg.Point = _Pose, _PoseStamped, _Point
    geo.msg = geo_msg
    sys.modules["geometry_msgs"] = geo
    sys.modules["geometry_msgs.msg"] = geo_msg

    nav = types.ModuleType("nav_msgs")
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.OccupancyGrid, nav_msg.Path = _OccupancyGrid, _Path
    nav.msg = nav_msg
    sys.modules["nav_msgs"] = nav
    sys.modules["nav_msgs.msg"] = nav_msg

    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.Bool = _Bool
    std_msg.String = _String
    std.msg = std_msg
    sys.modules["std_msgs"] = std
    sys.modules["std_msgs.msg"] = std_msg
    return _Pose, _Point, _OccupancyGrid


_Pose_t, _Point_t, _Occ_t = _install_stubs()
import astar_planner_node as apn  # noqa: E402
from sparx_agency.core.common.types import Path2D, Pose2D  # noqa: E402

FREE, OCC, UNK = 0, 100, -1
RES = 0.1
H = W = 40


def _occ_msg(grid2d):
    m = _Occ_t()
    m.info.height, m.info.width = grid2d.shape
    m.info.resolution = RES
    m.info.origin.position.x = 0.0
    m.info.origin.position.y = 0.0
    m.data = np.ascontiguousarray(grid2d.astype(np.int8)).ravel()
    return m


def _pose_msg(x, y):
    p = _Pose_t()
    p.position.x, p.position.y = x, y
    p.orientation.w = 1.0
    return p


def _point(x, y):
    return _Point_t(x, y)


def _node(**params):
    _PARAMS.clear()
    _PARAMS.update({"~min_free_cells_to_plan": 50})
    _PARAMS.update({("~" + k): v for k, v in params.items()})
    _CLOCK.t = 100.0
    return apn.AStarPlannerNode()


def _all_free():
    return np.full((H, W), FREE, np.int16)


def _advance(dt):
    _CLOCK.t += dt


def _n_paths(node):
    return len(node.pub_path.msgs)


def _last_status(node):
    return node.pub_status.msgs[-1].data if node.pub_status.msgs else None


def _bootstrap(node, grid, gx=3.8, gy=2.0, px=0.2, py=2.0):
    """Set goal + pose, feed the first BEV, return once a route is committed."""
    node._pose_cb(_pose_msg(px, py))
    node._goal_cb(_point(gx, gy))         # grid still None -> just stores goal
    node._bev_cb(_occ_msg(grid))          # first BEV -> first plan + commit
    return node


# ─── anti-oscillation ───────────────────────────────────────────────────────
def test_first_bev_commits_once():
    node = _node()
    _bootstrap(node, _all_free())
    assert node.has_plan
    assert _n_paths(node) == 1
    assert _last_status(node) is True


def test_goal_click_does_not_bypass_warmup_gate():
    """A goal click on a cold (mostly-UNKNOWN) map must NOT plan blind: the warmup
    gate applies to every planning entry point, not just the BEV-driven one."""
    node = _node(min_free_cells_to_plan=1000)   # high bar: never warms up here
    cold = np.full((H, W), UNK, np.int16)
    cold[0:2, 0:2] = FREE                        # only 4 FREE cells (< 1000)
    node._pose_cb(_pose_msg(0.2, 2.0))
    node._bev_cb(_occ_msg(cold))                 # BEV path holds (warmup)
    assert not node.has_plan
    node._goal_cb(_point(3.8, 2.0))              # click must also hold
    assert not node.has_plan, "goal click must not plan on an un-warmed map"
    assert _n_paths(node) == 0
    assert _last_status(node) is False           # reported not-ready (retry)


def test_static_map_never_republishes():
    """The core contract: an unchanging map must not re-publish (no follower reset)."""
    node = _node()
    _bootstrap(node, _all_free())
    grid = _all_free()
    for _ in range(30):
        _advance(1.0)                     # well past any commit window
        node._bev_cb(_occ_msg(grid))
    assert _n_paths(node) == 1, "static map must not trigger any replan/re-publish"


def test_quiet_committed_frame_still_publishes_status_true():
    """A committed route that is simply valid must emit status=True every frame, so
    a fallback arbiter counting consecutive successes (to resume A* from NavDP) is
    not starved during a long, event-free flight."""
    node = _node()
    _bootstrap(node, _all_free())
    before = len(node.pub_status.msgs)
    grid = _all_free()
    for _ in range(4):
        _advance(0.3)
        node._bev_cb(_occ_msg(grid))
    new_status = node.pub_status.msgs[before:]
    assert len(new_status) == 4, "one status sample per committed frame"
    assert all(s.data is True for s in new_status)
    assert _n_paths(node) == 1, "still no re-publish (status is not a path)"


def test_commitment_window_blocks_early_discovery():
    """Even with a big discovery, no opportunistic replan inside the commit window."""
    node = _node(replan_commit_min_s=5.0)
    _bootstrap(node, _all_free())
    node._new_known_in_corridor = lambda: 10_000     # force the discovery gate open
    calls = {"n": 0}
    orig = node._opportunistic_replan
    node._opportunistic_replan = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                                  orig(*a, **k))[1]
    _advance(2.0)                                     # < 5.0 commit window
    node._bev_cb(_occ_msg(_all_free()))
    assert calls["n"] == 0, "no opportunistic replan inside the commit window"
    _advance(4.0)                                     # now past 5.0 total
    node._bev_cb(_occ_msg(_all_free()))
    assert calls["n"] == 1, "opportunistic replan must run once the window elapses"


def test_discovery_suppressed_when_not_shorter():
    """Discovery fires, but a candidate that is not >=improve_frac shorter is kept
    (no re-publish) -- this is what kills the L/R flip-flop oscillation."""
    node = _node(replan_commit_min_s=1.0, replan_improve_frac=0.15)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    node._new_known_in_corridor = lambda: 10_000
    # Candidate the SAME length as the remaining committed route -> suppress.
    node._plan_candidate = lambda: Path2D(
        points=(Pose2D(0.2, 2.0), Pose2D(3.8, 2.0)), frame_id="world")
    _advance(2.0)
    node._bev_cb(_occ_msg(_all_free()))
    assert _n_paths(node) == n0, "a non-improving candidate must not be adopted"
    assert _last_status(node) is True


def test_suppressed_discovery_does_not_churn_astar():
    """After a discovery is evaluated and the route is kept, the SAME reveal must
    not re-run A* every commit window on an otherwise static map (the high-water
    mark latches it off); a genuine further reveal still fires."""
    node = _node(replan_commit_min_s=0.5, replan_corridor_radius_m=1.0)
    g0 = np.full((H, W), UNK, np.int16)
    g0[18:23, :] = FREE                    # known route band (warmup + straight route)
    _bootstrap(node, g0)
    assert node.has_plan
    calls = {"n": 0}
    orig = node._plan_candidate

    def _counting():
        calls["n"] += 1
        return orig()
    node._plan_candidate = _counting
    g1 = g0.copy()
    g1[15:18, :] = FREE                    # reveal a big in-corridor region
    _advance(1.0)
    node._bev_cb(_occ_msg(g1))
    assert node._new_known_in_corridor() >= node.replan_min_new_cells
    first = calls["n"]
    assert first == 1, "discovery evaluates A* exactly once"
    for _ in range(10):                    # same map, repeatedly, past the window
        _advance(1.0)
        node._bev_cb(_occ_msg(g1))
    assert calls["n"] == first, "a kept reveal must not re-run A* on a static map"
    assert _n_paths(node) == 1
    # A FURTHER reveal must still fire A* again.
    g2 = g1.copy()
    g2[12:15, :] = FREE
    _advance(1.0)
    node._bev_cb(_occ_msg(g2))
    assert calls["n"] == first + 1, "a genuine further reveal must re-evaluate"


def test_discovery_adopts_a_shorter_route():
    """A discovery that opens a meaningfully shorter route IS adopted (re-published)."""
    node = _node(replan_commit_min_s=1.0, replan_improve_frac=0.15)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    node._new_known_in_corridor = lambda: 10_000
    # Candidate ~half the length of the remaining committed route -> adopt.
    node._plan_candidate = lambda: Path2D(
        points=(Pose2D(0.2, 2.0), Pose2D(2.0, 2.0)), frame_id="world")
    _advance(2.0)
    node._bev_cb(_occ_msg(_all_free()))
    assert _n_paths(node) == n0 + 1, "a >=15% shorter candidate must be adopted"


def test_off_corridor_discovery_does_not_fire():
    """A big reveal far from the route corridor yields zero relevant new cells."""
    node = _node(replan_commit_min_s=1.0, replan_corridor_radius_m=0.5)
    # Commit with the route corridor (around row 20) FREE, everything else UNKNOWN.
    g0 = np.full((H, W), UNK, np.int16)
    g0[15:26, :] = FREE                   # a known band containing the y=2.0 route
    _bootstrap(node, g0)
    assert node.has_plan
    n0 = _n_paths(node)
    # Reveal a large region far from the corridor (rows 0..5) as free.
    g1 = g0.copy()
    g1[0:6, :] = FREE
    _advance(2.0)
    # The relevance count must be ~0 (revealed cells are outside the corridor).
    node.grid = node._decode(_occ_msg(g1))
    assert node._new_known_in_corridor() == 0
    node._bev_cb(_occ_msg(g1))
    assert _n_paths(node) == n0, "off-corridor discovery must not replan"


# ─── obstacle handling ───────────────────────────────────────────────────────
def _wall(rows):
    g = _all_free()
    g[rows, 18:22] = OCC                  # a vertical wall band across the route
    return g


def test_collision_reroutes_around_partial_wall():
    node = _node(replan_collision_confirm=2)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))            # blocks row 20 but leaves rows 30..39 open
    _advance(0.5)
    node._bev_cb(_occ_msg(wall))          # collision frame 1 (unconfirmed)
    assert _n_paths(node) == n0
    _advance(0.5)
    node._bev_cb(_occ_msg(wall))          # collision frame 2 (confirmed) -> reroute
    assert _n_paths(node) == n0 + 1, "confirmed collision must reroute"
    assert _last_status(node) is True, "a reroute was found -> status True"


def test_boxed_in_stops_once_and_signals_fallback():
    node = _node(replan_collision_confirm=1, stop_on_blocked_plan_fail=True)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    full = _wall(slice(0, H))             # full wall -> A* has no route
    _advance(0.5)
    node._bev_cb(_occ_msg(full))          # confirmed collision, no route -> STOP + False
    assert _n_paths(node) == n0 + 1, "a STOP hold must be published"
    hold = node.pub_path.msgs[-1]
    assert len(hold.poses) == 2
    p0, p1 = hold.poses[0].pose.position, hold.poses[1].pose.position
    assert (p0.x, p0.y) == (p1.x, p1.y), "STOP hold is a coincident 2-point path"
    assert _last_status(node) is False, "no route -> status False drives NavDP fallback"
    assert node._blocked_hold
    # A second boxed-in frame must NOT re-publish the hold (dedup), but keeps failing.
    _advance(0.5)
    node._bev_cb(_occ_msg(full))
    assert _n_paths(node) == n0 + 1, "STOP hold is published once, not every frame"
    assert _last_status(node) is False


def test_legacy_mode_still_plans_via_try_plan():
    """smart_replan:=false must fall back to the legacy periodic path unchanged."""
    node = _node(smart_replan=False)
    assert node.smart_replan is False
    node._pose_cb(_pose_msg(0.2, 2.0))
    node._goal_cb(_point(3.8, 2.0))       # legacy goal click -> _try_plan
    node._bev_cb(_occ_msg(_all_free()))   # legacy first-plan on BEV
    assert node.has_plan
    assert _n_paths(node) >= 1
    assert _last_status(node) is True


def test_boxed_in_recovers_when_route_reopens():
    node = _node(replan_collision_confirm=1, stop_on_blocked_plan_fail=True)
    _bootstrap(node, _all_free())
    n_after_commit = _n_paths(node)
    _advance(0.5)
    node._bev_cb(_occ_msg(_wall(slice(0, H))))     # box in -> STOP hold
    assert node._blocked_hold
    n_blocked = _n_paths(node)
    _advance(0.5)
    node._bev_cb(_occ_msg(_all_free()))            # wall clears -> recover + commit
    assert not node._blocked_hold
    assert _n_paths(node) == n_blocked + 1, "recovery re-publishes a real route"
    assert _last_status(node) is True


# ─── confidence-aware obstacle confirmation ──────────────────────────────────
def _conf_msg(conf_float):
    """OccupancyGrid carrying a [0,1] confidence as int8 0..100, co-registered
    with _occ_msg (same origin/resolution/shape)."""
    m = _Occ_t()
    m.info.height, m.info.width = conf_float.shape
    m.info.resolution = RES
    m.info.origin.position.x = 0.0
    m.info.origin.position.y = 0.0
    conf8 = np.clip(np.rint(conf_float * 100.0), 0, 100).astype(np.int8)
    m.data = np.ascontiguousarray(conf8).ravel()
    return m


def _conf_for(grid2d, value):
    """A confidence grid = ``value`` on the OCC cells of ``grid2d``, else 0."""
    return np.where(grid2d == OCC, float(value), 0.0).astype(np.float32)


def _feed(node, grid, conf):
    """Feed one co-registered (confidence, BEV) frame, in that order."""
    _advance(0.5)
    node._conf_cb(_conf_msg(conf))
    node._bev_cb(_occ_msg(grid))


def test_low_confidence_obstacle_is_not_rerouted():
    """The user's case: a noisy on-route cell that the map is NOT confident about
    must NOT trigger a reroute -- the committed route is kept so the drone keeps
    observing (and can clean) the cell instead of veering off it."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75,
                 replan_collision_confirm_max=0)   # no ceiling -> pure conf gate
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    conf = _conf_for(wall, 0.5)                    # below the 0.75 threshold
    for _ in range(20):                            # many frames, stably occupied
        _feed(node, wall, conf)
    assert _n_paths(node) == n0, "a low-confidence obstacle must not reroute"
    assert node._collision_streak >= 2, "collision is seen, just not acted on"


def test_confident_obstacle_reroutes():
    """A high-confidence on-route obstacle reroutes on the confirm frame as before."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    conf = _conf_for(wall, 0.9)                    # above the 0.75 threshold
    _feed(node, wall, conf)                        # streak 1 (< confirm)
    assert _n_paths(node) == n0
    _feed(node, wall, conf)                        # streak 2, confident -> reroute
    assert _n_paths(node) == n0 + 1, "a confident obstacle must reroute"
    assert _last_status(node) is True


def test_confidence_ceiling_forces_reroute():
    """A persistently-flagged but never-confident obstacle still reroutes once the
    frame ceiling is hit -- the safety backstop bounds flying toward it."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75,
                 replan_collision_confirm_max=5)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    conf = _conf_for(wall, 0.4)                    # stays below threshold forever
    for _ in range(4):                             # streaks 1..4: kept
        _feed(node, wall, conf)
        assert _n_paths(node) == n0
    _feed(node, wall, conf)                        # streak 5 == ceiling -> reroute
    assert _n_paths(node) == n0 + 1, "frame ceiling must force a reroute"


def test_confidence_gate_disabled_reroutes_on_frame_count():
    """~replan_confirm_conf=0 disables the confidence gate: reroute on the frame
    count regardless of how low the obstacle confidence is."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.0)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    conf = _conf_for(wall, 0.01)                   # near-zero confidence, ignored
    _feed(node, wall, conf)                        # streak 1
    _feed(node, wall, conf)                        # streak 2 -> reroute (gate off)
    assert _n_paths(node) == n0 + 1


def test_rising_confidence_reroutes_when_obstacle_firms_up():
    """The core promise: an on-route cell that starts LOW-confidence is kept (drone
    keeps looking), then reroutes as soon as the map firms it up -- and with the
    ceiling OFF, so it is the CONFIDENCE that triggers, not the frame count."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75,
                 replan_collision_confirm_max=0)   # no ceiling: pure conf trigger
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    for _ in range(4):                             # marginal evidence: kept
        _feed(node, wall, _conf_for(wall, 0.5))
    assert _n_paths(node) == n0, "low-confidence obstacle is kept, not rerouted"
    _feed(node, wall, _conf_for(wall, 0.9))        # firms up -> reroute
    assert _n_paths(node) == n0 + 1, "a firmed-up obstacle must reroute"


def test_low_confidence_obstacle_that_clears_never_reroutes():
    """The noise case end to end: a low-confidence on-route cell is kept, then the
    map cleans it (seen free) -- the streak resets and the drone never rerouted."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75,
                 replan_collision_confirm_max=0)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    for _ in range(4):
        _feed(node, wall, _conf_for(wall, 0.5))
    assert _n_paths(node) == n0
    _feed(node, _all_free(), _conf_for(_all_free(), 0.0))   # noise cleared
    assert node._collision_streak == 0, "a cleared collision resets the streak"
    assert _n_paths(node) == n0, "a speckle that clears must never reroute"


def test_mismatched_confidence_grid_falls_back_to_frame_count():
    """A confidence grid that is not co-registered with the BEV (different lattice)
    is ignored, so the node falls back to the pure frame-count gate and reroutes."""
    node = _node(replan_collision_confirm=2, replan_confirm_conf=0.75)
    _bootstrap(node, _all_free())
    n0 = _n_paths(node)
    wall = _wall(slice(0, 30))
    low = _conf_for(wall, 0.1)
    bad = _conf_msg(low)
    bad.info.origin.position.x = 99.0              # wrong origin -> key mismatch
    _advance(0.5); node._conf_cb(bad); node._bev_cb(_occ_msg(wall))   # streak 1
    _advance(0.5); node._conf_cb(bad); node._bev_cb(_occ_msg(wall))   # streak 2
    assert _n_paths(node) == n0 + 1, "stale/mismatched conf -> frame-count reroute"
