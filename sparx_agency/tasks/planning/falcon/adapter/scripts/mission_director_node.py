#!/usr/bin/env python3
"""mission_director_node.py -- pick the object to fly to, then arm the mission.

The FALCON object-approach stack can fly to a coordinate while hunting a named object,
and land on reaching it. This node is the front end that DECIDES the target and, only
then, arms the rest of the stack. Until an object is selected it publishes NOTHING, so
nothing plans and nothing flies (the gate); on selection it publishes three things that,
together, start the mission:

  1. the object's LABEL on ``~target_topic`` (/object_approach/goal, std_msgs/String) --
     re-prompts the (already-running) YOLO-World detector to hunt that label AND re-keys
     the in-container object_approach confirmation gate. One publish fans out to both.
  2. the object's WORLD (x, y) on ``~goal_topic`` (/waypoint_nav/goal, geometry_msgs/Point)
     -- the coordinate goal every planner (A* / hybrid / combination / fallback / NavDP)
     subscribes to and (re)plans a route toward. This is the point goal used "by A* alone".
  3. ``True`` on ``~enable_topic`` (/object_approach/enable, std_msgs/Bool) -- arms the
     object_approach mission (tracker + visual servo + arrival-land), which starts disabled.

All three are LATCHED so a late-joining planner / detector / bridge picks up the current
mission without a re-publish; latching also maps the label topic to TRANSIENT_LOCAL across
the ROS1<->ROS2 bridge, which the detector REQUIRES (a volatile publish is dropped
silently). See ``retarget_object.sh`` for the same durability note.

Two selection modes (``~selection_mode``):

  * ``random`` -- pick one object uniformly at random at startup (seedable via ``~seed``),
    arm the mission, and hold. One-shot: the target does not change afterwards.
  * ``gui`` (default) -- open a matplotlib window listing every object (label + world x,y).
    Click a row (or press its 1-9 number, or ``r`` for a random pick) to select it and arm
    the mission; click a different row at any time to RETARGET live -- the planner replans
    to the new goal and the detector re-prompts. matplotlib is the only GUI toolkit proven
    in the FALCON container (the BEV click-goal window uses it).

The catalog is loaded ROS-free by ``core.planning.mission.ObjectCatalog`` from a JSON list
(default: the ``objects.json`` shipped next to this task; override with ``~objects_file``).
Positions are metres in the same world frame as /waypoint_nav/goal, so (x, y) is published
straight through.

This node owns ONLY ROS + UI concerns (topics, the selection window); the catalog parsing
and random choice are the pure, unit-tested core module. See the file footer for the full
rosparam list.
"""
import random
from pathlib import Path

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String

import sparx_agency
from sparx_agency.core.planning.mission import ObjectCatalog


def _default_objects_file():
    """The objects.json shipped next to the FALCON task, resolved via the package.

    Works both on the host and inside the container (the repo is mounted read-only at
    /opt/sparx_agency), so no absolute path is hard-coded.
    """
    return str(Path(sparx_agency.__file__).resolve().parent
               / "tasks" / "planning" / "falcon" / "objects.json")


class MissionDirectorNode(object):
    def __init__(self):
        # disable_signals so a matplotlib GUI (gui mode) owns Ctrl+C; the random-mode
        # hold loop catches KeyboardInterrupt itself (Python's default handler stays
        # installed when rospy does not register its own).
        rospy.init_node("mission_director", disable_signals=True)
        G = rospy.get_param

        self.selection_mode = str(G("~selection_mode", "gui")).strip().lower()
        if self.selection_mode not in ("random", "gui"):
            raise ValueError("~selection_mode must be 'random' or 'gui', got %r"
                             % self.selection_mode)

        self.objects_file = str(G("~objects_file", _default_objects_file()))
        self.target_topic = G("~target_topic", "/object_approach/goal")
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        self.enable_topic = G("~enable_topic", "/object_approach/enable")
        self.status_topic = G("~status_topic", "/mission_director/status")
        # Whether the director also arms/gates object_approach's /enable. Leave true so
        # the mission is a single click; set false if some other node owns the enable.
        self.publish_enable = bool(G("~publish_enable", True))
        seed = int(G("~seed", -1))
        self._rng = random.Random() if seed < 0 else random.Random(seed)

        # ── Load the catalog (fail loud; no silent default) ───────────
        try:
            self.catalog = ObjectCatalog.from_json_file(self.objects_file)
        except (OSError, ValueError) as e:
            rospy.logfatal("mission_director: cannot load objects file %r: %s",
                           self.objects_file, e)
            raise
        if len(self.catalog) == 0:
            rospy.logfatal("mission_director: objects file %r has no objects",
                           self.objects_file)
            raise ValueError("empty object catalog: %s" % self.objects_file)

        self._selected = None

        # ── Publishers (all latched: hold the mission for late joiners; latched
        #    label maps to TRANSIENT_LOCAL across the bridge, which the detector needs).
        self.target_pub = rospy.Publisher(self.target_topic, String, queue_size=1,
                                          latch=True)
        self.goal_pub = rospy.Publisher(self.goal_topic, Point, queue_size=1, latch=True)
        self.enable_pub = rospy.Publisher(self.enable_topic, Bool, queue_size=1,
                                          latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1,
                                          latch=True)

        # THE GATE: assert the mission is DISARMED until a selection. A latched False
        # holds object_approach passive even if it was launched start_enabled:=true, and
        # we publish no goal, so no planner plans. (We only ever publish a goal / enable
        # True on an actual selection.)
        if self.publish_enable:
            self.enable_pub.publish(Bool(data=False))
        self.status_pub.publish(String(data="NOT ARMED -- awaiting object selection"))

        self._banner()

    # ── Selection (publish the mission) ───────────────────────────────
    def _select(self, obj):
        """Arm the mission for ``obj``: publish label + coordinate goal + enable.

        Idempotent per object and safe to call again with a different object to RETARGET
        live (the planner replans, the detector re-prompts, object_approach re-keys).
        """
        self._selected = obj
        self.target_pub.publish(String(data=obj.label))
        self.goal_pub.publish(Point(x=float(obj.x), y=float(obj.y), z=0.0))
        if self.publish_enable:
            self.enable_pub.publish(Bool(data=True))
        line = ("ARMED: hunting %r, flying to (%.2f, %.2f)%s"
                % (obj.label, obj.x, obj.y,
                   "" if self.publish_enable else "  (enable not published)"))
        rospy.loginfo("mission_director: %s", line)
        self.status_pub.publish(String(data=line))

    # ── Run: dispatch on the selection mode ───────────────────────────
    def start(self):
        if self.selection_mode == "gui":
            self._run_gui()
        else:
            self._run_random()

    def _run_random(self):
        obj = self.catalog.random(self._rng)
        rospy.loginfo("mission_director: RANDOM pick from %d objects -> %s",
                      len(self.catalog), obj.caption())
        self._select(obj)
        # Stay alive so the latched publications persist for late-joining subscribers.
        try:
            while not rospy.is_shutdown():
                rospy.sleep(0.5)
        except (rospy.ROSInterruptException, KeyboardInterrupt):
            pass

    # ── GUI (matplotlib clickable list; mirrors bev_click_goal_node) ──
    def _run_gui(self):
        # Imported lazily so random/headless mode needs no display or matplotlib.
        import matplotlib.pyplot as plt

        n = len(self.catalog)
        fig_h = min(2.2 + 0.42 * n, 12.0)
        self.fig, self.ax = plt.subplots(figsize=(8.0, fig_h))
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(-0.5, n - 0.5)
        self.ax.invert_yaxis()               # row 0 at the top
        self.ax.axis("off")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect(
            "close_event",
            lambda _e: rospy.signal_shutdown("mission_director window closed"))

        for i, obj in enumerate(self.catalog):
            self.ax.text(0.03, i, "%2d.  %s" % (i + 1, obj.caption()),
                         va="center", ha="left", fontsize=11, family="monospace")
        self.ax.set_title(
            "Select the object to fly to  (click a row / press its number 1-9 / r=random)\n"
            "Nothing flies until you pick. Click another row anytime to change target.",
            fontsize=10)
        self._highlight = None
        self._status_text = self.fig.text(0.5, 0.012, self._status_line(), ha="center",
                                          va="bottom", fontsize=10, color="0.15")
        # Close the window when ROS shuts down (rosnode kill / roslaunch teardown with
        # no SIGINT to this process). Poll it from a matplotlib timer, which fires on
        # the GUI main thread -- calling plt.close from rospy's XMLRPC shutdown thread
        # would touch Tk off the main thread ("main thread is not in main loop").
        self._shutdown_timer = self.fig.canvas.new_timer(interval=500)
        self._shutdown_timer.add_callback(self._poll_shutdown)
        self._shutdown_timer.start()
        rospy.loginfo("mission_director: GUI up -- select an object to arm the mission")
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    def _poll_shutdown(self):
        """matplotlib-timer (GUI main thread) callback: close the window once ROS shuts
        down. Returns True to keep polling, False to stop the timer once closed."""
        if rospy.is_shutdown():
            import matplotlib.pyplot as plt
            plt.close("all")
            return False
        return True

    def _status_line(self):
        if self._selected is None:
            return "NOT ARMED -- click an object to select it and arm the mission"
        o = self._selected
        return ("ARMED: hunting '%s', flying to (%.2f, %.2f)   -- click another to change"
                % (o.label, o.x, o.y))

    def _select_row(self, row):
        if not (0 <= row < len(self.catalog)):
            return
        self._select(self.catalog[row])
        # Highlight the armed row (recreate the band each time; the artist is cheap).
        if self._highlight is not None:
            self._highlight.remove()
        self._highlight = self.ax.axhspan(row - 0.5, row + 0.5, color="limegreen",
                                          alpha=0.30, zorder=0)
        self._status_text.set_text(self._status_line())
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.ydata is None:
            return
        self._select_row(int(round(event.ydata)))

    def _on_key(self, event):
        """1-9 select that object; ``r`` picks a random one. The numeric keypad reports
        ``kp_1`` etc. (backend/NumLock dependent), so strip that prefix."""
        key = (event.key or "").lower()
        if key.startswith("kp_"):
            key = key[3:]
        if key == "r":
            self._select_row(self._rng.randrange(len(self.catalog)))
        elif key.isdigit() and key != "0":
            self._select_row(int(key) - 1)   # 1-based; >9 objects use click

    # ── Banner ────────────────────────────────────────────────────────
    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("mission_director (select the object to fly to, then arm the mission)")
        L("  mode       = %s", self.selection_mode)
        L("  catalog    = %s  (%d objects: %s)", self.objects_file, len(self.catalog),
          ", ".join(self.catalog.unique_labels()))
        L("  target out = %s   (std_msgs/String label -> YOLO + gate, latched)",
          self.target_topic)
        L("  goal   out = %s   (geometry_msgs/Point x,y -> planners, latched)",
          self.goal_topic)
        L("  enable out = %s   (std_msgs/Bool -> object_approach, latched, %s)",
          self.enable_topic, "on" if self.publish_enable else "NOT published")
        L("  GATE       = nothing published until an object is selected; "
          "object_approach held disabled")
        L("=" * 64)


def main():
    try:
        MissionDirectorNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The catalog parsing + random
# choice are the ROS-free core.planning.mission.ObjectCatalog; this node owns the
# ROS I/O and the selection UI.
#
#   selection: ~selection_mode (gui | random). gui: matplotlib window, click/number/r
#       to select + live retarget. random: pick one uniformly at startup and hold.
#       ~seed (-1 = nondeterministic; >=0 = reproducible random pick)
#   catalog: ~objects_file (…/tasks/planning/falcon/objects.json, resolved via the
#       sparx_agency package so it works on host and in-container). JSON list of
#       {"label", "position_m":{"x","y","z"}, ...}; extra keys ignored; labels
#       normalised (strip+lower). Malformed / empty -> logfatal + raise.
#   outputs (all latched): ~target_topic (/object_approach/goal, String label ->
#       re-prompts YOLO AND re-keys object_approach's gate; latched = TRANSIENT_LOCAL
#       across the bridge, required by the detector) ~goal_topic (/waypoint_nav/goal,
#       Point x,y -> planners replan) ~enable_topic (/object_approach/enable, Bool ->
#       arm object_approach) ~status_topic (/mission_director/status, String)
#   gating: ~publish_enable (true = the director arms/gates object_approach's /enable;
#       it publishes False at startup, True on selection. false = leave /enable to
#       another owner and only publish target+goal on selection).
#
# THE GATE. Launch the planners with NO ~goal_x/~goal_y (they idle "until a click") and
# object_approach start_enabled:=false. This node publishes no goal and holds enable
# False until a selection, so nothing plans/flies until you pick. It relies on a FRESH
# roscore (no stale latched /waypoint_nav/goal from a previous run); run_falcon.sh's
# --rm container gives one per launch.
# ============================================================================
