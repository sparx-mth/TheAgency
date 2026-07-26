#!/usr/bin/env python3
"""ROS1 subscriber that turns the localization topics into a quality snapshot.

The ROS2 localization node publishes four signals alongside every pose, in the
same callback, one set per camera frame (~10 Hz):

    /xtend/localization_confidence          Float32  0..1
    /xtend/localization_pos_std             Float32  metres
    /xtend/localization_cmd_effectiveness   Float32  0..1
    /xtend/localization_source              String   "apriltag" | "apriltag_coast" | ...

This class owns the subscriptions and hands the controller a single immutable
:class:`LocalizationQuality`. It is deliberately the *only* place in the ROS1
stack that knows those topic names.

Three things it gets right that are easy to get wrong:

* **Silence is a signal.** A rejected frame, an exhausted coast budget or a
  non-AprilTag provider all publish *nothing*. Age is therefore measured from
  arrival time here, not read off a stamp, and a stream that stops is reported as
  old rather than as "still whatever it last said".
* **Coasting is read from the source string, never inferred from confidence.** A
  coasted pose caps at 0.25 and a genuine single-tag fix caps near 0.21, so the
  numbers cannot separate them. Only the string can.
* **Missing topics must not brick the drone.** If the signals never arrive at all
  — an older localization node, or a `bridge.yaml` without the entries — the
  monitor reports full confidence and warns loudly, so the controller degrades to
  exactly how every other controller in this stack already flies. Set
  ``require=True`` to demand them instead and hold the drone until they appear.
"""
import rospy
from std_msgs.msg import Float32, String

from sparx_agency.core.planning.trackers.drift_pid import LocalizationQuality

#: Value of /xtend/localization_source that means "this pose is dead reckoning".
COAST_SOURCE = "apriltag_coast"


class LocalizationQualityMonitor(object):
    """Subscribes the localization quality topics and snapshots them on demand."""

    def __init__(self, conf_topic="/xtend/localization_confidence",
                 std_topic="/xtend/localization_pos_std",
                 eff_topic="/xtend/localization_cmd_effectiveness",
                 source_topic="/xtend/localization_source",
                 require=False, warn_after_s=5.0):
        """Create the monitor and subscribe.

        Args:
            conf_topic: Pose-confidence topic (Float32, 0..1).
            std_topic: Position-standard-deviation topic (Float32, metres).
            eff_topic: Command-effectiveness topic (Float32, 0..1).
            source_topic: Provider/source topic (String).
            require: True holds the drone until the signals actually arrive.
                False (default) degrades to full confidence with a loud warning,
                which is how the rest of this stack already flies.
            warn_after_s: How long to wait before complaining that nothing has
                arrived (s).
        """
        self.require = bool(require)
        self.warn_after_s = float(warn_after_s)

        self._conf = 0.0
        self._std = 1.0
        self._eff = 1.0
        self._source = ""
        self._last_t = None          # arrival time of the last confidence message
        self._start_t = rospy.Time.now()

        rospy.Subscriber(conf_topic, Float32, self._conf_cb, queue_size=10)
        rospy.Subscriber(std_topic, Float32, self._std_cb, queue_size=10)
        rospy.Subscriber(eff_topic, Float32, self._eff_cb, queue_size=10)
        rospy.Subscriber(source_topic, String, self._source_cb, queue_size=10)
        rospy.loginfo("localization_quality: watching %s | %s | %s | %s "
                      "(require=%s)", conf_topic, std_topic, eff_topic,
                      source_topic, self.require)

    # ── Callbacks ────────────────────────────────────────────────
    def _conf_cb(self, msg):
        # Confidence is published with the pose on every fix, so its arrival is
        # the freshness clock for the whole set.
        self._conf = float(msg.data)
        self._last_t = rospy.Time.now()

    def _std_cb(self, msg):
        self._std = float(msg.data)

    def _eff_cb(self, msg):
        self._eff = float(msg.data)

    def _source_cb(self, msg):
        self._source = (msg.data or "").strip().lower()

    # ── Snapshot ─────────────────────────────────────────────────
    @property
    def seen(self):
        """True once any localization quality message has arrived."""
        return self._last_t is not None

    def snapshot(self):
        """Return the current :class:`LocalizationQuality`.

        Returns:
            An immutable snapshot. When nothing has ever arrived and ``require``
            is False, this is a fully-permissive snapshot (and a throttled warning
            is emitted), so the controller flies as the rest of the stack does.
        """
        now = rospy.Time.now()
        if self._last_t is None:
            waited = (now - self._start_t).to_sec()
            if waited > self.warn_after_s:
                rospy.logwarn_throttle(
                    30.0,
                    "localization_quality: no confidence signal after %.0fs. %s "
                    "Check that bridge.yaml bridges "
                    "/xtend/localization_confidence (+ _pos_std, "
                    "_cmd_effectiveness, _source) and that the ROS2 "
                    "localization node is running.",
                    waited,
                    "HOLDING the drone (require=true)." if self.require else
                    "Flying WITHOUT confidence gating -- no slow-down on a poor "
                    "pose, no drift-learning freeze while coasting.")
            if self.require:
                return LocalizationQuality(valid=False)
            return LocalizationQuality(confidence=1.0, pos_std_m=0.0, age_s=0.0,
                                       coasting=False, cmd_effectiveness=1.0,
                                       valid=True)
        return LocalizationQuality(
            confidence=self._conf,
            pos_std_m=self._std,
            age_s=(now - self._last_t).to_sec(),
            coasting=(self._source == COAST_SOURCE),
            cmd_effectiveness=self._eff,
            valid=True)
