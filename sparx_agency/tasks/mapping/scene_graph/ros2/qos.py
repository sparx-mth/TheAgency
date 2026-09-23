"""The two QoS profiles every scene-graph node uses, written once.

**Latched (``latched_qos``) is a correctness requirement, not a style
choice.** The scene-graph verdicts — the room map, the target latch, the
oracle's ranking, the follower claim on ``/cmd_vel`` — are *state*, not a
stream: each moves a handful of times in a whole mission, and a subscriber
that comes up late must still learn the current value. ``TRANSIENT_LOCAL``
is that promise, and ``depth=1`` keeps it to the newest verdict so a backlog
of stale claims can never be replayed at the aircraft.

Several of these topics also cross ``ros1_bridge`` into the FALCON Noetic
container, which is restarted independently of the ROS 2 stack. There a QoS
mismatch is **not an error — it is silently zero data**: a ``VOLATILE``
reader of a ``TRANSIENT_LOCAL`` writer receives nothing at all until the
writer's next change, which for a flag that moves twice a mission can be the
whole flight. ``tasks/planning/falcon_sjtu/config/bridge.yaml`` declares the
matching ROS 1 side per topic; the two must agree, so change neither alone.

``sensor_qos`` is the opposite contract for the high-rate camera and odometry
streams: ``BEST_EFFORT`` / ``VOLATILE``, because the sim's publishers are
best-effort and a reliable reader of a best-effort writer also receives
nothing.
"""
from __future__ import annotations

from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)


def latched_qos() -> QoSProfile:
    """RELIABLE + TRANSIENT_LOCAL + KEEP_LAST, depth 1.

    Returns:
        A fresh profile for a latched state topic (see the module docstring
        for why ``TRANSIENT_LOCAL`` is mandatory across ``ros1_bridge``).
    """
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


def sensor_qos(depth: int = 5) -> QoSProfile:
    """BEST_EFFORT + VOLATILE + KEEP_LAST for a high-rate sensor stream.

    Args:
        depth: KEEP_LAST queue depth. Defaults to the 5 the scene-graph
            image/odometry subscribers use.

    Returns:
        A fresh profile matching the Gazebo publishers' best-effort contract.
    """
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)
