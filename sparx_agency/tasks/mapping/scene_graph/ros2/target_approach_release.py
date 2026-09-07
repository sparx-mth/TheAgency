"""Release the ``cmd_vel`` mute even after this process's rclpy context is gone.

One function, and it exists because of one measured fact: **an rclpy node
cannot publish anything from its own teardown after a signal.** ``rclpy.init``
installs handlers for SIGINT *and* SIGTERM, and both shut the context down
before ``spin()`` returns, so the tidy "publish ``False`` in ``destroy_node``"
release raises instead of sending::

    RELEASE FAILED RCLError: Failed to publish: publisher's context is
    invalid, at ./src/rcl/publisher.c:423

Measured on this machine with a latched ``std_msgs/Bool`` and a live
subscriber, under both signals: the subscriber received every ``True`` and
never the ``False``.

That matters here more than it usually would. ``stop_scene_graph.sh`` stops the
host nodes with a plain ``kill`` — SIGTERM — so the *normal operator stop* is
exactly the failing case. Stop the stack while the target approach is flying
and the FALCON b-spline follower stays muted, publishing nothing, while the
SJTU plugin holds the last twist it was handed: the aircraft keeps flying at
its approach speed until the follower's ``~external_ctrl_timeout_s`` lapses.
The lease is a real backstop, but it is a *slow* one, and it should not be the
thing that catches the most ordinary way this mission ends.

So when the node's own publisher can no longer carry the release, we build a
throwaway ROS 2 participant on a **fresh context** — unaffected by the shutdown
of the default one — publish the ``False`` on it, and hold the process open
just long enough for the reader (the ros1_bridge) to be discovered and the
sample delivered. Measured cost on this machine: ~0.45 s to discovery, ~0.6 s
of teardown in total, comfortably inside the 2 s ``stop_scene_graph.sh`` allows
between its SIGTERM and its SIGKILL.

This is a last resort, not the normal path: it runs only when the node was
actually flying the aircraft *and* the ordinary publish already failed. A node
that idled through a whole flight never reaches it, and a hard ``kill -9``
still cannot run it — for that, the follower's lease remains the only answer.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.context import Context
from std_msgs.msg import Bool

from sparx_agency.tasks.mapping.scene_graph.ros2.qos import latched_qos

RELEASE_QOS = latched_qos()
"""Must match the mute publisher's QoS exactly, and ``config/bridge.yaml``.

A volatile writer against that transient_local bridge entry is not merely
slower, it delivers nothing at all until its next change — and this writer has
no next change, it publishes once and dies.
"""

DISCOVERY_TIMEOUT_S = 1.5
"""How long to wait for the reader before giving up and letting the lease win.

Bounded on purpose: this runs inside a signal teardown that
``stop_scene_graph.sh`` follows with a SIGKILL after 2 s.
"""

DELIVERY_SETTLE_S = 0.2
"""Held open after the last publish so the sample actually leaves the writer."""


def emergency_release(topic: str, node_name: str = "target_approach_release",
                      discovery_timeout_s: float = DISCOVERY_TIMEOUT_S) -> bool:
    """Publish a single latched ``False`` on ``topic`` from a fresh context.

    Never raises: it is called from a teardown path that must finish whatever
    happens, and a failure here only means falling back to the follower's
    staleness lease, which is exactly where we already were.

    Args:
        topic: The mute topic to release (``/scene_graph/external_ctrl``).
        node_name: Name for the throwaway node.
        discovery_timeout_s: Seconds to wait for a subscriber to appear before
            publishing again and leaving. The ``False`` is published once
            immediately regardless, so a reader already discovered gets it at
            once.

    Returns:
        True if the release was published and at least one subscriber had been
        discovered by the time we left; False otherwise (including every
        failure), meaning the follower's lease must be what recovers it.
    """
    context = Context()
    node = None
    try:
        rclpy.init(context=context)
        node = rclpy.create_node(node_name, context=context)
        pub = node.create_publisher(Bool, topic, RELEASE_QOS)
        pub.publish(Bool(data=False))
        deadline = time.time() + max(0.0, float(discovery_timeout_s))
        while time.time() < deadline and pub.get_subscription_count() == 0:
            time.sleep(0.05)
        # Published again after discovery: the first sample went out before any
        # reader was known, and while transient_local is meant to carry it to a
        # late joiner, a second copy costs nothing and this is the message that
        # decides whether the aircraft keeps flying.
        pub.publish(Bool(data=False))
        seen = pub.get_subscription_count() > 0
        time.sleep(DELIVERY_SETTLE_S)
        return bool(seen)
    except Exception as exc:                # noqa: BLE001 -- teardown must not raise
        print("[target_approach] emergency mute release on %s failed (%s: %s); "
              "the follower's staleness lease must lapse it"
              % (topic, type(exc).__name__, exc))
        return False
    finally:
        try:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown(context=context)
        except Exception:                   # noqa: BLE001 -- nothing left to do
            pass
