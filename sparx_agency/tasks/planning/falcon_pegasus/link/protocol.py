"""The wire format between Isaac Sim and the FALCON ROS1 container.

Two processes that cannot import each other's world: Isaac Sim's Python 3.12
(numpy 2.5, no ROS at all) and ROS Noetic's Python 3.8 (numpy 1.17, rospy). They
run in separate containers that share the host network namespace, so plain TCP on
``127.0.0.1`` reaches across with nothing in between.

The format is deliberately the smallest thing that works, because the only
libraries guaranteed on both sides are the standard library and numpy:

.. code-block:: text

    <4s  magic  b"FLCN">
    <B   version>
    <B   kind>
    <H   reserved, 0>
    <I   header length>      UTF-8 JSON, the metadata
    <I   payload length>     raw little-endian array bytes, or nothing
    ------------------------- 16 bytes -------------------------
    <header bytes><payload bytes>

Everything structured travels as JSON, which both interpreters agree on exactly.
The one bulk field -- the depth image -- travels as raw ``tobytes()`` with its
dtype named in the JSON, because a ``.npy`` or a pickle written by numpy 2.5 is
not something numpy 1.17 is obliged to read, while a block of little-endian
``uint16`` is the same block of bytes in every version there has ever been.

**This module must stay Python 3.8 compatible.** It is imported by a rospy node
inside a ``ros:noetic`` container, exactly like ``core/`` is -- so no PEP 604
unions outside ``from __future__ import annotations``, no ``match``, no
``dataclass(slots=True)``.
"""
from __future__ import annotations

import json
import struct

MAGIC = b"FLCN"
VERSION = 1

_HEADER = struct.Struct("<4sBBHII")
HEADER_SIZE = _HEADER.size

KIND_HELLO = 0
"""Isaac -> FALCON, once per connection. Announces the camera it will send."""
KIND_FRAME = 1
"""Isaac -> FALCON, at camera rate. Depth image + the camera pose that took it."""
KIND_ODOM = 2
"""Isaac -> FALCON, fast. The aircraft's ground-truth state."""
KIND_POSCMD = 3
"""FALCON -> Isaac, 100 Hz. One ``quadrotor_msgs/PositionCommand``."""
KIND_EVENT = 4
"""Either direction. A named, structured thing happened -- see :data:`EVENTS`."""
KIND_BSPLINE = 5
"""FALCON -> Isaac, on every replan. The trajectory itself, not a sample of it.

Carried *alongside* :data:`KIND_POSCMD` rather than instead of it, so both
control paths stay flyable from one run of the FALCON side and can be compared
without rebuilding anything."""

KIND_NAMES = {
    KIND_HELLO: "hello", KIND_FRAME: "frame", KIND_ODOM: "odom",
    KIND_POSCMD: "poscmd", KIND_EVENT: "event", KIND_BSPLINE: "bspline",
}

EVENT_EXPLORATION_FINISHED = "exploration_finished"
"""FALCON declared the space explored. Its trajectory server exits; land."""
EVENT_PLANNER_GONE = "planner_gone"
"""The command stream stopped without a finish. Hold, then land."""
EVENT_MISSION_OVER = "mission_over"
"""Isaac is done flying. The ROS1 side should close its recording."""
EVENT_TRAJECTORY_UNSAFE = "trajectory_unsafe"
"""FALCON found an obstacle ON the trajectory the aircraft is currently flying.

Its own answer is to replan, which is right and also incomplete: replanning takes
a hundred milliseconds or more, and upstream that cost nothing because the
geometry simulator's aircraft stopped the instant its reference did. An aircraft
with mass carries its momentum into the obstacle while the planner thinks. So the
aircraft is told, and holds station until a new trajectory arrives."""

EVENTS = (EVENT_EXPLORATION_FINISHED, EVENT_PLANNER_GONE, EVENT_MISSION_OVER,
          EVENT_TRAJECTORY_UNSAFE)

DEPTH_DTYPE = "<u2"
"""Depth pixels on the wire: little-endian uint16 millimetres, 0 = no return.

This is the exact type FALCON's back-projection loop reads
(``map_server.cpp``: ``depth_image.ptr<uint16_t>(v)`` then ``* 0.001``), so
converting here rather than publishing 32FC1 removes a per-frame ``convertTo``
inside the mapper and halves the bytes on the wire.
"""

MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 64 << 20


class ProtocolError(Exception):
    """The stream does not contain what this protocol says it should."""


def encode(kind, header, payload=b""):
    # type: (int, dict, bytes) -> bytes
    """Serialise one message.

    Args:
        kind: One of the ``KIND_*`` constants.
        header: JSON-serialisable metadata.
        payload: Raw bulk bytes, or empty.

    Returns:
        The complete framed message.

    Raises:
        ProtocolError: If ``kind`` is not a known kind.
    """
    if kind not in KIND_NAMES:
        raise ProtocolError("unknown message kind %r" % (kind,))
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(MAGIC, VERSION, kind, 0, len(blob), len(payload)) + blob + payload


class Decoder:
    """Reassembles messages from a byte stream that arrives in arbitrary chunks.

    TCP has no message boundaries, so every reader needs one of these. Feed it
    whatever ``recv`` returned and drain whatever became complete:

    .. code-block:: python

        decoder = Decoder()
        for kind, header, payload in decoder.feed(sock.recv(65536)):
            ...
    """

    def __init__(self):
        # type: () -> None
        self._buffer = bytearray()

    def feed(self, chunk):
        # type: (bytes) -> list
        """Add received bytes and return every message they completed.

        Args:
            chunk: Bytes straight off the socket.

        Returns:
            A list of ``(kind, header_dict, payload_bytes)`` tuples, in order.

        Raises:
            ProtocolError: If the stream does not start with the magic, carries
                an unknown version, or declares an implausible length. All three
                mean the stream is desynchronised, and continuing would silently
                interpret noise as depth.
        """
        self._buffer.extend(chunk)
        messages = []
        while True:
            message = self._take_one()
            if message is None:
                return messages
            messages.append(message)

    def _take_one(self):
        # type: () -> object
        """Pop one complete message off the front of the buffer, or None."""
        if len(self._buffer) < HEADER_SIZE:
            return None
        magic, version, kind, _reserved, header_len, payload_len = _HEADER.unpack_from(
            self._buffer, 0)
        if magic != MAGIC:
            raise ProtocolError(
                "stream desynchronised: expected magic %r, got %r" % (MAGIC, magic))
        if version != VERSION:
            raise ProtocolError(
                "peer speaks protocol version %d, this is version %d" % (version, VERSION))
        if header_len > MAX_HEADER_BYTES or payload_len > MAX_PAYLOAD_BYTES:
            raise ProtocolError(
                "implausible message: header %d B, payload %d B" % (header_len, payload_len))
        total = HEADER_SIZE + header_len + payload_len
        if len(self._buffer) < total:
            return None
        header = json.loads(
            bytes(self._buffer[HEADER_SIZE:HEADER_SIZE + header_len]).decode("utf-8"))
        payload = bytes(self._buffer[HEADER_SIZE + header_len:total])
        del self._buffer[:total]
        return kind, header, payload


def hello(intrinsics, scene, run):
    # type: (object, str, str) -> bytes
    """The opening message: which camera the depth frames were taken with.

    The ROS1 side compares this against the ``/uav_model/sensing_parameters``
    rosparams FALCON will back-project with, and refuses to run on a mismatch.
    Intrinsics that disagree do not produce an error anywhere in FALCON -- they
    produce a complete, plausible-looking map of a building that is the wrong
    size, which is the worst failure this system can have.

    Args:
        intrinsics: A :class:`~sparx_agency.core.common.types.Intrinsics`.
        scene: The Isaac scene being explored, for the log.
        run: The campaign run name, for the log.

    Returns:
        A framed ``HELLO`` message.
    """
    return encode(KIND_HELLO, {
        "width": int(intrinsics.width), "height": int(intrinsics.height),
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx), "cy": float(intrinsics.cy),
        "depth_dtype": DEPTH_DTYPE, "depth_units": "mm",
        "scene": str(scene), "run": str(run),
    })


def frame(stamp_s, width, height, translation, quaternion_xyzw, depth_bytes):
    # type: (float, int, int, tuple, tuple, bytes) -> bytes
    """A depth image and the camera pose that took it, sharing one timestamp.

    The single timestamp is the whole point. FALCON matches a depth image to a
    camera pose within 1 ms or refuses to fuse it, so carrying one capture time
    per frame and stamping both published messages with it makes that tolerance
    structurally impossible to violate, whatever the link does.

    Args:
        stamp_s: Capture time, seconds on the wall clock both containers share.
        width: Image width, pixels.
        height: Image height, pixels.
        translation: World-frame ``(x, y, z)`` of the camera **optical** frame.
        quaternion_xyzw: World-from-optical rotation, scalar last.
        depth_bytes: ``height * width`` :data:`DEPTH_DTYPE` values.

    Returns:
        A framed ``FRAME`` message.
    """
    return encode(KIND_FRAME, {
        "t": float(stamp_s), "w": int(width), "h": int(height),
        "p": [float(v) for v in translation],
        "q": [float(v) for v in quaternion_xyzw],
        "dtype": DEPTH_DTYPE,
    }, depth_bytes)


def odometry(stamp_s, position, quaternion_xyzw, linear_velocity, angular_velocity):
    # type: (float, tuple, tuple, tuple, tuple) -> bytes
    """The aircraft's world-frame state.

    ``linear_velocity`` must be in the **world** frame, not the body frame.
    FALCON reads ``twist.twist.linear`` straight into the start state of its next
    B-spline without rotating it by the orientation -- which is the opposite of
    what REP-147 says a ``nav_msgs/Odometry`` twist means, and would put every
    replan's initial velocity in the wrong direction.

    Args:
        stamp_s: Wall-clock seconds.
        position: World ``(x, y, z)``, metres.
        quaternion_xyzw: World-from-body rotation, scalar last.
        linear_velocity: World-frame ``(vx, vy, vz)``, m/s.
        angular_velocity: Body-frame ``(wx, wy, wz)``, rad/s.

    Returns:
        A framed ``ODOM`` message.
    """
    return encode(KIND_ODOM, {
        "t": float(stamp_s),
        "p": [float(v) for v in position],
        "q": [float(v) for v in quaternion_xyzw],
        "v": [float(v) for v in linear_velocity],
        "w": [float(v) for v in angular_velocity],
    })


def position_command(stamp_s, traj_id, position, velocity, acceleration, yaw, yaw_dot):
    # type: (float, int, tuple, tuple, tuple, float, float) -> bytes
    """One reference state from FALCON's trajectory server.

    A transcription of ``quadrotor_msgs/PositionCommand``, minus the fields
    ``traj_server`` never fills (``jerk`` is always zero, ``trajectory_flag`` is
    set once at start-up and never updated).

    Args:
        stamp_s: Wall-clock seconds the command was published at.
        traj_id: Which trajectory this point belongs to. Changes on every replan.
        position: World ``(x, y, z)``, metres.
        velocity: World ``(vx, vy, vz)``, m/s.
        acceleration: World ``(ax, ay, az)``, m/s^2.
        yaw: Reference heading, radians CCW from +x.
        yaw_dot: Reference yaw rate, rad/s.

    Returns:
        A framed ``POSCMD`` message.
    """
    return encode(KIND_POSCMD, {
        "t": float(stamp_s), "id": int(traj_id),
        "p": [float(v) for v in position],
        "v": [float(v) for v in velocity],
        "a": [float(v) for v in acceleration],
        "yaw": float(yaw), "yaw_dot": float(yaw_dot),
    })


def bspline(start_time_s, traj_id, order, knots, position_points, yaw_points, yaw_dt):
    # type: (float, int, int, object, object, object, float) -> bytes
    """One whole trajectory, as FALCON's ``trajectory/Bspline`` message carries it.

    Small enough for JSON: a minute-long exploration leg is a few dozen control
    points, so this is a couple of kilobytes arriving a handful of times a
    second -- against a depth frame every 80 ms on the same link.

    The asymmetry between the two curves is FALCON's and is preserved exactly.
    The **position** curve carries an explicit knot vector, because the
    optimiser reparameterises it to respect the velocity limit and its knots are
    genuinely unevenly spaced. The **yaw** curve carries only an interval,
    because nothing reparameterises it. Sending the position curve's knots for
    both would fly the right shape at the wrong speed.

    Args:
        start_time_s: Wall-clock instant the trajectory begins. Normally a
            little in the *future*: FALCON starts each curve a planning-time
            ahead so it joins the one still being flown.
        traj_id: Monotonically increasing trajectory identifier.
        order: Degree of the position curve; 3 in practice.
        knots: The position curve's knot vector.
        position_points: ``(N, 3)`` control points, metres.
        yaw_points: ``(M,)`` yaw control points, radians.
        yaw_dt: The yaw curve's uniform knot interval, seconds.

    Returns:
        A framed ``BSPLINE`` message.
    """
    return encode(KIND_BSPLINE, {
        "t": float(start_time_s), "id": int(traj_id), "order": int(order),
        "knots": [float(v) for v in knots],
        "pos": [[float(c) for c in point] for point in position_points],
        "yaw": [float(v) for v in yaw_points],
        "yaw_dt": float(yaw_dt),
    })


def event(name, detail=""):
    # type: (str, str) -> bytes
    """Announce something that changes what the other side should do.

    Args:
        name: One of :data:`EVENTS`.
        detail: Free text for the log.

    Returns:
        A framed ``EVENT`` message.

    Raises:
        ProtocolError: If ``name`` is not a known event. An unrecognised event
            would be silently ignored by the receiver, so it is rejected here
            where the mistake is visible.
    """
    if name not in EVENTS:
        raise ProtocolError("unknown event %r; known events are %r" % (name, EVENTS))
    return encode(KIND_EVENT, {"name": name, "detail": str(detail)})
