"""The aircraft's end of the link to FALCON.

One rule shapes this file: **nothing here may block ``world.step()``.** Isaac
Sim drives PX4 SITL's lockstep clock, so a simulation that stalls stops the
autopilot, which stops answering, which is the deadlock the whole simulator was
written to avoid. A slow or dead FALCON must cost dropped frames, never a
stalled aircraft.

So sending happens on a background thread behind two queues with very different
policies:

* **depth frames: keep the newest, drop the rest.** A stale depth frame is worse
  than no frame -- FALCON would fuse it against a pose the aircraft has already
  left -- and there is never a reason to send two.
* **odometry and events: a short FIFO.** They are tiny, and ordering matters:
  an ``exploration finished`` behind a backlog of poses would arrive late.

Receiving is polled, not threaded, so the reference the controller reads is
always the newest one that had arrived by the start of this control tick, and
never changes underneath it mid-computation.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional, Tuple

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.socket_link import (
    DOWNLINK_PORT, UPLINK_PORT, connect,
)

CONTROL_QUEUE_DEPTH = 64


class FalconLink:
    """Connects to the FALCON bridge and exchanges state for references.

    Args:
        uplink_port: Where the bridge listens for depth and odometry.
        downlink_port: Where it listens for the aircraft to collect commands.
        connect_timeout_s: How long to keep retrying before giving up.
    """

    def __init__(self, uplink_port: int = UPLINK_PORT, downlink_port: int = DOWNLINK_PORT,
                 connect_timeout_s: float = 180.0):
        self._uplink_port = uplink_port
        self._downlink_port = downlink_port
        self._timeout_s = connect_timeout_s
        self._uplink = None
        self._downlink = None

        self._pending_frame = None                      # type: Optional[bytes]
        self._pending_control = deque(maxlen=CONTROL_QUEUE_DEPTH)
        self._wake = threading.Condition()
        self._sender = None                             # type: Optional[threading.Thread]
        self._running = False

        self.reference = None                           # type: Optional[TrajectoryPoint]
        self.reference_stamp_s = None                   # type: Optional[float]
        # The most recently planned curve, or None if the trajectory stream is
        # not being used. Separate from `reference` on purpose: the two control
        # paths consume different things from the same link, and a run can be
        # flown either way without restarting the FALCON side.
        self.trajectory = None                          # type: Optional[BsplineTrajectory]
        self.trajectories_received = 0
        self.trajectory_id = 0
        self.commands_received = 0
        self.frames_sent = 0
        self.frames_dropped = 0
        self.events = []                                # type: List[Tuple[str, str]]

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self, intrinsics, scene: str, run: str) -> None:
        """Open both sockets and announce the camera.

        The uplink is connected first, matching the order the bridge accepts in.
        The ``HELLO`` goes out synchronously, before the sender thread starts, so
        it cannot be reordered behind a depth frame -- the bridge validates the
        camera from it and refuses the run on a mismatch, which it can only do if
        it arrives first.

        Args:
            intrinsics: The camera the depth frames will be taken with.
            scene: Isaac scene name, for the bridge's log.
            run: Run name, for the bridge's log.

        Raises:
            RuntimeError: If the bridge never accepted. See
                :func:`~..link.socket_link.connect` for what that usually means.
        """
        self._uplink = connect(self._uplink_port, self._timeout_s, name="uplink")
        self._downlink = connect(self._downlink_port, self._timeout_s, name="downlink")
        self._uplink.send(protocol.hello(intrinsics, scene, run))
        self._running = True
        self._sender = threading.Thread(target=self._pump, name="falcon-uplink",
                                        daemon=True)
        self._sender.start()

    def close(self) -> None:
        """Stop the sender and drop both sockets. Safe to call twice."""
        with self._wake:
            self._running = False
            self._wake.notify_all()
        if self._sender is not None:
            self._sender.join(timeout=2.0)
            self._sender = None
        for endpoint in (self._uplink, self._downlink):
            if endpoint is not None:
                endpoint.close()
        self._uplink = self._downlink = None

    @property
    def alive(self) -> bool:
        """True while the bridge is still on the other end of both sockets."""
        return (self._uplink is not None and not self._uplink.closed
                and self._downlink is not None and not self._downlink.closed)

    # ── sending ──────────────────────────────────────────────────────────

    def send_frame(self, stamp_s: float, width: int, height: int, translation,
                   quaternion_xyzw, depth_bytes: bytes) -> None:
        """Queue one depth frame and the camera pose that took it.

        Replaces any frame still waiting. See this module's docstring.
        """
        message = protocol.frame(stamp_s, width, height, translation,
                                 quaternion_xyzw, depth_bytes)
        with self._wake:
            if self._pending_frame is not None:
                self.frames_dropped += 1
            self._pending_frame = message
            self._wake.notify()

    def send_odometry(self, stamp_s: float, position, quaternion_xyzw,
                      linear_velocity, angular_velocity) -> None:
        """Queue the aircraft's world-frame state.

        ``linear_velocity`` must be in the **world** frame. FALCON reads it
        straight into the initial velocity of its next B-spline without rotating
        it by the attitude, so a body-frame twist -- which is what REP-147 says a
        ``nav_msgs/Odometry`` carries -- would send every replan off sideways.
        """
        self._queue(protocol.odometry(stamp_s, position, quaternion_xyzw,
                                      linear_velocity, angular_velocity))

    def send_event(self, name: str, detail: str = "") -> None:
        """Queue a named event for the bridge."""
        self._queue(protocol.event(name, detail))

    def _queue(self, message: bytes) -> None:
        with self._wake:
            self._pending_control.append(message)
            self._wake.notify()

    def _pump(self) -> None:
        """Background sender: control messages first, then the newest frame."""
        while True:
            with self._wake:
                while self._running and not self._pending_control and self._pending_frame is None:
                    self._wake.wait(0.2)
                if not self._running:
                    return
                control = list(self._pending_control)
                self._pending_control.clear()
                frame, self._pending_frame = self._pending_frame, None
            uplink = self._uplink
            if uplink is None or uplink.closed:
                continue
            for message in control:
                uplink.send(message)
            if frame is not None and uplink.send(frame):
                self.frames_sent += 1

    # ── receiving ────────────────────────────────────────────────────────

    def poll(self) -> List[Tuple[str, str]]:
        """Drain the downlink and update :attr:`reference` and :attr:`trajectory`.

        Only the newest of each is kept: the controller runs at the physics rate
        and wants the freshest state, not a backlog of the ones it missed. That
        is safe for both -- a command is a snapshot, and a trajectory supersedes
        its predecessor outright.

        Returns:
            The ``(name, detail)`` events that arrived in this call.
        """
        if self._downlink is None:
            return []
        events = []
        for kind, header, _payload in self._downlink.poll(timeout=0.0):
            if kind == protocol.KIND_POSCMD:
                self.reference = _to_trajectory_point(header)
                self.reference_stamp_s = float(header["t"])
                self.trajectory_id = int(header["id"])
                self.commands_received += 1
            elif kind == protocol.KIND_BSPLINE:
                self.trajectory = _to_trajectory(header)
                self.trajectories_received += 1
                # Also advances the id, so `has_trajectory` and the mission's
                # planner-stall watchdog work identically on either control path.
                self.trajectory_id = max(self.trajectory_id, int(header["id"]))
            elif kind == protocol.KIND_EVENT:
                events.append((header.get("name", ""), header.get("detail", "")))
        self.events.extend(events)
        return events

    @property
    def has_trajectory(self) -> bool:
        """True once FALCON has actually planned something.

        Not the same as "a command has arrived". Before any B-spline exists,
        ``traj_server`` publishes ~200 commands parking the aircraft at the map
        config's init pose, and they carry ``trajectory_id == 0`` -- upstream's
        own convention is that real trajectory ids start at 1, "allowing you
        comparing it with 0" (``quadrotor_msgs/PositionCommand.msg``). Treating
        the parked command as a plan makes the aircraft hand over to a planner
        that has not planned, and then time out when the start-up burst ends.
        """
        return self.trajectory_id >= 1

    def reference_age_s(self, now_s: float) -> float:
        """How stale the held reference is, seconds. Infinite if there is none."""
        if self.reference_stamp_s is None:
            return float("inf")
        return max(now_s - self.reference_stamp_s, 0.0)


def _to_trajectory(header: dict) -> BsplineTrajectory:
    """One ``BSPLINE`` header rebuilt into the curve FALCON planned.

    Uses the same construction rules ``traj_server`` applies to the same
    message, so both sides of the link evaluate the identical polynomial.
    """
    return BsplineTrajectory.from_falcon(
        order=int(header["order"]),
        knots=header["knots"],
        position_points=header["pos"],
        yaw_points=header["yaw"],
        yaw_dt=float(header["yaw_dt"]),
        start_time_s=float(header["t"]),
        traj_id=int(header["id"]))


def _to_trajectory_point(header: dict) -> TrajectoryPoint:
    """One ``POSCMD`` header as the repo's shared trajectory type."""
    position, velocity, acceleration = header["p"], header["v"], header["a"]
    return TrajectoryPoint(
        t=float(header["t"]),
        x=position[0], y=position[1], z=position[2],
        vx=velocity[0], vy=velocity[1], vz=velocity[2],
        ax=acceleration[0], ay=acceleration[1], az=acceleration[2],
        yaw=float(header["yaw"]), yaw_rate=float(header["yaw_dot"]),
    )
