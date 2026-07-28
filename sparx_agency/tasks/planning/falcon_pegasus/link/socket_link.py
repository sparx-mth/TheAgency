"""TCP endpoints for the Isaac <-> FALCON link, on shared-loopback ``127.0.0.1``.

Both containers run with ``--network host``, so they share one loopback device
and one port space: no publishing, no bridge network, no ``host.docker.internal``.
That is the whole reason this can be a plain socket.

Two rules shape everything here.

**The FALCON side listens; Isaac connects.** The ROS1 stack is up in seconds and
stays up for the whole campaign; Isaac Sim takes minutes to load a stage and may
be restarted between runs. Whoever takes longer to appear should be the one that
retries.

**Two ports, not one.** A 500 kB depth frame in flight on a shared stream
head-of-line-blocks the 100 Hz command behind it. Uplink and downlink get their
own socket and both get ``TCP_NODELAY``.

Nothing here ever blocks for long. On the Isaac side a blocked send would stall
``world.step()``, and PX4 SITL's lockstep clock is driven by ``world.step()`` --
so a slow reader would stop the simulation, which would stop the autopilot, which
is the deadlock class this whole simulator was carefully written to avoid.

Kept Python 3.8 compatible: the ROS1 node imports it.
"""
from __future__ import annotations

import errno
import socket
import time

from sparx_agency.tasks.planning.falcon_pegasus.link.protocol import Decoder

UPLINK_PORT = 5599
"""Isaac -> FALCON: depth frames, camera poses, odometry."""
DOWNLINK_PORT = 5600
"""FALCON -> Isaac: position commands and events."""

_SEND_BUFFER_BYTES = 4 << 20


def _configure(sock):
    # type: (socket.socket) -> socket.socket
    """Apply the options every socket in this link wants."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SEND_BUFFER_BYTES)
    return sock


class LinkEndpoint:
    """A connected socket plus the decoder that reassembles messages from it.

    Args:
        sock: An established TCP connection.
        name: Label used in error messages.
    """

    def __init__(self, sock, name="link"):
        # type: (socket.socket, str) -> None
        self._sock = _configure(sock)
        self._decoder = Decoder()
        self._name = name
        self.closed = False

    def send(self, message):
        # type: (bytes) -> bool
        """Write one framed message.

        Returns:
            True if it went out, False if the peer has gone away. A dead peer is
            a return value rather than an exception because on the Isaac side
            this is called from the physics loop, where the only acceptable
            response to a dead link is to carry on flying and land.
        """
        if self.closed:
            return False
        try:
            self._sock.sendall(message)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.closed = True
            return False

    def poll(self, timeout=0.0):
        # type: (float) -> list
        """Read whatever has arrived and return the complete messages in it.

        Args:
            timeout: Seconds to wait for the first byte. 0 returns immediately.

        Returns:
            A list of ``(kind, header, payload)`` tuples, possibly empty. An
            empty list from a closed peer is indistinguishable here; check
            :attr:`closed`.
        """
        if self.closed:
            return []
        messages = []
        self._sock.settimeout(timeout)
        while True:
            try:
                chunk = self._sock.recv(1 << 16)
            except socket.timeout:
                return messages
            except OSError as error:
                if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return messages
                self.closed = True
                return messages
            if not chunk:
                self.closed = True
                return messages
            messages.extend(self._decoder.feed(chunk))
            # Everything buffered has been drained; do not block again on a
            # second recv, or a quiet link costs `timeout` on every call.
            self._sock.settimeout(0.0)

    def close(self):
        # type: () -> None
        """Release the socket. Safe to call twice."""
        self.closed = True
        try:
            self._sock.close()
        except OSError:
            pass


class LinkServer:
    """Binds a port and waits for the Isaac side to connect to it.

    Args:
        port: TCP port on ``127.0.0.1``.
        name: Label used in error messages.
    """

    def __init__(self, port, name="link"):
        # type: (int, str) -> None
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(1)
        self.port = port
        self._name = name

    def accept(self, timeout=None):
        # type: (object) -> object
        """Wait for a connection.

        Args:
            timeout: Seconds to wait, or None to wait forever.

        Returns:
            A :class:`LinkEndpoint`, or None if the timeout expired.
        """
        self._listener.settimeout(timeout)
        try:
            sock, _address = self._listener.accept()
        except socket.timeout:
            return None
        return LinkEndpoint(sock, self._name)

    def close(self):
        # type: () -> None
        """Stop listening."""
        try:
            self._listener.close()
        except OSError:
            pass


def connect(port, timeout_s=120.0, retry_s=0.5, name="link"):
    # type: (int, float, float, str) -> LinkEndpoint
    """Connect to a :class:`LinkServer`, retrying until it is up.

    Args:
        port: TCP port on ``127.0.0.1``.
        timeout_s: How long to keep retrying.
        retry_s: Delay between attempts.
        name: Label used in error messages.

    Returns:
        A connected :class:`LinkEndpoint`.

    Raises:
        RuntimeError: If nothing accepted within ``timeout_s``. The message names
            the most likely cause, because a refused connection here looks
            exactly like a start-up ordering bug and is usually neither: it is
            the FALCON container not running, or running without host networking.
    """
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(("127.0.0.1", port))
            return LinkEndpoint(sock, name)
        except OSError as error:
            last = error
            sock.close()
            time.sleep(retry_s)
    raise RuntimeError(
        "no FALCON %s endpoint on 127.0.0.1:%d after %.0f s (last error: %s). "
        "The ROS1 side must be running pegasus_bridge_node, and its container "
        "must be started with --network host -- on any other network mode this "
        "loopback address is a different machine."
        % (name, port, timeout_s, last))
