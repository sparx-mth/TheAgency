"""Tests for the TCP endpoints, over real loopback sockets.

Not mocked. The properties worth testing here are the ones a mock would define
away: that a large payload survives being split across recv boundaries, that a
peer going away is a return value rather than an exception, and that a connect
to nothing raises something a human can act on.
"""
import threading

import numpy as np
import pytest

from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.socket_link import (
    LinkServer, connect,
)

PORT = 47599   # not the real link's port, so a test cannot disturb a live run


@pytest.fixture
def link():
    """A connected server/client pair on loopback."""
    server = LinkServer(PORT, "test")
    accepted = {}

    def accept():
        accepted["endpoint"] = server.accept(timeout=10.0)

    thread = threading.Thread(target=accept)
    thread.start()
    client = connect(PORT, timeout_s=10.0, name="test")
    thread.join()
    yield accepted["endpoint"], client
    client.close()
    if accepted["endpoint"] is not None:
        accepted["endpoint"].close()
    server.close()


def _drain(endpoint, expected, timeout=5.0):
    """Poll until ``expected`` messages have arrived."""
    messages = []
    while len(messages) < expected:
        got = endpoint.poll(timeout=timeout)
        if not got and endpoint.closed:
            break
        messages.extend(got)
    return messages


def test_a_message_arrives_intact(link):
    server_end, client = link
    assert client.send(protocol.event(protocol.EVENT_MISSION_OVER, "bye"))
    (kind, header, _payload), = _drain(server_end, 1)
    assert kind == protocol.KIND_EVENT
    assert header["detail"] == "bye"


def test_a_depth_sized_payload_survives_the_recv_boundary(link):
    """640x480 uint16 is 614 kB, far more than one recv returns."""
    server_end, client = link
    depth = np.arange(640 * 480, dtype="<u2").reshape(480, 640)
    client.send(protocol.frame(1.0, 640, 480, (1, 2, 3), (0, 0, 0, 1), depth.tobytes()))
    (kind, header, payload), = _drain(server_end, 1)
    assert kind == protocol.KIND_FRAME
    assert header["h"] == 480
    np.testing.assert_array_equal(
        np.frombuffer(payload, "<u2").reshape(480, 640), depth)


def test_many_messages_keep_their_order(link):
    server_end, client = link
    for index in range(50):
        client.send(protocol.encode(protocol.KIND_ODOM, {"t": float(index)}))
    messages = _drain(server_end, 50)
    assert [header["t"] for _k, header, _p in messages] == [float(i) for i in range(50)]


def test_polling_a_quiet_link_returns_nothing_and_does_not_block(link):
    server_end, _client = link
    assert server_end.poll(timeout=0.0) == []
    assert not server_end.closed


def test_a_closed_peer_is_reported_not_raised(link):
    server_end, client = link
    client.close()
    for _ in range(20):
        server_end.poll(timeout=0.1)
        if server_end.closed:
            break
    assert server_end.closed
    # And sending into the void is False, not an exception -- the Isaac side
    # calls this from the physics loop and must not be interrupted by a link.
    assert server_end.send(protocol.encode(protocol.KIND_ODOM, {})) is False


def test_connecting_to_nothing_explains_itself():
    with pytest.raises(RuntimeError) as excinfo:
        connect(PORT + 1, timeout_s=1.0, retry_s=0.1, name="test")
    message = str(excinfo.value)
    assert "network host" in message   # the actual usual cause, named
    assert "127.0.0.1:%d" % (PORT + 1) in message
