"""The deadlock watchdog's wire path: FALCON's coverage reaching the aircraft.

The watchdog itself lives in the ``_explore`` loops of ``isaac/mission.py`` and
``stub/run_stub.py`` and is exercised end to end by a stub flight. What is worth
pinning in a unit test is the one link that is silent when it breaks: the
explored-volume number FALCON computes in its own container has to survive the
wire and land in ``FalconLink.coverage_m3``, or the watchdog degrades to a
movement-only check without anyone noticing.
"""
from sparx_agency.tasks.planning.falcon_pegasus.isaac.falcon_client import FalconLink
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol


def _decode_one(blob):
    """Round-trip one framed message back through the decoder."""
    messages = protocol.Decoder().feed(blob)
    assert len(messages) == 1
    return messages[0]


def test_position_command_carries_coverage():
    """Coverage rides the command header as ``cov`` when supplied."""
    kind, header, _payload = _decode_one(protocol.position_command(
        1.0, 7, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0,
        coverage_m3=12.5))
    assert kind == protocol.KIND_POSCMD
    assert header["cov"] == 12.5


def test_position_command_omits_coverage_when_none():
    """A bridge with no coverage yet sends a valid, backward-compatible command."""
    _kind, header, _payload = _decode_one(protocol.position_command(
        1.0, 7, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0))
    assert "cov" not in header


class _FakeDownlink:
    """A stand-in for the socket endpoint that yields queued messages once."""

    def __init__(self, messages):
        self._messages = messages
        self.closed = False

    def poll(self, timeout=0.0):
        drained, self._messages = self._messages, []
        return drained


def test_link_stores_forwarded_coverage():
    """``FalconLink.poll`` latches the coverage a command carried."""
    link = FalconLink()
    link._downlink = _FakeDownlink([
        (protocol.KIND_POSCMD,
         {"t": 1.0, "id": 3, "p": [0.0, 0.0, 1.0], "v": [0.0, 0.0, 0.0],
          "a": [0.0, 0.0, 0.0], "yaw": 0.0, "yaw_dot": 0.0, "cov": 42.0}, b""),
    ])
    link.poll()
    assert link.coverage_m3 == 42.0
    assert link.trajectory_id == 3


def test_link_keeps_last_coverage_when_a_command_omits_it():
    """A command without ``cov`` must not reset the last known coverage."""
    link = FalconLink()
    link.coverage_m3 = 17.0
    link._downlink = _FakeDownlink([
        (protocol.KIND_POSCMD,
         {"t": 2.0, "id": 4, "p": [0.0, 0.0, 1.0], "v": [0.0, 0.0, 0.0],
          "a": [0.0, 0.0, 0.0], "yaw": 0.0, "yaw_dot": 0.0}, b""),
    ])
    link.poll()
    assert link.coverage_m3 == 17.0

