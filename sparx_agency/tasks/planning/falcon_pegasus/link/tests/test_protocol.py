"""Tests for the Isaac <-> FALCON wire format.

The properties that matter are the ones a debugger cannot see: that a message
survives being chopped into arbitrary TCP-sized pieces, that a desynchronised
stream is caught rather than interpreted, and that the depth encoding makes the
distinction between "nothing is there" and "no reading" that the map depends on.
"""
import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.depth_codec import (
    FAR_M, NEAR_M, decode_depth, encode_depth,
)


def _drain(decoder, blob, chunk_size):
    """Feed ``blob`` through ``decoder`` in fixed-size pieces."""
    messages = []
    for start in range(0, len(blob), chunk_size):
        messages.extend(decoder.feed(blob[start:start + chunk_size]))
    return messages


def test_round_trips_a_header_only_message():
    blob = protocol.encode(protocol.KIND_ODOM, {"t": 1.5, "p": [1.0, 2.0, 3.0]})
    (kind, header, payload), = protocol.Decoder().feed(blob)
    assert kind == protocol.KIND_ODOM
    assert header["p"] == [1.0, 2.0, 3.0]
    assert payload == b""


def test_round_trips_a_payload_message():
    depth = np.arange(6, dtype="<u2").reshape(2, 3)
    blob = protocol.frame(1.25, 3, 2, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), depth.tobytes())
    (kind, header, payload), = protocol.Decoder().feed(blob)
    assert kind == protocol.KIND_FRAME
    assert header["t"] == 1.25 and header["w"] == 3 and header["h"] == 2
    np.testing.assert_array_equal(decode_depth(payload, 3, 2), depth)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 15, 16, 17, 1000])
def test_survives_arbitrary_tcp_fragmentation(chunk_size):
    """TCP has no message boundaries; the decoder must supply them."""
    depth = np.full((4, 5), 1234, dtype="<u2")
    blob = (protocol.encode(protocol.KIND_ODOM, {"t": 0.0})
            + protocol.frame(1.0, 5, 4, (0, 0, 0), (0, 0, 0, 1), depth.tobytes())
            + protocol.event(protocol.EVENT_EXPLORATION_FINISHED, "done"))
    kinds = [kind for kind, _h, _p in _drain(protocol.Decoder(), blob, chunk_size)]
    assert kinds == [protocol.KIND_ODOM, protocol.KIND_FRAME, protocol.KIND_EVENT]


def test_partial_message_yields_nothing_then_everything():
    blob = protocol.encode(protocol.KIND_ODOM, {"t": 3.0})
    decoder = protocol.Decoder()
    assert decoder.feed(blob[:-1]) == []
    (kind, header, _payload), = decoder.feed(blob[-1:])
    assert kind == protocol.KIND_ODOM and header["t"] == 3.0


def test_a_desynchronised_stream_raises_rather_than_being_interpreted():
    decoder = protocol.Decoder()
    with pytest.raises(protocol.ProtocolError):
        decoder.feed(b"not a falcon message at all, really not")


def test_a_future_protocol_version_is_rejected():
    blob = bytearray(protocol.encode(protocol.KIND_ODOM, {}))
    blob[4] = protocol.VERSION + 1
    with pytest.raises(protocol.ProtocolError):
        protocol.Decoder().feed(bytes(blob))


def test_unknown_kind_is_rejected_at_encode():
    with pytest.raises(protocol.ProtocolError):
        protocol.encode(99, {})


def test_unknown_event_is_rejected_at_encode():
    with pytest.raises(protocol.ProtocolError):
        protocol.event("something_nobody_handles")


def test_hello_carries_every_intrinsic_the_mapper_back_projects_with():
    blob = protocol.hello(Intrinsics(width=640, height=480, fx=320.0, fy=321.0,
                                     cx=322.0, cy=240.0), "office", "r1")
    (_kind, header, _payload), = protocol.Decoder().feed(blob)
    assert (header["width"], header["height"]) == (640, 480)
    assert (header["fx"], header["fy"], header["cx"], header["cy"]) == (320.0, 321.0, 322.0, 240.0)
    assert header["depth_units"] == "mm"


def test_position_command_carries_the_whole_reference():
    blob = protocol.position_command(2.0, 7, (1, 2, 3), (0.1, 0.2, 0.3),
                                     (0.01, 0.02, 0.03), 1.57, -0.5)
    (_kind, header, _payload), = protocol.Decoder().feed(blob)
    assert header["id"] == 7
    assert header["p"] == [1.0, 2.0, 3.0]
    assert header["a"] == [0.01, 0.02, 0.03]
    assert header["yaw"] == 1.57 and header["yaw_dot"] == -0.5


# ── depth codec ─────────────────────────────────────────────────────────────

def test_metres_become_millimetres():
    encoded = encode_depth(np.array([[1.0, 2.5, 0.5]], dtype=np.float32))
    np.testing.assert_array_equal(encoded, np.array([[1000, 2500, 500]], dtype="<u2"))


def test_infinity_is_a_measurement_and_nan_is_not():
    """The single most consequential line in this codec.

    ``inf`` means the ray left through a window -- that space is free and must be
    carved. ``NaN`` means the sensor said nothing, and inventing free space there
    would delete real obstacles from the map.
    """
    encoded = encode_depth(np.array([[math.inf, math.nan]], dtype=np.float32))
    assert encoded[0, 0] == int(round(FAR_M * 1000))
    assert encoded[0, 1] == 0


def test_negative_infinity_is_discarded():
    encoded = encode_depth(np.array([[-math.inf]], dtype=np.float32))
    assert encoded[0, 0] == 0


def test_readings_closer_than_the_minimum_range_are_discarded():
    encoded = encode_depth(np.array([[NEAR_M - 0.01, NEAR_M + 0.01]], dtype=np.float32))
    assert encoded[0, 0] == 0
    assert encoded[0, 1] > 0


def test_far_readings_are_clamped_not_wrapped():
    encoded = encode_depth(np.array([[500.0]], dtype=np.float32))
    assert encoded[0, 0] == int(round(FAR_M * 1000))


def test_the_far_clamp_stays_above_falcons_raycast_limit():
    """Clamping at or below raycast_max would build a wall across every doorway."""
    falcon_raycast_max_m = 5.0
    assert FAR_M > falcon_raycast_max_m * 2


def test_encoded_depth_dtype_is_what_the_mapper_reads():
    encoded = encode_depth(np.zeros((2, 2), dtype=np.float32))
    assert encoded.dtype.str == protocol.DEPTH_DTYPE


@pytest.mark.parametrize("far_m,near_m", [(0.1, 0.2), (100.0, 0.1)])
def test_invalid_range_configuration_is_rejected(far_m, near_m):
    with pytest.raises(ValueError):
        encode_depth(np.zeros((1, 1), dtype=np.float32), near_m=near_m, far_m=far_m)


def test_a_wrong_sized_payload_is_rejected_rather_than_resnaped():
    with pytest.raises(ValueError):
        decode_depth(b"\x00" * 10, width=4, height=4)


def test_encode_decode_round_trip_preserves_every_pixel():
    rng = np.random.default_rng(0)
    metres = rng.uniform(0.2, 15.0, size=(392, 504)).astype(np.float32)
    encoded = encode_depth(metres)
    decoded = decode_depth(encoded.tobytes(), 504, 392)
    np.testing.assert_array_equal(decoded, encoded)
    assert np.max(np.abs(decoded.astype(float) / 1000.0 - metres)) < 1e-3


def test_round_trips_a_whole_trajectory():
    """The curve survives the wire, and rebuilds into the same polynomial.

    Checked by evaluating the rebuilt curve rather than by comparing fields: the
    thing that must hold is that both sides of the link agree on where the
    aircraft should be, and a knot vector that round-trips but is applied
    differently would pass a field comparison and fail in flight.
    """
    from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory

    points = [[float(i) * 0.5, float(i % 3), 1.4] for i in range(9)]
    knots = [(-3 + i) * 0.35 for i in range(len(points) + 4)]
    yaws = [0.1, 0.4, 0.9, 1.3, 1.6, 1.9]
    blob = protocol.bspline(1234.5, 7, 3, knots, points, yaws, 0.5)

    (kind, header, payload), = protocol.Decoder().feed(blob)
    assert kind == protocol.KIND_BSPLINE
    assert payload == b""
    assert header["id"] == 7
    assert header["t"] == 1234.5

    rebuilt = BsplineTrajectory.from_falcon(
        header["order"], header["knots"], header["pos"], header["yaw"],
        header["yaw_dt"], header["t"], header["id"])
    original = BsplineTrajectory.from_falcon(3, knots, points, yaws, 0.5, 1234.5, 7)
    assert rebuilt.duration == pytest.approx(original.duration)
    for t in (0.0, 0.4, 1.1, rebuilt.duration):
        here, there = rebuilt.sample(t), original.sample(t)
        assert (here.x, here.y, here.z) == pytest.approx((there.x, there.y, there.z))
        assert here.yaw == pytest.approx(there.yaw)


def test_a_trajectory_message_survives_being_chopped_up():
    """It is the largest header on the link, so it is the one TCP will split."""
    points = [[float(i), 0.0, 1.0] for i in range(40)]
    knots = [(-3 + i) * 0.2 for i in range(len(points) + 4)]
    blob = protocol.bspline(0.0, 1, 3, knots, points, [0.0] * 6, 0.2)
    assert len(blob) > 1024
    messages = _drain(protocol.Decoder(), blob, chunk_size=97)
    assert len(messages) == 1
    assert messages[0][0] == protocol.KIND_BSPLINE
    assert len(messages[0][1]["pos"]) == 40


def test_the_bspline_kind_has_a_name():
    """Unnamed kinds show up in logs as integers and waste an afternoon."""
    assert protocol.KIND_NAMES[protocol.KIND_BSPLINE] == "bspline"
