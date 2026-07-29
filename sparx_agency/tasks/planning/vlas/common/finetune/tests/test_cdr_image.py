"""Round-trip tests for the host-side CDR reader (numpy, no ROS).

An independent little-endian CDR encoder (mirroring the FastCDR alignment rules)
builds Image / CameraInfo byte strings, and the reader must recover them. This
locks in the one subtle rule: 8-byte doubles align relative to the payload start
(offset 4), not the buffer start.
"""
import struct

import numpy as np

from sparx_agency.tasks.planning.vlas.common.finetune.datasets.cdr_image import (
    read_camera_info,
    read_image,
    read_stamp,
    to_ndarray,
)


class _Writer:
    """Minimal CDR_LE encoder used only to fabricate test messages."""

    def __init__(self) -> None:
        self.b = bytearray(b"\x00\x01\x00\x00")  # CDR_LE encapsulation header

    def _align(self, n: int) -> None:
        while (len(self.b) - 4) % n:
            self.b.append(0)

    def u8(self, v: int) -> None:
        self.b.append(v & 0xFF)

    def u32(self, v: int) -> None:
        self._align(4)
        self.b += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.b += struct.pack("<i", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.b += struct.pack("<d", v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def seq_bytes(self, data: bytes) -> None:
        self.u32(len(data))
        self.b += data


def test_image_roundtrip_bgr8():
    h, w = 2, 3
    pixels = np.arange(h * w * 3, dtype=np.uint8)
    wr = _Writer()
    wr.i32(1780231575)
    wr.u32(364093871)
    wr.string("xtend_camera")
    wr.u32(h)
    wr.u32(w)
    wr.string("bgr8")
    wr.u8(0)
    wr.u32(w * 3)               # step
    wr.seq_bytes(pixels.tobytes())
    msg = read_image(bytes(wr.b))

    assert msg.stamp_ns == 1780231575 * 1_000_000_000 + 364093871
    assert read_stamp(bytes(wr.b)) == msg.stamp_ns
    assert (msg.height, msg.width, msg.encoding) == (h, w, "bgr8")
    arr = to_ndarray(msg)
    assert arr.shape == (h, w, 3)
    np.testing.assert_array_equal(arr.reshape(-1), pixels)


def test_image_roundtrip_16uc1():
    h, w = 2, 4
    depth_mm = np.array([[700, 800, 900, 1000], [1100, 1200, 1300, 1400]], np.uint16)
    wr = _Writer()
    wr.i32(5)
    wr.u32(0)
    wr.string("cam")
    wr.u32(h)
    wr.u32(w)
    wr.string("16UC1")
    wr.u8(0)
    wr.u32(w * 2)
    wr.seq_bytes(depth_mm.tobytes())
    arr = to_ndarray(read_image(bytes(wr.b)))
    assert arr.dtype == np.uint16 and arr.shape == (h, w)
    np.testing.assert_array_equal(arr, depth_mm)


def test_camera_info_double_alignment():
    # 5 distortion coeffs -> odd count, so K starts at an 8-misaligned offset:
    # exercises the payload-relative 8-byte alignment rule.
    fx, fy, cx, cy = 322.635, 323.389, 242.065, 90.030
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    D = [-0.2972, 0.0801, -0.0037, -0.0006, 0.0]
    wr = _Writer()
    wr.i32(0)
    wr.u32(0)
    wr.string("xtend_camera")
    wr.u32(294)
    wr.u32(504)
    wr.string("plumb_bob")
    wr.u32(len(D))
    for x in D:
        wr.f64(x)
    for x in K:
        wr.f64(x)
    info = read_camera_info(bytes(wr.b))
    assert (info["width"], info["height"]) == (504, 294)
    assert abs(info["fx"] - fx) < 1e-9 and abs(info["fy"] - fy) < 1e-9
    assert abs(info["cx"] - cx) < 1e-9 and abs(info["cy"] - cy) < 1e-9
