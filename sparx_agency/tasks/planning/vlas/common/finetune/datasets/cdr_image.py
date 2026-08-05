"""Minimal CDR deserializer for the two ROS2 message types a flight-recording
exporter needs: ``sensor_msgs/msg/Image`` and ``sensor_msgs/msg/CameraInfo``.

rosbag2 stores every message as CDR — the FastCDR encapsulation used by
``rmw_fastrtps``: a 4-byte encapsulation header followed by the payload, with
each primitive aligned to its own size *relative to the start of the payload*.
This reader implements exactly that, enough to pull image pixels and camera
intrinsics straight out of a ``.db3`` on the host, with **no ROS install** and
**identical results for Foxy- and Humble-written bags** (the message definitions
and CDR wire format are stable across those distros).

It is deliberately tiny (no third-party deps beyond numpy) so a rosbag exporter
can decode a bag natively instead of replaying it through ``ros2 bag play``,
which drops frames under best-effort QoS.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# Native numpy dtype + channel count for each supported ROS image encoding.
_ENCODINGS = {
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


class _Cdr:
    """A cursor over a CDR-encapsulated message body."""

    __slots__ = ("buf", "pos", "e")

    def __init__(self, buf: bytes) -> None:
        if buf[0] != 0x00:
            raise ValueError("unexpected CDR encapsulation identifier")
        self.buf = buf
        self.e = "<" if buf[1] == 0x01 else ">"  # 0x01 == little-endian (CDR_LE)
        self.pos = 4  # payload starts right after the 4-byte encapsulation header

    def _align(self, n: int) -> None:
        # FastCDR aligns each primitive to its size, measured from the payload
        # start (offset 4) — not the buffer start. Matters for 8-byte doubles.
        rem = (self.pos - 4) % n
        if rem:
            self.pos += n - rem

    def u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.e + "I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from(self.e + "i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def f64(self) -> float:
        self._align(8)
        v = struct.unpack_from(self.e + "d", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def string(self) -> str:
        n = self.u32()
        raw = self.buf[self.pos:self.pos + n]
        self.pos += n
        return raw.rstrip(b"\x00").decode("utf-8", "replace")


@dataclass
class ImageMsg:
    """A deserialized ``sensor_msgs/msg/Image`` (pixels kept as a raw uint8 view)."""

    stamp_ns: int
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: np.ndarray  # 1-D uint8 view into the message buffer


def read_stamp(buf: bytes) -> int:
    """Return the ``header.stamp`` in integer nanoseconds.

    Both Image and CameraInfo begin with ``std_msgs/Header``, whose first field
    is ``builtin_interfaces/Time {int32 sec, uint32 nanosec}``, so a 12-byte
    prefix of the message is enough — the caller can read just that.
    """
    c = _Cdr(buf)
    return c.i32() * 1_000_000_000 + c.u32()


def read_image(buf: bytes) -> ImageMsg:
    """Deserialize a ``sensor_msgs/msg/Image``; validates ``step*height == len``."""
    c = _Cdr(buf)
    sec, nsec = c.i32(), c.u32()
    c.string()  # frame_id (unused)
    height, width = c.u32(), c.u32()
    encoding = c.string()
    is_be = c.u8()
    step = c.u32()
    n = c.u32()
    if step * height != n:
        raise ValueError(f"image payload {n} != step*height {step * height}")
    data = np.frombuffer(buf, dtype=np.uint8, count=n, offset=c.pos)
    return ImageMsg(sec * 1_000_000_000 + nsec, height, width, encoding, is_be, step, data)


def to_ndarray(msg: ImageMsg) -> np.ndarray:
    """Reshape an :class:`ImageMsg` into a native-dtype ``(H, W[, C])`` array.

    Applies the declared endianness and channel count. No unit or colour-order
    convention is imposed here — that is the exporter's responsibility.
    """
    if msg.encoding not in _ENCODINGS:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")
    base, channels = _ENCODINGS[msg.encoding]
    dt = np.dtype(base).newbyteorder(">" if msg.is_bigendian else "<")
    arr = msg.data.view(dt)
    if channels > 1:
        arr = arr.reshape(msg.height, msg.width, channels)
    else:
        arr = arr.reshape(msg.height, msg.width)
    return np.ascontiguousarray(arr)


def read_camera_info(buf: bytes) -> dict:
    """Deserialize the intrinsics of a ``sensor_msgs/msg/CameraInfo``."""
    c = _Cdr(buf)
    c.i32(), c.u32()  # header.stamp
    frame_id = c.string()
    height, width = c.u32(), c.u32()
    model = c.string()
    d = [c.f64() for _ in range(c.u32())]  # D: sequence<double>
    k = [c.f64() for _ in range(9)]        # K: double[9] row-major
    return {
        "width": width, "height": height, "frame_id": frame_id,
        "distortion_model": model,
        "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5], "D": d,
    }
