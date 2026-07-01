"""Parser for the drone's "frame path" sensor messages.

To avoid serializing full RGB/depth images over ROS, the drone writes each
captured frame to disk and instead publishes a tiny ``std_msgs/String`` carrying
the file path plus the capture timestamp it was synchronized to::

    /tmp/xtend_frames/frame_00000216.jpg 1780843795 329645196
    /tmp/xtend_depth/frame_00006888.npy 1780845414 842679492

The format is ``"<path> <sec> <nsec>"`` where ``sec``/``nsec`` are the ROS-style
capture stamp (seconds and nanoseconds). The timestamp is already aligned with
the localization stream upstream, so consumers only need to load the file and
reuse this stamp -- no extra synchronization.

This module is deliberately ROS-free and Python 3.8 compatible so it can be
shared by the ROS1 adapter nodes (which import ``core`` under Python 3.8).
"""

from typing import NamedTuple


class ParsedFramePath(NamedTuple):
    """A parsed frame-path message.

    Attributes:
        path: Filesystem path to the saved frame (``.jpg`` RGB or ``.npy`` depth).
        sec: Capture timestamp, whole seconds (ROS ``stamp.sec``).
        nsec: Capture timestamp, nanoseconds part (ROS ``stamp.nsec``).
    """

    path: str
    sec: int
    nsec: int

    @property
    def stamp_seconds(self) -> float:
        """The capture timestamp as floating-point seconds."""
        return self.sec + self.nsec * 1e-9


def parse_frame_path_message(data: str) -> ParsedFramePath:
    """Parse a ``"<path> <sec> <nsec>"`` frame-path message.

    The two timestamp tokens are split off from the RIGHT, so a path that itself
    contains spaces is preserved intact.

    Args:
        data: The raw string payload of the message.

    Returns:
        The parsed path and capture stamp.

    Raises:
        ValueError: If the message is empty, has fewer than three tokens, carries
            an empty path, or has non-integer ``sec``/``nsec`` tokens.
    """
    text = (data or "").strip()
    if not text:
        raise ValueError("empty frame-path message")
    parts = text.rsplit(None, 2)
    if len(parts) != 3:
        raise ValueError(
            "frame-path message must be '<path> <sec> <nsec>', got %r" % data)
    path, sec_str, nsec_str = parts
    if not path:
        raise ValueError("frame-path message has an empty path: %r" % data)
    try:
        sec, nsec = int(sec_str), int(nsec_str)
    except ValueError:
        raise ValueError(
            "frame-path stamp tokens must be integers, got %r" % data)
    return ParsedFramePath(path=path, sec=sec, nsec=nsec)
