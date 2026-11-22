"""How a frame's pixels become the bytes a store keeps, and back: JPEG by default, PNG when it must be lossless.

A frame is encoded once, when it is logged, and every store keeps those
bytes: a file on disk and an entry in RAM hold the same image, so swapping
the store changes nothing a reader of the log can see. Measured on one MP3D
episode's 640 x 480 frames (2026-11-22, 243 of them): JPEG at quality 95 is
111 kB a frame at the median (68 to 147) and 27 MB for the episode, quality
85 about half that, a PNG about 490 kB, the raw array 922 kB; encoding or
decoding a frame takes under 2 ms. PNG returns exactly the array it was
given, for a comparison that must not see compression.

**Channel order is the one thing to get wrong here.** The contract's frame
is RGB; OpenCV encodes and decodes BGR. Both directions swap the channels
explicitly, and the swap makes its own contiguous copy, so a frame shared
read-only by an environment is never written into and never handed to
OpenCV as a view.

``cv2`` is imported inside the two functions that need it, so importing this
module -- and the package -- stays numpy-only.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameLogError

JPEG = "jpg"
PNG = "png"
#: The image formats a codec may be built with.
FORMATS = (JPEG, PNG)
#: High enough that compression is not what a reader of the frame sees.
DEFAULT_JPEG_QUALITY = 95


@dataclass(frozen=True)
class FrameCodec:
    """One image format for a whole log.

    Attributes:
        image_format: :data:`JPEG` (the default) or :data:`PNG` (lossless).
        jpeg_quality: 1-100, read only for JPEG.

    Raises:
        FrameLogError: On a format outside :data:`FORMATS` or a quality
            outside 1-100.
    """

    image_format: str = JPEG
    jpeg_quality: int = DEFAULT_JPEG_QUALITY

    def __post_init__(self) -> None:
        if self.image_format not in FORMATS:
            raise FrameLogError("a frame is stored as one of %s, got %r" % (", ".join(FORMATS), self.image_format))
        if (not isinstance(self.jpeg_quality, int) or isinstance(self.jpeg_quality, bool)
                or not 1 <= self.jpeg_quality <= 100):
            raise FrameLogError("jpeg_quality is an integer from 1 to 100, got %r" % (self.jpeg_quality,))

    @property
    def extension(self) -> str:
        """The format's file extension, dot included."""
        return "." + self.image_format

    def encode(self, rgb: np.ndarray) -> bytes:
        """One frame as the bytes of an image file.

        Args:
            rgb: ``(H, W, 3)`` uint8 in **RGB** order. Not written into.

        Returns:
            The encoded image.

        Raises:
            FrameLogError: If ``rgb`` is not such an array, or the encoder
                refuses it.
        """
        import cv2

        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or 0 in rgb.shape):
            raise FrameLogError("a frame is a non-empty (H, W, 3) uint8 array in RGB order, got %s" % _describe(rgb))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        options = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)] if self.image_format == JPEG else []
        try:
            encoded, buffer = cv2.imencode(self.extension, bgr, options)
        except cv2.error as exc:
            raise FrameLogError("the %s encoder failed on a %s frame: %s" % (self.image_format, _describe(rgb), exc)) from exc
        if not encoded:
            raise FrameLogError("the %s encoder refused a %s frame" % (self.image_format, _describe(rgb)))
        return buffer.tobytes()

    def decode(self, image: bytes) -> np.ndarray:
        """The frame an encoded image holds.

        Args:
            image: What :meth:`encode` returned (any image file OpenCV reads).

        Returns:
            ``(H, W, 3)`` uint8 in **RGB** order, a fresh array.

        Raises:
            FrameLogError: If the bytes are not a decodable image.
        """
        import cv2

        if not isinstance(image, (bytes, bytearray, memoryview)):
            raise FrameLogError("an encoded frame is a bytes object, got %s" % type(image).__name__)
        if len(image) == 0:
            raise FrameLogError("an encoded frame of 0 bytes is not an image")
        try:
            bgr = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise FrameLogError("%d bytes that are not a decodable image: %s" % (len(image), exc)) from exc
        if bgr is None:
            raise FrameLogError("%d bytes that are not a decodable image" % len(image))
        return np.ascontiguousarray(bgr[:, :, ::-1])

    def describe(self) -> Dict[str, Any]:
        """The format as a run's record and an episode's header state it."""
        if self.image_format == JPEG:
            return {"format": JPEG, "jpeg_quality": int(self.jpeg_quality)}
        return {"format": self.image_format}


def _describe(array: Any) -> str:
    """Shape and dtype of an array, or its type when it is not one."""
    if isinstance(array, np.ndarray):
        return "shape %r dtype %s" % (array.shape, array.dtype)
    return type(array).__name__
