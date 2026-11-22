"""The codec: a frame survives being stored -- its channels in order, its pixels exact (PNG) or close (JPEG)."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.codec import DEFAULT_JPEG_QUALITY, JPEG, PNG, FrameCodec
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameLogError
from sparx_agency.core.planning.object_search.frame_log.tests.rig import pattern


def smooth_frame(height=96, width=128):
    """A frame without hard edges -- what a camera sees -- with a different ramp in every channel."""
    x = np.linspace(0.0, 1.0, width)[None, :]
    y = np.linspace(0.0, 1.0, height)[:, None]
    rgb = np.stack([220.0 * x + 0.0 * y, 30.0 + 150.0 * y + 0.0 * x, 255.0 * (1.0 - x) * (1.0 - y)], axis=2)
    return rgb.astype(np.uint8)


def test_a_png_frame_comes_back_exactly_and_a_jpeg_frame_within_compression():
    rgb = smooth_frame()
    lossless = FrameCodec(PNG).decode(FrameCodec(PNG).encode(rgb))
    assert lossless.dtype == np.uint8 and np.array_equal(lossless, rgb)
    lossy = FrameCodec().decode(FrameCodec().encode(rgb))
    assert lossy.shape == rgb.shape and lossy.dtype == np.uint8
    error = np.abs(lossy.astype(int) - rgb.astype(int))
    assert error.mean() < 1.5 and error.max() <= 12, (error.mean(), error.max())


@pytest.mark.parametrize("codec", [FrameCodec(), FrameCodec(PNG)])
def test_red_stays_red_through_the_store_and_back(codec):
    """The contract's frame is RGB and OpenCV's is BGR: a swap in either direction turns this frame blue."""
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 230, 120, 10
    back = codec.decode(codec.encode(rgb)).reshape(-1, 3).mean(axis=0)
    assert back == pytest.approx([230, 120, 10], abs=3)


def test_a_frame_shared_read_only_is_encoded_without_being_written_into():
    """The fake environment hands every policy the same read-only array; an adapter may reuse a buffer."""
    rgb = pattern(4)
    rgb.setflags(write=False)
    before = rgb.copy()
    codec = FrameCodec(PNG)
    assert np.array_equal(codec.decode(codec.encode(rgb)), before)
    assert np.array_equal(rgb, before) and not rgb.flags.writeable
    view = np.ascontiguousarray(before)[::-1, ::-1]                    # not contiguous: a flipped view
    assert np.array_equal(codec.decode(codec.encode(view)), view)


def test_the_decoded_frame_is_the_callers_own_array():
    codec = FrameCodec(PNG)
    image = codec.encode(pattern(1))
    first, second = codec.decode(image), codec.decode(image)
    first[...] = 0
    assert np.array_equal(second, pattern(1)) and second.flags.writeable and second.flags.c_contiguous


def test_the_same_frame_encodes_to_the_same_bytes():
    """A run that records its frames and one that keeps them in RAM hold byte-identical images."""
    for codec in (FrameCodec(), FrameCodec(PNG)):
        assert codec.encode(pattern(7)) == codec.encode(pattern(7).copy())
        assert codec.encode(pattern(7)) != codec.encode(pattern(8))


@pytest.mark.parametrize("not_a_frame", [
    np.zeros((32, 48), np.uint8),                 # no channels
    np.zeros((32, 48, 4), np.uint8),              # an alpha channel the adapter forgot to drop
    np.zeros((32, 48, 3), np.float32),            # not bytes
    np.zeros((0, 48, 3), np.uint8),               # empty
    [[[0, 0, 0]]],                                # not an array
    None,
])
def test_what_is_not_an_rgb_frame_is_refused_by_name(not_a_frame):
    with pytest.raises(FrameLogError, match=r"\(H, W, 3\) uint8 array in RGB order"):
        FrameCodec().encode(not_a_frame)


def test_bytes_that_are_not_an_image_are_refused():
    codec = FrameCodec()
    with pytest.raises(FrameLogError, match="not a decodable image"):
        codec.decode(b"this is not a jpeg")
    with pytest.raises(FrameLogError, match="0 bytes"):
        codec.decode(b"")
    with pytest.raises(FrameLogError, match="bytes object"):
        codec.decode("step_000001.jpg")
    whole = codec.encode(pattern(2))
    assert codec.decode(bytearray(whole)).shape == pattern(2).shape, "any bytes-like object is read"


def test_a_codec_is_one_of_two_formats_with_a_quality_from_1_to_100():
    assert FrameCodec().image_format == JPEG and FrameCodec().jpeg_quality == DEFAULT_JPEG_QUALITY == 95
    assert FrameCodec().extension == ".jpg" and FrameCodec(PNG).extension == ".png"
    assert FrameCodec().describe() == {"format": "jpg", "jpeg_quality": 95}
    assert FrameCodec(PNG).describe() == {"format": "png"}, "a PNG has no quality to state"
    assert FrameCodec(JPEG, 60).describe() == {"format": "jpg", "jpeg_quality": 60}
    with pytest.raises(FrameLogError, match="one of jpg, png"):
        FrameCodec("webp")
    for quality in (0, 101, 95.0, True, None):
        with pytest.raises(FrameLogError, match="1 to 100"):
            FrameCodec(JPEG, quality)


def test_a_lower_quality_makes_a_smaller_file():
    rgb = smooth_frame()
    assert len(FrameCodec(JPEG, 40).encode(rgb)) < len(FrameCodec(JPEG, 95).encode(rgb))
