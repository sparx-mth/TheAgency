"""Pin the HTTP wire contract of every VLA client.

These clients are the single integration seam between our stack and the policy
servers, and the FALCON XTEND flight path depends on the exact multipart field
names, the uint16 depth encoding and the goal JSON. Nothing verified that before
the clients were refactored onto the shared
:class:`~sparx_agency.core.planning.vlas.common.http_client.HttpPolicyClient`, so
these tests exist to make the contract a checked property rather than a comment.

``requests`` is monkeypatched, so no server is needed.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.vlas.common import image_codec
from sparx_agency.core.planning.vlas.common.errors import VlaError
from sparx_agency.core.planning.vlas.flownav.client import (
    FlowNavClientError,
    FlowNavImageGoalClient,
)
from sparx_agency.core.planning.vlas.navdp.client import (
    DEPTH_SCALE,
    NavDPError,
    NavDPPointgoalClient,
)

PIL = pytest.importorskip("PIL.Image", reason="Pillow required for the PNG codec")


class _Response:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _FakeRequests:
    """Records the last POST/GET so a test can assert on the wire payload."""

    def __init__(self, response=None, raises=False):
        self.response = response or _Response(payload={})
        self.raises = raises
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.raises:
            raise OSError("connection refused")
        return self.response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self.raises:
            raise OSError("connection refused")
        return self.response


@pytest.fixture()
def fake_requests(monkeypatch):
    """Install a fake ``requests`` module for the lazily-importing clients."""
    fake = _FakeRequests()
    monkeypatch.setitem(__import__("sys").modules, "requests", fake)
    return fake


def _rgb(h=8, w=6):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


# ── depth encoding ───────────────────────────────────────────────────────
def test_depth_cap_rejects_values_that_would_wrap_uint16():
    # 65535/10000 = 6.5535 m. Anything above wraps a far pixel to a near one --
    # a phantom wall exactly where the operator clicked the far floor.
    with pytest.raises(ValueError):
        NavDPPointgoalClient("http://x", depth_max_m=7.0)


def test_depth_cap_accepts_the_ceiling_exactly():
    client = NavDPPointgoalClient("http://x", depth_max_m=image_codec.MAX_DEPTH_M)
    assert client.depth_max_m == pytest.approx(image_codec.MAX_DEPTH_M)


def test_depth_png_roundtrips_metres_through_uint16():
    depth = np.array([[0.0, 1.0], [2.5, 99.0]], dtype=np.float32)
    buf = image_codec.depth_to_png(depth, depth_max_m=5.0)
    decoded = np.asarray(PIL.open(buf)).astype(np.float32) / DEPTH_SCALE
    # 99 m is clipped to the 5 m cap, everything else survives the round trip.
    assert decoded == pytest.approx(np.array([[0.0, 1.0], [2.5, 5.0]]), abs=1e-4)


def test_rgb_png_roundtrips_exactly():
    rgb = _rgb()
    assert np.array_equal(image_codec.png_to_rgb(image_codec.rgb_to_png(rgb).read()), rgb)


# ── NavDP wire contract ──────────────────────────────────────────────────
def test_navdp_reset_posts_the_intrinsic_matrix(fake_requests):
    intr = Intrinsics(width=504, height=294, fx=322.6, fy=323.4, cx=242.1, cy=90.0)
    assert NavDPPointgoalClient("http://host:8888/").reset(intr) is True
    method, url, kwargs = fake_requests.calls[-1]
    assert (method, url) == ("POST", "http://host:8888/navigator_reset")
    assert kwargs["json"]["intrinsic"] == [[322.6, 0.0, 242.1],
                                           [0.0, 323.4, 90.0],
                                           [0.0, 0.0, 1.0]]
    assert kwargs["json"]["stop_threshold"] == -999
    assert kwargs["json"]["batch_size"] == 1


def test_navdp_step_posts_the_documented_multipart_fields(fake_requests):
    fake_requests.response = _Response(payload={"trajectory": [[[1.0, 2.0]]]})
    client = NavDPPointgoalClient("http://host:8888")
    out = client.pointgoal_step(_rgb(), np.ones((8, 6), np.float32), 1.5, -0.5,
                                click_px=12, click_py=34, altitude=0.8)
    assert out == {"trajectory": [[[1.0, 2.0]]]}
    method, url, kwargs = fake_requests.calls[-1]
    assert (method, url) == ("POST", "http://host:8888/pointgoal_step")
    assert set(kwargs["files"]) == {"image", "depth"}
    assert kwargs["files"]["image"][0] == "rgb.png"
    assert kwargs["files"]["depth"][0] == "depth.png"
    goal = json.loads(kwargs["data"]["goal_data"])
    assert goal == {"goal_x": [1.5], "goal_y": [-0.5],
                    "click_px": 12, "click_py": 34, "altitude": 0.8}


def test_navdp_step_omits_altitude_when_not_given(fake_requests):
    NavDPPointgoalClient("http://h").pointgoal_step(
        _rgb(), np.ones((8, 6), np.float32), 1.0, 0.0)
    assert "altitude" not in json.loads(fake_requests.calls[-1][2]["data"]["goal_data"])


def test_navdp_transport_failure_returns_none_and_logs(fake_requests):
    # A dropped frame is normal at video rate: return None, let the caller re-send.
    fake_requests.raises = True
    seen = []
    client = NavDPPointgoalClient("http://h", logger=lambda *a: seen.append(a))
    assert client.pointgoal_step(_rgb(), np.ones((8, 6), np.float32), 1.0, 0.0) is None
    assert client.reset(Intrinsics(1, 1, 1.0, 1.0, 0.0, 0.0)) is False
    assert seen, "transport failures must be logged, not swallowed silently"


def test_navdp_non_200_returns_none(fake_requests):
    fake_requests.response = _Response(status_code=503)
    assert NavDPPointgoalClient("http://h").pointgoal_step(
        _rgb(), np.ones((8, 6), np.float32), 1.0, 0.0) is None


# ── FlowNav wire contract ────────────────────────────────────────────────
def test_flownav_step_sends_only_the_observation_without_a_goal(fake_requests):
    fake_requests.response = _Response(payload={"trajectory": [[1.0, 2.0]]})
    out = FlowNavImageGoalClient("http://host:8889").step(_rgb())
    assert out == {"trajectory": [[1.0, 2.0]]}
    method, url, kwargs = fake_requests.calls[-1]
    assert (method, url) == ("POST", "http://host:8889/imagegoal_step")
    assert set(kwargs["files"]) == {"image"}


def test_flownav_step_adds_the_goal_image_when_given(fake_requests):
    FlowNavImageGoalClient("http://h").step(_rgb(), goal_rgb=_rgb())
    assert set(fake_requests.calls[-1][2]["files"]) == {"image", "goal_image"}


def test_flownav_get_goal_decodes_the_returned_png(fake_requests):
    rgb = _rgb()
    fake_requests.response = _Response(content=image_codec.rgb_to_png(rgb).read())
    assert np.array_equal(FlowNavImageGoalClient("http://h").get_goal(), rgb)


def test_flownav_get_goal_returns_none_on_failure(fake_requests):
    fake_requests.raises = True
    assert FlowNavImageGoalClient("http://h").get_goal() is None


# ── trajectory decoding: malformed content DOES raise ────────────────────
@pytest.mark.parametrize("client_cls,error_cls", [
    (NavDPPointgoalClient, NavDPError),
    (FlowNavImageGoalClient, FlowNavClientError),
])
@pytest.mark.parametrize("payload", [
    None,
    {},
    {"trajectory": []},
    {"trajectory": [[1.0]]},          # fewer than 2 columns
])
def test_malformed_trajectory_raises_not_returns(client_cls, error_cls, payload):
    # Unlike a transport drop, a response that arrived but cannot be flown must
    # raise -- returning None would let the caller keep flying a stale path.
    with pytest.raises(error_cls):
        client_cls.best_trajectory(payload)


def test_batched_trajectory_is_unwrapped_to_the_first_item():
    traj = NavDPPointgoalClient.best_trajectory({"trajectory": [[[1.0, 2.0], [3.0, 4.0]]]})
    assert traj.shape == (2, 2)
    assert traj.dtype == np.float32
    assert traj[1] == pytest.approx([3.0, 4.0])


def test_unbatched_trajectory_is_accepted_as_is():
    # FlowNav's server already picks sample 0, so it returns (T, C).
    assert FlowNavImageGoalClient.best_trajectory({"trajectory": [[1.0, 2.0]]}).shape == (1, 2)


def test_every_client_error_is_catchable_as_the_shared_base():
    for error_cls in (NavDPError, FlowNavClientError):
        assert issubclass(error_cls, VlaError)
        assert issubclass(error_cls, RuntimeError)  # unchanged from before the split
