"""The uniform NavigationPolicy contract and the registry that serves it.

These are the pieces that decide whether adding a sixth VLA is cheap, so they
are worth pinning: the registry must stay import-free, goal-modality mismatches
must fail loudly at wire-up, and every adapter must agree on what a dropped
frame versus a malformed response means.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.vlas.common import image_codec  # noqa: F401
from sparx_agency.core.planning.vlas.flownav.errors import FlowNavClientError
from sparx_agency.core.planning.vlas.interfaces import (
    ImageGoal,
    LanguageGoal,
    NavigationPolicy,
    PointGoal,
    PolicyObservation,
    PolicyResult,
    PoseGoal,
)
from sparx_agency.core.planning.vlas.navdp.errors import NavDPError
from sparx_agency.core.planning.vlas.registry import (
    VlaFactory,
    VlaRegistry,
    default_vla_registry,
)

pytest.importorskip("PIL.Image", reason="Pillow required for the PNG codec")


def _rgb(h=8, w=6):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _obs(**kw):
    base = dict(rgb=_rgb(), depth_m=np.ones((8, 6), np.float32),
                intrinsics=Intrinsics(6, 8, 1.0, 1.0, 3.0, 4.0), altitude_m=0.8)
    base.update(kw)
    return PolicyObservation(**base)


class _FakeRequests:
    def __init__(self, payload=None, fail=False):
        self.payload = payload
        self.fail = fail

    def post(self, url, **kwargs):
        if self.fail:
            raise OSError("connection refused")
        payload = self.payload

        class _R:
            status_code = 200

            def json(self):
                return payload
        return _R()


@pytest.fixture()
def fake_requests(monkeypatch):
    import sys
    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


# ── registry ─────────────────────────────────────────────────────────────
def test_default_registry_lists_the_adapters_that_exist():
    assert default_vla_registry().names() == ["flownav", "internvla_n1", "navdp"]


def test_registry_reports_goal_modality_without_constructing():
    registry = default_vla_registry()
    assert registry.get("navdp").goal_modality == "point"
    assert registry.get("flownav").goal_modality == "image"
    assert registry.get("internvla_n1").goal_modality == "language"


def test_unknown_name_lists_what_is_available():
    with pytest.raises(KeyError, match="flownav, internvla_n1, navdp"):
        default_vla_registry().get("pi0")


def test_duplicate_registration_raises():
    # Silently overwriting would make which implementation flies depend on
    # import order.
    registry = VlaRegistry()
    registry.register(VlaFactory(name="x", create=lambda **k: None))
    with pytest.raises(ValueError):
        registry.register(VlaFactory(name="x", create=lambda **k: None))


def test_registry_construction_imports_nothing_heavy():
    # The factories must defer their imports; building the registry itself must
    # not pull requests/TensorRT into the process.
    import subprocess
    import sys
    import pathlib
    root = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]
    code = ("import sys\n"
            "from sparx_agency.core.planning.vlas.registry import default_vla_registry\n"
            "default_vla_registry().names()\n"
            "print(','.join(sorted({'requests','tensorrt','torch'} "
            "& {m.split('.')[0] for m in sys.modules})))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(root))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", "registry pulled %s" % out.stdout.strip()


# ── goal modality checking ───────────────────────────────────────────────
@pytest.mark.parametrize("policy_name,wrong_goal", [
    ("navdp", ImageGoal()),
    ("navdp", LanguageGoal(instruction="go left")),
    ("flownav", PointGoal(forward_m=1.0)),
    ("flownav", PoseGoal(forward_m=1.0)),
])
def test_wrong_goal_modality_raises_typeerror(policy_name, wrong_goal):
    policy = default_vla_registry().create(policy_name, url="http://127.0.0.1:1")
    with pytest.raises(TypeError):
        policy.step(_obs(), wrong_goal)


def test_accepted_goal_modality_is_declared():
    registry = default_vla_registry()
    assert registry.create("navdp", url="http://h").accepts == (PointGoal,)
    assert registry.create("flownav", url="http://h").accepts == (ImageGoal,)


# ── the shared "drop vs malformed" rule ──────────────────────────────────
def test_transport_drop_is_a_not_ok_result_not_an_exception(fake_requests):
    # At video rate a dropped frame is routine; the caller re-sends next frame.
    fake_requests.fail = True
    for name, goal in (("navdp", PointGoal(forward_m=1.0)), ("flownav", ImageGoal())):
        result = default_vla_registry().create(name, url="http://h").step(_obs(), goal)
        assert isinstance(result, PolicyResult)
        assert not result.ok
        assert result.metadata["transport_failed"] is True


@pytest.mark.parametrize("name,goal,error_cls", [
    ("navdp", PointGoal(forward_m=1.0), NavDPError),
    ("flownav", ImageGoal(), FlowNavClientError),
])
def test_malformed_response_raises(fake_requests, name, goal, error_cls):
    # A response that arrived but cannot be flown must raise -- silently
    # returning "no result" would let the caller keep flying a stale path.
    fake_requests.payload = {"nothing": "useful"}
    with pytest.raises(error_cls):
        default_vla_registry().create(name, url="http://h").step(_obs(), goal)


def test_good_response_becomes_an_ok_result(fake_requests):
    fake_requests.payload = {"trajectory": [[[1.0, 2.0], [3.0, 4.0]]],
                             "all_values": [[0.75]]}
    result = default_vla_registry().create("navdp", url="http://h").step(
        _obs(), PointGoal(forward_m=2.0, left_m=0.5))
    assert result.ok
    assert result.trajectory.shape == (2, 2)
    assert result.score == pytest.approx(0.75)


# ── observations a policy cannot act on ──────────────────────────────────
def test_navdp_requires_depth():
    policy = default_vla_registry().create("navdp", url="http://h")
    with pytest.raises(ValueError, match="rgb and depth_m"):
        policy.step(_obs(depth_m=None), PointGoal(forward_m=1.0))


def test_navdp_reset_requires_intrinsics():
    policy = default_vla_registry().create("navdp", url="http://h")
    with pytest.raises(ValueError, match="intrinsics"):
        policy.reset(_obs(intrinsics=None))


def test_flownav_needs_no_intrinsics(fake_requests):
    # FlowNav is RGB-only and preprocesses server-side, so an observation with
    # no camera model is fine -- this asymmetry is the point of the contract.
    fake_requests.payload = {"trajectory": [[1.0, 2.0]]}
    policy = default_vla_registry().create("flownav", url="http://h")
    assert policy.step(_obs(intrinsics=None, depth_m=None), ImageGoal()).ok


# ── the ABC itself ───────────────────────────────────────────────────────
def test_policy_cannot_be_instantiated_without_implementing_the_contract():
    class Incomplete(NavigationPolicy):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_result_ok_is_false_for_an_empty_trajectory():
    assert not PolicyResult().ok
    assert not PolicyResult(trajectory=np.zeros((0, 2), np.float32)).ok
    assert PolicyResult(trajectory=np.zeros((3, 2), np.float32)).ok
