"""The TRT server must 501 the unexported goal modes (point-goal only).

Light contract test using Flask's test client; skipped if flask/cv2/PIL are not
installed. The point-goal happy path needs real engines and is covered by the
on-target benchmark + a manual end-to-end run.
"""
from __future__ import annotations

import importlib.util

import pytest

_HAS = all(importlib.util.find_spec(m) is not None for m in ("flask", "cv2", "PIL"))
pytestmark = pytest.mark.skipif(not _HAS, reason="needs flask + cv2 + PIL")


def test_unsupported_modes_return_501():
    from sparx_agency.tasks.planning.vlas.navdp.serve import navdp_trt_server as srv
    client = srv.app.test_client()
    for route in ("/pixelgoal_step", "/imagegoal_step", "/nogoal_step"):
        resp = client.post(route)
        assert resp.status_code == 501, route


def test_pointgoal_before_reset_409():
    from sparx_agency.tasks.planning.vlas.navdp.serve import navdp_trt_server as srv
    srv._AGENT = None
    resp = srv.app.test_client().post("/pointgoal_step")
    assert resp.status_code == 409
