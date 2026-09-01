"""Torch-free routing selftest for the scene-graph detection server.

Backs the server's ``--selftest`` flag: spins the real HTTP stack
(:mod:`.detection_server`'s handler + context) on an ephemeral loopback port
with a canned :class:`_StubDetector` instead of YOLO-World, then drives every
route — health, set_classes, detect, and the error paths — through real
``urllib`` requests. No model, no torch, no GPU: it runs in the plain ``.venv``
so the request plumbing can be verified on machines where the conda side cannot
even import.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server --selftest
"""
from __future__ import annotations

import json
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.tasks.mapping.scene_graph.serve.contract import encode_frame
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import (
    _TAG,
    _make_server,
    _ServerContext,
)


class _StubDetector(DetectionModel):
    """Canned detector: one fixed box per current prompt, no model, no torch."""

    def __init__(self) -> None:
        self._prompts: List[str] = []

    def set_prompts(self, prompts: Sequence[str]) -> None:
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        self._prompts = cleaned

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        return [
            Detection2D(label=p, score=0.9 - 0.1 * i,
                        bbox_xyxy=(i, i, i + 10, i + 20), frame_w=w, frame_h=h)
            for i, p in enumerate(self._prompts[:3])
        ]


def _request(port: int, method: str, path: str, body: Optional[bytes] = None,
             content_type: str = "application/octet-stream") -> Tuple[int, dict]:
    """One HTTP round-trip against loopback; returns ``(status, parsed_json)``."""
    from urllib import error, request
    req = request.Request("http://127.0.0.1:%d%s" % (port, path), data=body,
                          method=method, headers={"Content-Type": content_type})
    try:
        with request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def run_selftest() -> None:
    """Route the full request surface through a stub detector; raise on any lie."""
    detector = _StubDetector()
    detector.set_prompts(["person", "chair"])
    ctx = _ServerContext(detector, model_name="<stub>", device="none",
                         classes=["person", "chair"])
    server = _make_server(ctx, "127.0.0.1", 0)         # ephemeral port
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, health = _request(port, "GET", "/health")
        assert status == 200 and health["ok"] and health["model"] == "<stub>", health
        assert health["classes"] == ["person", "chair"], health
        assert health["frames_served"] == 0, health

        body = json.dumps({"classes": ["iv stand", "hospital bed"]}).encode()
        status, resp = _request(port, "POST", "/set_classes", body,
                                "application/json")
        assert status == 200 and resp["classes"] == ["iv stand", "hospital bed"], resp

        frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        status, det = _request(port, "POST", "/detect", encode_frame(frame))
        assert status == 200 and det["w"] == 64 and det["h"] == 48, det
        assert isinstance(det["ms"], float) and det["ms"] >= 0.0, det
        labels = [d["cls"] for d in det["detections"]]
        assert labels == ["iv stand", "hospital bed"], det
        assert all(len(d["xyxy"]) == 4 for d in det["detections"]), det

        status, health = _request(port, "GET", "/health")
        assert health["frames_served"] == 1, health
        assert health["classes"] == ["iv stand", "hospital bed"], health

        status, resp = _request(port, "POST", "/detect", b"not a jpeg")
        assert status == 400 and not resp["ok"], (status, resp)
        status, resp = _request(port, "POST", "/set_classes",
                                b'{"classes": []}', "application/json")
        assert status == 400 and not resp["ok"], (status, resp)
        status, resp = _request(port, "GET", "/nope")
        assert status == 404, (status, resp)
        status, resp = _request(port, "POST", "/nope", b"")
        assert status == 404, (status, resp)
    finally:
        server.shutdown()
        server.server_close()
    print("%s SELFTEST PASSED: health/set_classes/detect/error routing OK "
          "(stub detector, no model loaded)" % _TAG)


if __name__ == "__main__":
    run_selftest()
