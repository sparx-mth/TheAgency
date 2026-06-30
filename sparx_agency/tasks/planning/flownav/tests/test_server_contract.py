"""Contract tests for the FlowNav host server routes (no GPU; policy mocked)."""
import io
from collections import deque

import numpy as np
import pytest

from sparx_agency.tasks.planning.flownav.server import flownav_trt_server as srv


def _png_bytes(value=128):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.full((40, 40, 3), value, np.uint8), "RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


class _FakePolicy:
    precision, num_samples, num_steps = "fp16", 8, 4

    def predict(self, obs_img, goal_img):
        assert obs_img.shape == (1, 12, 96, 96)
        assert goal_img.shape == (1, 3, 96, 96)
        return np.arange(8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2), 1.23


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(srv, "_POLICY", None)
    monkeypatch.setattr(srv, "_FRAMES", deque(maxlen=4))
    monkeypatch.setattr(srv, "_CFG", {"context_size": 3, "image_size": 96})
    return srv.app.test_client()


def test_step_before_init_returns_503(client):
    r = client.post("/imagegoal_step",
                    data={"image": (_png_bytes(), "rgb.png"),
                          "goal_image": (_png_bytes(), "goal.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503


def test_step_missing_files_returns_400(client, monkeypatch):
    monkeypatch.setattr(srv, "_POLICY", _FakePolicy())
    r = client.post("/imagegoal_step", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_step_happy_path_returns_trajectory(client, monkeypatch):
    monkeypatch.setattr(srv, "_POLICY", _FakePolicy())
    r = client.post("/imagegoal_step",
                    data={"image": (_png_bytes(100), "rgb.png"),
                          "goal_image": (_png_bytes(200), "goal.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    traj = np.asarray(body["trajectory"])
    assert traj.shape == (8, 2)                       # chosen sample 0, 8 waypoints
    assert np.asarray(body["all_trajectory"]).shape == (8, 8, 2)
    assert body["distance"] == pytest.approx(1.23)


def test_reset_clears_buffer(client, monkeypatch):
    monkeypatch.setattr(srv, "_POLICY", _FakePolicy())
    srv._FRAMES.append(np.zeros((40, 40, 3), np.uint8))
    assert len(srv._FRAMES) == 1
    assert client.post("/reset").status_code == 200
    assert len(srv._FRAMES) == 0


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["algo"] == "flownav-trt"
