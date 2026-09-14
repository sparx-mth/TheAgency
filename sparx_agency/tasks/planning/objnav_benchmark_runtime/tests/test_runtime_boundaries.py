"""Dataset-free import, embodiment, RGB-D and service-identity contracts."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator, habitat_pose, metric_depth
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import VerifiedLLMClient, public_service_url
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.fixtures import camera, actions


def test_runtime_has_no_dataset_specific_imports():
    root = Path(__file__).resolve().parents[1]
    shared = list(root.glob("*.py"))
    shared += list((root / "methods").rglob("*.py")) + list((root / "habitat").rglob("*.py"))
    for path in shared:
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any(".gibson" in n or ".datasets." in n for n in names), path


def test_imports_do_not_load_simulators_models_or_native_ompl():
    root = next(p for p in Path(__file__).resolve().parents if (p / "sparx_agency").is_dir())
    code = '''
import importlib.abc, sys
attempts = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"ompl", "habitat_sim", "torch", "ai2thor"}:
            attempts.append(fullname)
            raise ImportError("Forbidden optional import")
sys.meta_path.insert(0, Guard())
from sparx_agency.core.planning.planners import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime import evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat import simulator, smoke
assert not attempts, attempts
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_robot_geometry_is_required_and_not_derived_from_camera_height():
    with pytest.raises(TypeError):
        RPTSettings()
    with pytest.raises(TypeError):
        HabitatRGBDSimulator(camera(), actions(), 0.2, False)
    bridge = HabitatRGBDSimulator(camera(), actions(), 0.2, False, height_m=0.7)
    assert bridge.height_m == 0.7 and bridge.camera.height_m == 1.0
    with pytest.raises(ValueError):
        HabitatRGBDSimulator(camera(), actions(), 0.2, False, height_m=float("nan"))


def test_habitat_pose_and_depth_conversion_use_shared_units():
    pose = habitat_pose((1, 2, 3), np.eye(3), np.eye(3))
    assert (pose.x, pose.y, pose.z, pose.yaw) == (-3, -1, 2, 0)
    left = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert habitat_pose((0, 0, 0), left, left).yaw == pytest.approx(np.pi / 2)
    values = metric_depth(np.array([[0, 0.1, 1, 5]], np.float32), camera())
    assert np.isnan(values[0, :2]).all() and values[0, 2] == 1 and np.isinf(values[0, 3])


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_detector_pins_atomic_vocabulary_and_checkpoint_identity():
    detector = HttpDetector("http://localhost:8092", ("chair",))
    metadata = {"checkpoint_sha256": "test", "detector_config": {"conf_thresh": 0.25}}
    health = {"ok": True, "model": "test", "device": "cpu", "classes": ["chair"], "metadata": metadata}
    reply = {"h": 8, "w": 8, "classes": ["chair"], "metadata": metadata, "detections": []}
    detector.session = SimpleNamespace(get=lambda *a, **k: Response(health),
                                       post=lambda *a, **k: Response(reply))
    assert detector.detect(np.zeros((8, 8, 3), np.uint8)) == []
    reply["classes"] = ["bed"]
    with pytest.raises(ObjNavInternalError, match="reconfigured"):
        detector.detect(np.zeros((8, 8, 3), np.uint8))
    health["metadata"] = dict(metadata, checkpoint_sha256="changed")
    with pytest.raises(ObjNavInternalError, match="changed"):
        detector.health()


def test_llm_readiness_is_requested_model_not_just_live_port():
    cfg = SimpleNamespace(backend="ollama", base_url="http://localhost:11434", timeout_s=3, model="test-model")
    payload = {"models": []}
    raw = SimpleNamespace(cfg=cfg, sess=SimpleNamespace(get=lambda *a, **k: Response(payload)),
                          _auth_header=lambda: {}, chat_json=lambda *a, **k: {"ok": True})
    client = VerifiedLLMClient(raw)
    with pytest.raises(ObjNavInternalError, match="not provisioned"):
        client.health()
    payload["models"] = [{"name": "test-model:latest", "digest": "one"}]
    assert client.health()["immutable_revision_exposed"]
    payload["models"] = [{"name": "test-model:latest", "digest": "two"}]
    with pytest.raises(ObjNavInternalError, match="changed"):
        client.chat_json("system", "user")


@pytest.mark.parametrize("url", ["http://user:secret@host", "http://host?key=secret", "file:///tmp/model"])
def test_service_urls_do_not_leak_credentials(url):
    with pytest.raises(ValueError):
        public_service_url(url)


def test_real_http_service_pins_metadata_and_preserves_rgb():
    import threading
    from sparx_agency.core.common.types.perception import Detection2D
    from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import _ServerContext, _make_server

    class Detector:
        def detect(self, rgb):
            assert float(rgb[..., 0].mean()) > 240 and float(rgb[..., 2].mean()) < 15
            return [Detection2D(label="chair", score=0.9, bbox_xyxy=(1, 1, 6, 6), frame_w=8, frame_h=8)]

        def set_prompts(self, names):
            pass

    metadata = {"checkpoint_sha256": "synthetic", "detector_config": {"conf_thresh": 0.25}}
    server = _make_server(_ServerContext(Detector(), "test", "cpu", ["chair"], metadata), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpDetector("http://127.0.0.1:%d" % server.server_port, ["chair"])
    try:
        image = np.zeros((8, 8, 3), np.uint8)
        image[..., 0] = 255
        assert client.detect(image)[0].cls == "chair"
        assert client.health()["metadata"] == metadata
        client.session.post(client.url + "/set_classes", json={"classes": ["bed"]}).raise_for_status()
        with pytest.raises(ObjNavInternalError, match="vocabulary"):
            client.detect(image)
    finally:
        client.session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


