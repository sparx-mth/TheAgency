"""One-scene demo service bring-up using existing environments and checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[5]


def environment_python(name, override=""):
    """Find an existing interpreter; never create or globally modify an environment."""
    if override:
        path = Path(override).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise ValueError("Python executable not found: %s" % path)
    for root in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path.home() / "miniforge3"):
        path = root / "envs" / name / "bin" / "python"
        if path.is_file():
            return str(path)
    raise ValueError("Set the existing %s Python interpreter in the demo settings" % name)


def health(url):
    try:
        with urlopen(url, timeout=3) as response:
            return json.load(response)
    except (OSError, ValueError, URLError):
        return None


def start_ollama(container="ollama-scene-graph"):
    """Reuse the configured local Ollama container only if it has no GPU devices."""
    if health("http://127.0.0.1:11434/api/tags") is not None:
        return
    result = subprocess.run(["docker", "inspect", container], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)[0]
    config = info["HostConfig"]
    if config.get("DeviceRequests") or config.get("Devices") or config.get("Runtime") not in ("runc", ""):
        raise RuntimeError("Refusing GPU-enabled Ollama; Habitat needs the rendering GPU")
    subprocess.run(["docker", "start", container], check=True, capture_output=True, text=True)
    for _ in range(30):
        if health("http://127.0.0.1:11434/api/tags") is not None:
            return
        time.sleep(1)
    raise RuntimeError("Ollama did not become ready; inspect its container log")


def start_detector(python, checkpoint, vocabulary, output, port=18092, cancelled=None):
    """Start a dedicated CPU detector; never take over another service's port/vocabulary.

    Returns a process only when this call started it. Callers stop only that
    owned process; a pre-existing matching service is left alone.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Local detector checkpoint missing: %s" % checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    url = "http://127.0.0.1:%d/health" % port
    current = health(url)
    if current is not None:
        if tuple(current.get("classes", ())) != tuple(vocabulary) or current.get("device") != "cpu":
            raise RuntimeError("Port %d holds another detector; choose another dedicated port" % port)
        threshold = current.get("metadata", {}).get("detector_config", {}).get("conf_thresh", 1.0)
        if float(threshold) > 0.05:
            raise RuntimeError("Restart the dedicated detector with --conf 0.05 for door candidates")
        if current.get("metadata", {}).get("checkpoint_sha256") != digest.hexdigest():
            raise RuntimeError("Selected checkpoint differs from the running detector; use another dedicated port or restart it")
        return None
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Local YOLO-World checkpoint missing: %s" % checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    log = (output / "detector.log").open("ab")
    process = subprocess.Popen([
        python, "-u", "-m", "sparx_agency.tasks.mapping.scene_graph.serve.detection_server",
        "--model", str(checkpoint), "--device", "cpu", "--host", "127.0.0.1",
        "--port", str(port), "--conf", "0.05", "--classes", ",".join(vocabulary)],
        cwd=str(REPO_ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    log.close()
    try:
        for _ in range(180):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Detector startup cancelled")
            if process.poll() is not None:
                raise RuntimeError("Detector startup failed; see %s" % (output / "detector.log"))
            current = health(url)
            if current is not None:
                if tuple(current.get("classes", ())) != tuple(vocabulary) or current.get("device") != "cpu":
                    raise RuntimeError("A different detector owns the selected port")
                return process
            time.sleep(1)
        raise RuntimeError("Detector startup timed out; see detector.log (auxiliary weights may be downloading)")
    except BaseException:
        stop_process(process)
        raise


def stop_process(process):
    """Stop only a process created by this launcher, with a bounded shutdown."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


