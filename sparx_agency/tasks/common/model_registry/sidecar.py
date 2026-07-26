"""Per-artifact ``.engine.json`` sidecar: the version lock for a local engine.

Records what device/TRT build produced the file next to it, so a stale or
wrong-device engine fails with "rebuild, here's why" instead of a confusing
crash inside ``deserialize_cuda_engine``. Mirrors the manifest
``yolo_world_trt/build_engine.py`` already writes next to each engine it builds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from sparx_agency.tasks.common.model_registry.errors import IncompatibleArtifactError
from sparx_agency.tasks.common.model_registry.key import ModelKey

logger = logging.getLogger(__name__)


def sidecar_path(artifact_path: Path) -> Path:
    return Path(str(artifact_path) + ".json")


def read_sidecar(artifact_path: Path) -> Optional[dict]:
    p = sidecar_path(artifact_path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_sidecar(artifact_path: Path, data: dict) -> None:
    sidecar_path(artifact_path).write_text(json.dumps(data, indent=2, sort_keys=True))


def build_sidecar(key: ModelKey, profile, *, origin: str, trt_version: Optional[str] = None,
                   onnx_sha256: Optional[str] = None, engine_sha256: Optional[str] = None,
                   inputs=None, outputs=None) -> dict:
    """Assemble the sidecar dict for an artifact just downloaded or built."""
    return {
        "stem": key.stem(),
        "model_id": key.model_id,
        "role": key.role,
        "precision": key.precision,
        "input_hw": [key.height, key.width] if key.height and key.width else None,
        "device": key.device,
        "target_tag": profile.target_tag,
        "trt_version": trt_version,
        "sm": profile.sm,
        "arch": profile.arch,
        "gpu_name": profile.gpu_name,
        "power_budget_w": profile.power_budget_w,
        "onnx_sha256": onnx_sha256,
        "engine_sha256": engine_sha256,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "origin": origin,
    }


def _trt_major(version: Optional[str]) -> Optional[int]:
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return None


def validate_sidecar(sidecar: Optional[dict], *, profile, strict: bool = False) -> None:
    """Raise if ``sidecar`` describes an artifact incompatible with ``profile``.

    Arch/SM checks are pure stdlib (from the already-detected hardware
    profile). The TRT-version check is best-effort: it only runs if
    ``tensorrt`` happens to be importable, so calling this never forces a
    TensorRT dependency onto the resolve path.
    """
    if sidecar is None:
        return

    arch = sidecar.get("arch")
    if arch and arch != profile.arch:
        raise IncompatibleArtifactError(
            f"artifact built for arch={arch!r}, this machine is {profile.arch!r} -- "
            f"engines don't cross architectures, this needs a rebuild or a different download")

    sm = sidecar.get("sm")
    if sm is not None and profile.sm is not None and sm != profile.sm:
        raise IncompatibleArtifactError(
            f"artifact built for sm={sm}, this GPU is sm={profile.sm} -- rebuild required")

    recorded_trt = sidecar.get("trt_version")
    if not recorded_trt:
        return
    try:
        import tensorrt as trt
    except ImportError:
        return  # can't check without tensorrt installed; arch/sm checks already ran
    current_trt = str(trt.__version__)
    if recorded_trt == current_trt:
        return
    if strict:
        raise IncompatibleArtifactError(
            f"TRT version mismatch: artifact built with {recorded_trt}, "
            f"runtime is {current_trt} (SPARX_MODEL_STRICT=1)")
    if _trt_major(recorded_trt) != _trt_major(current_trt):
        raise IncompatibleArtifactError(
            f"TRT major version mismatch: artifact built with {recorded_trt}, "
            f"runtime is {current_trt} -- rebuild with `cli build`")
    logger.warning("TRT version drift: artifact built with %s, runtime is %s "
                    "(minor mismatch, proceeding)", recorded_trt, current_trt)
