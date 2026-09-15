"""Backend construction and auditable local checkpoint identity for the service."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.tasks.common.model_registry.download.verify import sha256_of
from sparx_agency.tasks.mapping.scene_graph.serve.contract import JPEG_QUALITY


def checkpoint_identity(path):
    """Hash YOLO bytes or the complete local HF inference bundle, not its name."""
    path = Path(path)
    if path.is_file():
        return {"checkpoint_sha256": sha256_of(path),
                "checkpoint_files": {path.name: sha256_of(path)}}
    if not path.is_dir() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError("Missing local detector checkpoint: %s" % path)
    files = sorted(p for p in path.iterdir()
                   if p.is_file() and p.suffix in (".json", ".txt", ".safetensors"))
    manifest = {p.name: sha256_of(p) for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source = path / "source.json"
    return {"checkpoint_sha256": digest, "checkpoint_files": manifest,
            "checkpoint_source": json.loads(source.read_text()) if source.is_file() else
            {"kind": "local snapshot", "upstream_revision": "unverified"}}


def _configure_runtime(args):
    """Refuse an occupied GPU before creating a CUDA context; never kill owners."""
    if args.device.startswith("cuda"):
        # nvidia-smi is independent of torch/CUDA context creation. Query ALL
        # cards conservatively, including graphics memory used by simulators.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        used = [int(line.strip()) for line in result.stdout.splitlines()]
        if not used or max(used) > 512:
            raise RuntimeError("GPU occupied (>512 MiB); use --device cpu or a separate idle GPU host")
    import torch

    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
        torch.cuda.set_per_process_memory_fraction(0.70, args.device)


def build_detector(args, classes):
    """Construct via the shared registry and warm-load the requested backend."""
    path = Path(args.model)
    identity = checkpoint_identity(path)
    if args.backend == "yolo_world" and not path.is_file():
        raise ValueError("YOLO-World requires a local .pt file")
    _configure_runtime(args)
    if args.backend == "yolo_world":
        from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig

        config = YoloWorldConfig(model_path=str(path), device=args.device,
                                 conf_thresh=args.conf, imgsz=args.imgsz,
                                 iou_thresh=args.iou, max_det=args.max_det)
    elif args.backend == "llmdet":
        from sparx_agency.core.mapping.detection.llmdet import LlmDetConfig

        config = LlmDetConfig(model_path=str(path), device=args.device,
                             conf_thresh=args.conf, shortest_edge=args.shortest_edge,
                             longest_edge=args.longest_edge, max_det=args.max_det,
                             chunk_size=args.chunk_size, dtype=args.dtype)
    else:
        raise ValueError("Unknown backend: %s" % args.backend)
    registry = default_detection_registry(**{args.backend + "_config": config})
    detector = registry.create(args.backend)
    detector.set_prompts(classes)
    detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    # Ultralytics select_device resets CPU threads during its first predict.
    # Apply the operator's setting AFTER that initialization as well.
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)
    if checkpoint_identity(path) != identity:
        raise RuntimeError("Checkpoint changed during model loading")
    return detector, detector_metadata(detector, args.backend, identity)


def detector_metadata(detector, backend, identity):
    """Static identity; request timing and live memory do not cause identity drift."""
    import torch

    names = ["torch", "numpy", "Pillow", "requests"]
    names += (["transformers", "tokenizers", "safetensors", "huggingface-hub"]
              if backend == "llmdet" else ["ultralytics", "torchvision"])
    packages = {name: importlib.metadata.version(name) for name in names}
    import cv2
    packages["opencv_loaded"] = cv2.__version__
    preprocessing = (detector.preprocessing() if backend == "llmdet" else
                     yolo_preprocessing(detector))
    return dict(identity, backend=backend, detector_config=asdict(detector.cfg),
                packages=packages, preprocessing=preprocessing,
                wire={"encoding": "JPEG", "quality": JPEG_QUALITY, "decoder": "OpenCV BGR -> RGB"},
                runtime={"torch_threads": torch.get_num_threads(),
                         "cuda_version": torch.version.cuda,
                         "gpu_memory_fraction": 0.70 if detector.cfg.device.startswith("cuda") else None})


def yolo_preprocessing(detector):
    """Pin the actual cached text features, including auxiliary CLIP effects."""
    features = detector._model.model.txt_feats.detach().float().cpu().numpy()
    return {"input": "RGB uint8 HWC -> BGR for ultralytics",
            "resize": "ultralytics letterbox", "imgsz": detector.cfg.imgsz,
            "boxes": "original pixels XYXY, integer",
            "text_encoder": "Ultralytics CLIP ViT-B/32",
            "prompt_features_shape": list(features.shape),
            "prompt_features_sha256": hashlib.sha256(features.tobytes()).hexdigest()}


def refresh_vocabulary_metadata(detector, metadata):
    """Refresh caption provenance atomically with a successful set_classes."""
    updated = dict(metadata)
    if metadata.get("backend") == "llmdet":
        updated["preprocessing"] = detector.preprocessing()
    elif metadata.get("backend") == "yolo_world":
        updated["preprocessing"] = yolo_preprocessing(detector)
    return updated
