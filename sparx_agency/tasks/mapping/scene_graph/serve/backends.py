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
from sparx_agency.core.mapping.detection.snapshots import snapshot_files
from sparx_agency.tasks.common.model_registry.download.verify import sha256_of
from sparx_agency.tasks.mapping.scene_graph.serve.contract import JPEG_QUALITY


def checkpoint_identity(path):
    """Hash YOLO bytes or the complete local HF inference bundle, not its name."""
    path = Path(path).expanduser()
    if path.is_file():
        digest = sha256_of(path)
        return {"checkpoint_sha256": digest, "checkpoint_files": {path.name: digest}}
    files = snapshot_files(path)
    manifest = {p.name: sha256_of(p) for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    source = path / "source.json"
    return {"checkpoint_sha256": digest, "checkpoint_files": manifest,
            "checkpoint_source": json.loads(source.read_text()) if source.is_file() else
            {"kind": "local snapshot", "upstream_revision": "unverified"}}


def _configure_runtime(args):
    """Require an idle GPU unless the operator explicitly authorized sharing."""
    if args.device.startswith("cuda") and not getattr(args, "allow_shared_gpu", False):
        # nvidia-smi is independent of torch/CUDA context creation. Query ALL
        # cards conservatively, including graphics memory used by simulators.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        used = [int(line.strip()) for line in result.stdout.splitlines()]
        if not used or max(used) > 512:
            raise RuntimeError("GPU occupied (>512 MiB); use --device cpu, an idle GPU, or explicitly --allow-shared-gpu after ownership/memory preflight")
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
    identity = selected_checkpoint_identity(args)
    if args.backend == "yolo_world" and not path.is_file():
        raise ValueError("YOLO-World requires a local .pt file")
    _configure_runtime(args)
    if args.backend == "yolo_world":
        from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig

        config = YoloWorldConfig(model_path=str(path), device=args.device,
                                 conf_thresh=args.conf, imgsz=args.imgsz,
                                 iou_thresh=args.iou, max_det=args.max_det,
                                 clip_path=getattr(args, "clip_model", None))
    elif args.backend in ("llmdet", "grounding_dino", "grounded_vlm", "hybrid"):
        from sparx_agency.core.mapping.detection.llmdet import LlmDetConfig
        from sparx_agency.core.mapping.detection.grounding_dino import GroundingDinoConfig

        config_type = LlmDetConfig if args.backend == "llmdet" else GroundingDinoConfig
        config = config_type(model_path=str(path), device=args.device,
                             conf_thresh=args.conf, shortest_edge=args.shortest_edge,
                             longest_edge=args.longest_edge, max_det=args.max_det,
                             chunk_size=args.chunk_size, dtype=args.dtype)
        if args.backend in ("grounded_vlm", "hybrid"):
            config = _verification_config(args, config)
    else:
        raise ValueError("Unknown backend: %s" % args.backend)
    registry = default_detection_registry(**{args.backend + "_config": config})
    detector = registry.create(args.backend)
    detector.set_prompts(classes)
    if hasattr(detector, "warmup"):
        detector.warmup()
    else:
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    # Ultralytics select_device resets CPU threads during its first predict.
    # Apply the operator's setting AFTER that initialization as well.
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)
    if selected_checkpoint_identity(args) != identity:
        raise RuntimeError("Checkpoint changed during model loading")
    return detector, detector_metadata(detector, args.backend, identity,
                                       allow_shared_gpu=getattr(args, "allow_shared_gpu", False))


def selected_checkpoint_identity(args):
    """Only selected models participate: grounded-only does not require YOLO."""
    primary = checkpoint_identity(args.model)
    clip_path = getattr(args, "clip_model", None)
    if args.backend == "yolo_world" and clip_path:
        auxiliary = checkpoint_identity(clip_path)
        return dict(primary, auxiliary_clip=auxiliary)
    if args.backend not in ("grounded_vlm", "hybrid"):
        return primary
    components = {"grounding_dino": primary, "blip2": checkpoint_identity(args.blip2_model)}
    if args.backend == "hybrid":
        if not Path(args.yolo_model).is_file():
            raise FileNotFoundError("Hybrid mode requires a local YOLO .pt checkpoint")
        components["yolo_world"] = checkpoint_identity(args.yolo_model)
        components["clip"] = checkpoint_identity(args.clip_model)
    digest = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
    return {"checkpoint_sha256": digest, "components": components,
            "checkpoint_files": {name + "/" + file: value for name, info in components.items()
                                 for file, value in info["checkpoint_files"].items()}}


def _verification_config(args, grounding):
    from sparx_agency.core.mapping.detection.blip2 import Blip2Config
    from sparx_agency.core.mapping.detection.grounded_vlm import GroundedVlmConfig
    from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig

    yolo = None
    if args.backend == "hybrid":
        yolo = YoloWorldConfig(model_path=args.yolo_model, device=args.device,
                              conf_thresh=args.conf if args.yolo_conf is None else args.yolo_conf,
                              imgsz=args.imgsz, iou_thresh=args.iou, max_det=args.max_det,
                              clip_path=args.clip_model)
    labels = () if args.verify_labels == "*" else tuple(s.strip() for s in args.verify_labels.split(","))
    return GroundedVlmConfig(grounding=grounding, yolo=yolo,
                            verifier=Blip2Config(args.blip2_model, args.device, args.dtype, args.blip2_max_tokens),
                            verify_labels=labels, max_verifications=args.max_verifications,
                            duplicate_iou=args.verification_iou)


def detector_metadata(detector, backend, identity, *, allow_shared_gpu=False):
    """Static identity; request timing and live memory do not cause identity drift."""
    import torch

    names = ["torch", "numpy", "Pillow", "requests"]
    if backend != "yolo_world":
        names += ["transformers", "tokenizers", "safetensors", "huggingface-hub"]
    if backend in ("yolo_world", "hybrid"):
        names += ["ultralytics", "torchvision"]
    if backend in ("grounded_vlm", "hybrid"):
        names.append("sentencepiece")
    packages = {name: importlib.metadata.version(name) for name in names}
    import cv2
    packages["opencv_loaded"] = cv2.__version__
    preprocessing = (detector.preprocessing() if backend != "yolo_world" else
                     yolo_preprocessing(detector))
    if backend == "hybrid":
        preprocessing["yolo_world"] = yolo_preprocessing(detector.yolo)
    config = dict(asdict(detector.cfg), conf_thresh=detector.cfg.conf_thresh)
    return dict(identity, backend=backend, detector_config=config,
                packages=packages, preprocessing=preprocessing,
                wire={"encoding": "JPEG", "quality": JPEG_QUALITY, "decoder": "OpenCV BGR -> RGB"},
                runtime={"torch_threads": torch.get_num_threads(),
                         "cuda_version": torch.version.cuda,
                         "allow_shared_gpu": bool(allow_shared_gpu),
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
    if metadata.get("backend") in ("llmdet", "grounding_dino", "grounded_vlm", "hybrid"):
        updated["preprocessing"] = detector.preprocessing()
        if metadata["backend"] == "hybrid":
            updated["preprocessing"]["yolo_world"] = yolo_preprocessing(detector.yolo)
    elif metadata.get("backend") == "yolo_world":
        updated["preprocessing"] = yolo_preprocessing(detector)
    return updated
