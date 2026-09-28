"""Official pretrained LLMDet via Transformers, behind the shared detector ABC.

Only inference methods import model packages. Loading is offline and strict:
no remote code, downloads, model substitution, or random missing weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.detection.llmdet_loading import (
    checkpoint_model_class, verify_decoder_weights)
from sparx_agency.core.mapping.detection.llmdet_postprocess import (
    caption_and_spans, decode_token_detections, tokens_for_spans, validate_prompts)
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


@dataclass(frozen=True)
class LlmDetConfig:
    """Local HF snapshot and explicit inference settings (not YOLO scores).

    ``conf_thresh`` is a starting operating point, not a calibrated probability.
    The HTTP entrypoint requires an explicit threshold for this backend.
    """

    model_path: str = "models/objnav/llmdet-large"
    device: str = "cpu"
    conf_thresh: float = 0.3
    shortest_edge: int = 800
    longest_edge: int = 1333
    max_det: int = 100
    chunk_size: int = 80
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("LLMDet requires a local checkpoint directory")
        if not math.isfinite(self.conf_thresh) or not 0 <= self.conf_thresh <= 1:
            raise ValueError("conf_thresh must be finite and in [0, 1]")
        for name in ("shortest_edge", "longest_edge", "max_det", "chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.longest_edge < self.shortest_edge:
            raise ValueError("longest_edge cannot be smaller than shortest_edge")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported LLMDet dtype")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU LLMDet requires float32; no implicit dtype fallback")


class LlmDetDetector(DetectionModel):
    """Text-category localization; RGB uint8 in, original-frame XYXY out."""

    def __init__(self, config: Optional[LlmDetConfig] = None) -> None:
        self.cfg = config or LlmDetConfig()
        self._model = self._processor = None
        self._prompts: List[str] = []
        self._queries = []
        self._verified_decoder_tensors = 0

    @property
    def prompts(self) -> List[str]:
        return list(self._prompts)

    def set_prompts(self, prompts: Sequence[str]) -> None:
        cleaned = validate_prompts(prompts)
        # Build first: a rejected reconfiguration must leave the live model intact.
        queries = (self._prepare_queries(cleaned, self._processor.tokenizer,
                                         self._model.config.max_text_len)
                   if self._model is not None else [])
        self._prompts, self._queries = cleaned, queries

    def _prepare_queries(self, prompts, tokenizer, max_tokens):
        """Deterministically chunk by both category count AND actual BERT tokens."""
        queries, start = [], 0
        while start < len(prompts):
            end = min(len(prompts), start + self.cfg.chunk_size)
            while end > start:
                labels = prompts[start:end]
                caption, spans = caption_and_spans(labels)
                encoded = tokenizer(caption, return_offsets_mapping=True,
                                    truncation=False, add_special_tokens=True)
                if len(encoded["input_ids"]) <= max_tokens:
                    groups = tokens_for_spans(encoded["offset_mapping"], spans)
                    queries.append((caption, labels, groups))
                    break
                end -= 1
            if end == start:
                raise ValueError("A single category exceeds LLMDet's text token limit")
            start = end
        return queries

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        path = Path(self.cfg.model_path)
        if not path.is_dir() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError("LLMDet needs a local HF safetensors snapshot: %s" % path)
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, MMGroundingDinoForObjectDetection
        except ImportError as exc:
            raise ImportError("LLMDet needs the separate detector environment; see serve/README.md") from exc
        config = AutoConfig.from_pretrained(str(path), local_files_only=True,
                                            trust_remote_code=False)
        if config.model_type != "mm-grounding-dino":
            raise ValueError("Expected official iSEE-Laboratory LLMDet MM-GDINO weights")
        # Portable PyTorch deformable attention, not an implicit JIT/CUDA build.
        config.disable_custom_kernels = True
        processor = AutoProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_fast=False)
        queries = self._prepare_queries(self._prompts, processor.tokenizer, config.max_text_len)
        model_class = checkpoint_model_class(MMGroundingDinoForObjectDetection)
        model, info = model_class.from_pretrained(
            str(path), config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=getattr(torch, self.cfg.dtype),
            output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("LLMDet checkpoint did not load exactly: %s" % info)
        verified = verify_decoder_weights(model, path / "model.safetensors")
        model.eval().to(self.cfg.device)
        self._model, self._processor, self._queries = model, processor, queries
        self._verified_decoder_tensors = verified
        return model

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        if not self._prompts:
            raise RuntimeError("LLMDet.detect requires set_prompts first")
        image = np.asarray(rgb)
        if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                or min(image.shape[:2]) == 0):
            raise ValueError("LLMDet expects non-empty HxWx3 uint8 RGB")
        model = self._ensure_model()
        import torch

        detections = []
        with torch.inference_mode():
            for caption, labels, groups in self._queries:
                inputs = self._processor(
                    images=np.ascontiguousarray(image), text=caption,
                    size={"shortest_edge": self.cfg.shortest_edge,
                          "longest_edge": self.cfg.longest_edge},
                    return_tensors="pt", truncation=False)
                if inputs["input_ids"].shape[-1] > model.config.max_text_len:
                    raise RuntimeError("Unexpected tokenizer drift/truncation")
                inputs = inputs.to(self.cfg.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, self.cfg.dtype))
                outputs = model(**inputs)
                detections.extend(decode_token_detections(
                    outputs.logits[0].sigmoid().float().cpu().numpy(),
                    outputs.pred_boxes[0].float().cpu().numpy(), groups, labels,
                    image.shape[:2], self.cfg.conf_thresh, self.cfg.max_det))
        return sorted(detections, key=lambda d: -d.score)[:self.cfg.max_det]

    def preprocessing(self):
        """Actual processor and caption settings, published in service provenance."""
        if self._processor is None:
            raise RuntimeError("Load LLMDet before requesting preprocessing provenance")
        return {"input": "RGB uint8 HWC", "boxes": "original pixels XYXY, clipped, integer",
                "image_processor": self._processor.image_processor.to_dict(),
                "size_override": {"shortest_edge": self.cfg.shortest_edge,
                                  "longest_edge": self.cfg.longest_edge},
                "tokenizer": type(self._processor.tokenizer).__name__,
                "max_text_len": self._model.config.max_text_len,
                "captions": [query[0] for query in self._queries],
                "score": "mean sigmoid over category tokens; query/category top-k",
                "nms": "none; shared downstream alias suppression unchanged",
                "weight_loading": "decoder aliases only; independent pretrained heads verified",
                "verified_decoder_tensors": self._verified_decoder_tensors,
                "disable_custom_kernels": True}
