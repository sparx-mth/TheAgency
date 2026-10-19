"""Grounding DINO, not DINOv2: local text-category detection with exact labels.

Grounding DINO and MM-GDINO share a token-logit/normalized-box contract. Reuse
LLMDet's caption chunking, RGB preprocessing and numpy decoding, not its model
or independent-head loading. Heavy imports stay inside the loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sparx_agency.core.mapping.detection.llmdet import LlmDetConfig, LlmDetDetector
from sparx_agency.core.mapping.detection.llmdet_loading import verify_decoder_weights
from sparx_agency.core.mapping.detection.grounding_dino_loading import require_materialized, shared_bbox_model_class
from sparx_agency.core.mapping.detection.snapshots import snapshot_files


@dataclass(frozen=True)
class GroundingDinoConfig(LlmDetConfig):
    """Same explicit category-scoring settings; never substitute MM-GDINO weights."""

    model_path: str = "models/objnav/grounding-dino-base"


class GroundingDinoDetector(LlmDetDetector):
    """RGB uint8 input; exact prompted labels and original-image XYXY output."""

    def __init__(self, config=None):
        super().__init__(config or GroundingDinoConfig())

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        path = Path(self.cfg.model_path).expanduser()
        snapshot_files(path)
        import torch
        from transformers import AutoConfig, AutoProcessor, GroundingDinoForObjectDetection

        config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if config.model_type != "grounding-dino":
            raise ValueError("Expected a Grounding DINO snapshot, not DINOv2 or MM-GDINO")
        config.disable_custom_kernels = True
        processor = AutoProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_fast=False)
        queries = self._prepare_queries(self._prompts, processor.tokenizer, config.max_text_len)
        model_class = shared_bbox_model_class(GroundingDinoForObjectDetection, config,
                                             path / "model.safetensors")
        model, info = model_class.from_pretrained(
            str(path), config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=getattr(torch, self.cfg.dtype), output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("Grounding DINO checkpoint did not load exactly: %s" % info)
        # The official Base snapshot is single-file. Verify actual decoder values
        # as well as loader keys; an apparently clean tied-weight load can lie.
        self._verified_decoder_tensors = verify_decoder_weights(model, path / "model.safetensors")
        require_materialized(model)
        model.eval().to(self.cfg.device)
        self._model, self._processor, self._queries = model, processor, queries
        return model

    def preprocessing(self):
        details = super().preprocessing()
        details["weight_loading"] = "direct aliases to the saved shared Base decoder; values verified"
        return details
