"""Conservative visual verification; generated answers are not probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparx_agency.core.mapping.detection.snapshots import snapshot_files


PROMPT_VERSION = "marked-frame-and-crop/1"
CONTEXT_QUESTION = "Question: Does the red rectangle show {label}? Answer yes, no, or uncertain. Answer:"
CROP_QUESTION = "Question: Does this image show {label}? Answer yes, no, or uncertain. Answer:"


@dataclass(frozen=True)
class Blip2Config:
    model_path: str = "models/objnav/blip2-flan-t5-xl"
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = 12

    def __post_init__(self):
        if not str(self.model_path).strip():
            raise ValueError("BLIP-2 requires a local snapshot directory")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported BLIP-2 dtype")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU BLIP-2 requires explicit float32; no precision fallback")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 32:
            raise ValueError("BLIP-2 answer length must be between 1 and 32 tokens")


def answer_verdict(answer):
    """Accept only the requested short answers; never guess from a substring."""
    text = str(answer).strip().lower().rstrip(".!?").strip()
    return {"yes": "yes", "no": "no"}.get(text, "uncertain")


class Blip2Verifier:
    """Two views of one observation, not two independent confirmations."""

    def __init__(self, config=None):
        self.cfg = config or Blip2Config()
        self._model = self._processor = None

    def load(self):
        if self._model is not None:
            return
        path = Path(self.cfg.model_path).expanduser()
        snapshot_files(path)
        import torch
        from transformers import AutoConfig, AutoProcessor, Blip2ForConditionalGeneration

        config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if config.model_type != "blip-2" or config.text_config.model_type != "t5":
            raise ValueError("Expected BLIP-2 FLAN-T5 weights; other architectures are not substituted")
        processor = AutoProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_fast=False)
        model, info = Blip2ForConditionalGeneration.from_pretrained(
            str(path), config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=getattr(torch, self.cfg.dtype), output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError("BLIP-2 checkpoint did not load exactly: %s" % info)
        model.eval().to(self.cfg.device)
        self._model, self._processor = model, processor

    def verify(self, rgb, detection):
        """Return JSON-safe evidence for this box only; never relabel or add boxes."""
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("BLIP-2 expects HxWx3 uint8 RGB")
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Verification box is outside the submitted frame")
        self.load()
        from PIL import Image, ImageDraw

        source = Image.fromarray(np.ascontiguousarray(image))
        marked = source.copy()
        ImageDraw.Draw(marked).rectangle((x1, y1, x2 - 1, y2 - 1),
                                        outline=(255, 0, 0), width=max(2, w // 200))
        crop = source.crop((x1, y1, x2, y2))
        questions = [CONTEXT_QUESTION.format(label=detection.label),
                     CROP_QUESTION.format(label=detection.label)]
        answers = self._answer([marked, crop], questions)
        if len(answers) != 2:
            raise RuntimeError("BLIP-2 returned an unexpected number of answers")
        decisions = [answer_verdict(answer) for answer in answers]
        verdict = "yes" if decisions == ["yes", "yes"] else "no" if "no" in decisions else "uncertain"
        return {"verdict": verdict, "answers": list(answers), "questions": questions,
                "prompt_version": PROMPT_VERSION}

    def _answer(self, images, questions):
        import torch

        with torch.inference_mode():
            inputs = self._processor(images=images, text=questions, padding=True,
                                     return_tensors="pt").to(self.cfg.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, self.cfg.dtype))
            outputs = self._model.generate(**inputs, do_sample=False, num_beams=1,
                                           max_new_tokens=self.cfg.max_new_tokens)
            return self._processor.batch_decode(outputs, skip_special_tokens=True)

    def preprocessing(self):
        if self._processor is None:
            raise RuntimeError("Load BLIP-2 before requesting provenance")
        return {"input": "RGB uint8; marked full image and exact box crop",
                "processor": self._processor.image_processor.to_dict(),
                "tokenizer": type(self._processor.tokenizer).__name__,
                "prompt_version": PROMPT_VERSION,
                "questions": [CONTEXT_QUESTION, CROP_QUESTION],
                "generation": {"do_sample": False, "num_beams": 1,
                               "max_new_tokens": self.cfg.max_new_tokens},
                "acceptance": "both answers exactly yes; unknown answers abstain",
                "score": "none; preserve detector score, never a VLM probability"}
