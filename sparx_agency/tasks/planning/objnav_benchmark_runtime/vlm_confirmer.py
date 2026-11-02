"""The Ollama-served vision-language model behind the closing's ``CandidateConfirmer``: one crop, one yes/no question."""
from __future__ import annotations

import base64
import io
import time
from typing import Sequence

import numpy as np

from sparx_agency.core.planning.object_search.closing.vlm_confirmation import (
    UNSURE, Verdict, crop_around, parse_answer, question_for)


class OllamaConfirmer:
    """Qwen-VL style models behind an Ollama server; any service failure is ``unsure``."""

    def __init__(self, model: str, url: str, margin: float, timeout_s: float = 120.0) -> None:
        self.model, self.url, self.margin, self.timeout_s = model, url.rstrip("/"), margin, timeout_s

    def confirm(self, rgb: np.ndarray, box_xyxy: Sequence[float], label: str) -> Verdict:
        started = time.monotonic()
        try:
            content = self._ask(crop_around(rgb, box_xyxy, self.margin), question_for(label))
            return Verdict(parse_answer(content), _ms(started), content)
        except Exception as exc:
            return Verdict(UNSURE, _ms(started), "error: %r" % (exc,))

    def _ask(self, crop: np.ndarray, question: str) -> str:
        import requests
        from PIL import Image

        png = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(crop).astype(np.uint8)).save(png, format="PNG")
        body = {"model": self.model, "stream": False, "format": "json", "options": {"temperature": 0, "num_predict": 3000},
                "messages": [{"role": "user", "content": question,
                              "images": [base64.b64encode(png.getvalue()).decode()]}]}
        reply = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout_s)
        reply.raise_for_status()
        return reply.json()["message"]["content"]


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 1)


def attach_confirmer(policy) -> None:
    """Give the policy the VLM its settings ask for (its closing reads it on every reset); nothing when ``vlm_confirm`` is off."""
    s = policy.settings.target_closing
    if s.vlm_confirm:
        policy.target_confirmer = OllamaConfirmer(s.vlm_model, s.vlm_url, s.vlm_crop_margin)
