"""The Ollama client of the vision-model check before a STOP: one boxed crop, one yes/no question."""
import base64
import json

import numpy as np

from sparx_agency.core.planning.object_search.closing.box_check import box_prompt, category_phrase, crop_around_box

REQUEST_TIMEOUT_S = 20.0


class OllamaBoxCheck:
    """Asks a vision model whether the object in a boxed crop is the goal category: ``yes``, ``no`` or ``unsure``."""

    def __init__(self, url, model, post=None, timeout_s=REQUEST_TIMEOUT_S):
        import requests
        self.url, self.model, self.timeout_s = url, model, timeout_s
        self.post = post or requests.post

    def verdict(self, rgb, xyxy, category):
        try:
            reply = self.post(self.url, json=self._request(rgb, xyxy, category), timeout=self.timeout_s)
            answer = json.loads(reply.json()["message"]["content"])["answer"]
        except Exception:
            return "unsure"
        answer = str(answer).strip().lower()
        return answer if answer in ("yes", "no") else "unsure"

    def _request(self, rgb, xyxy, category):
        import cv2
        crop = crop_around_box(np.asarray(rgb), xyxy)
        png = cv2.imencode(".png", crop[:, :, ::-1])[1].tobytes()
        return {"model": self.model, "stream": False, "think": False, "format": "json",
                "options": {"temperature": 0, "num_predict": 30, "num_ctx": 4096},
                "messages": [{"role": "user", "content": box_prompt(category_phrase(category)),
                              "images": [base64.b64encode(png).decode()]}]}
