"""Detector contract for the webcam demo.

Mirrors the shape of the production ``YoloTRTDetector`` (``set_prompts`` +
``detect``) so the offline :class:`~...object_approach_offline.pipeline.TargetLockPipeline`
is driven the same way it is on the drone -- only the *source* of the
:class:`~sparx_agency.core.common.types.perception.Detection2D` list changes. A
laptop backend (colour blob, or a CPU Ultralytics model) stands in for the
TensorRT open-vocab detector that only exists on the Jetson.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D


class WebcamDetector(ABC):
    """A laptop-runnable object detector producing per-frame ``Detection2D``s."""

    @abstractmethod
    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Set the target (and any distractor) labels; ``prompts[0]`` is the target."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        """Detect objects in an ``HxWx3`` RGB frame; return the ``Detection2D`` list."""
        raise NotImplementedError
