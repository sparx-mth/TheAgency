"""Registry of open-vocabulary detector backends (factory idiom).

Mirrors :mod:`sparx_agency.core.planning.trackers.registry`. Lets a task node
select YOLO-World or LLMDet without importing model dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


@dataclass(frozen=True)
class DetectorFactory:
    """A named zero-arg factory producing a :class:`DetectionModel`."""

    name: str
    create: Callable[[], DetectionModel]


class DetectionRegistry:
    """Name -> detector-factory map."""

    def __init__(self) -> None:
        self._factories: Dict[str, DetectorFactory] = {}

    def register(self, factory: DetectorFactory) -> None:
        if factory.name in self._factories:
            raise ValueError("Detector '%s' already registered" % factory.name)
        self._factories[factory.name] = factory

    def names(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> DetectionModel:
        if name not in self._factories:
            raise KeyError(
                "Unknown detector '%s'. Available: %s" % (name, self.names())
            )
        return self._factories[name].create()


def default_detection_registry(*, yolo_world_config=None, llmdet_config=None,
                               grounding_dino_config=None, grounded_vlm_config=None,
                               hybrid_config=None) -> DetectionRegistry:
    """Registry with the built-in backends registered.

    The factory imports the backend lazily so a registry can be constructed (and
    listed) without ultralytics/torch installed.
    """
    reg = DetectionRegistry()

    def _make_yolo_world() -> DetectionModel:
        from sparx_agency.core.mapping.detection.yolo_world import (
            YoloWorldConfig,
            YoloWorldDetector,
        )

        return YoloWorldDetector(yolo_world_config or YoloWorldConfig())

    def _make_llmdet() -> DetectionModel:
        from sparx_agency.core.mapping.detection.llmdet import LlmDetDetector

        return LlmDetDetector(llmdet_config)

    def _make_grounding_dino() -> DetectionModel:
        from sparx_agency.core.mapping.detection.grounding_dino import GroundingDinoDetector

        return GroundingDinoDetector(grounding_dino_config)

    def _make_grounded_vlm() -> DetectionModel:
        from sparx_agency.core.mapping.detection.grounded_vlm import GroundedVlmConfig, GroundedVlmDetector

        config = grounded_vlm_config or GroundedVlmConfig()
        if config.yolo is not None:
            raise ValueError("grounded_vlm must not include YOLO; select hybrid instead")
        return GroundedVlmDetector(config)

    def _make_hybrid() -> DetectionModel:
        from sparx_agency.core.mapping.detection.grounded_vlm import GroundedVlmConfig, GroundedVlmDetector
        from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig

        config = hybrid_config or GroundedVlmConfig(yolo=YoloWorldConfig(device="cpu"))
        if config.yolo is None:
            raise ValueError("hybrid requires an explicit YOLO configuration")
        return GroundedVlmDetector(config)

    reg.register(DetectorFactory(name="yolo_world", create=_make_yolo_world))
    reg.register(DetectorFactory(name="llmdet", create=_make_llmdet))
    reg.register(DetectorFactory(name="grounding_dino", create=_make_grounding_dino))
    reg.register(DetectorFactory(name="grounded_vlm", create=_make_grounded_vlm))
    reg.register(DetectorFactory(name="hybrid", create=_make_hybrid))
    return reg
