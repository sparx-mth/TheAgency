"""Registry of open-vocabulary detector backends (factory idiom).

Mirrors :mod:`sparx_agency.core.planning.trackers.registry`. Lets a task node
select a detector by name (``"yolo_world"`` today; a TensorRT runtime or NanoOWL
backend later) without importing the heavy backend module until it is created.
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


def default_detection_registry() -> DetectionRegistry:
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

        return YoloWorldDetector(YoloWorldConfig())

    reg.register(DetectorFactory(name="yolo_world", create=_make_yolo_world))
    return reg
