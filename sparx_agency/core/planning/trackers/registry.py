from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from sparx_agency.core.planning.trackers.pure_pursuit.params import PurePursuitParams
from sparx_agency.core.planning.trackers.pure_pursuit.tracker import PurePursuitTracker


@dataclass(frozen=True, slots=True)
class TrackerFactory:
    name: str
    create: Callable[[], object]


class TrackerRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, TrackerFactory] = {}

    def register(self, factory: TrackerFactory) -> None:
        if factory.name in self._factories:
            raise ValueError(f"Tracker '{factory.name}' already registered")
        self._factories[factory.name] = factory

    def names(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> object:
        if name not in self._factories:
            raise KeyError(f"Unknown tracker '{name}'. Available: {self.names()}")
        return self._factories[name].create()


def default_tracker_registry() -> TrackerRegistry:
    reg = TrackerRegistry()
    reg.register(
        TrackerFactory(
            name="pure_pursuit",
            create=lambda: PurePursuitTracker(params=PurePursuitParams()),
        )
    )
    return reg
