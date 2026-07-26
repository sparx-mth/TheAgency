"""Registry of classic box-tracker backends (factory idiom).

Lets a task node pick the propagation backend by name — ``"lucas_kanade"`` today,
a correlation/DNN tracker later — and inject it into a
:class:`~sparx_agency.core.mapping.tracking.target_tracker.TargetTracker`
without changing the servo or FSM above it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from sparx_agency.core.mapping.tracking.interface import BoxTracker
from sparx_agency.core.mapping.tracking.lk_box_tracker import (
    LucasKanadeBoxTracker,
    LKBoxTrackerConfig,
)
from sparx_agency.core.mapping.tracking.median_flow_box_tracker import (
    MedianFlowBoxTracker,
    MedianFlowConfig,
)


@dataclass(frozen=True)
class BoxTrackerFactory:
    """A named zero-arg factory producing a :class:`BoxTracker`."""

    name: str
    create: Callable[[], BoxTracker]


class BoxTrackerRegistry:
    """Name -> box-tracker-factory map."""

    def __init__(self) -> None:
        self._factories: Dict[str, BoxTrackerFactory] = {}

    def register(self, factory: BoxTrackerFactory) -> None:
        if factory.name in self._factories:
            raise ValueError("Box tracker '%s' already registered" % factory.name)
        self._factories[factory.name] = factory

    def names(self) -> List[str]:
        return sorted(self._factories.keys())

    def create(self, name: str) -> BoxTracker:
        if name not in self._factories:
            raise KeyError(
                "Unknown box tracker '%s'. Available: %s" % (name, self.names())
            )
        return self._factories[name].create()


def default_box_tracker_registry() -> BoxTrackerRegistry:
    """Registry with the built-in trackers registered (robust default first)."""
    reg = BoxTrackerRegistry()
    reg.register(
        BoxTrackerFactory(
            name="median_flow",
            create=lambda: MedianFlowBoxTracker(MedianFlowConfig()),
        )
    )
    reg.register(
        BoxTrackerFactory(
            name="lucas_kanade",
            create=lambda: LucasKanadeBoxTracker(LKBoxTrackerConfig()),
        )
    )
    return reg
