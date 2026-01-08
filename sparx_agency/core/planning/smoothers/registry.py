"""
Smoother registry.

Higher-level code (apps/tests/experiments) can choose a smoother by name and
construct it via this registry. This keeps orchestration code stable while
allowing algorithm replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sparx_agency.core.planning.interfaces.smoother import BaseSmoother


@dataclass(frozen=True)
class RegistryEntry:
    """A registered smoother factory entry."""
    name: str
    factory: Callable[..., BaseSmoother]


class SmootherRegistry:
    """
    Register and construct smoothers by string key.

    Typical usage:
        SmootherRegistry.register("minsnap", lambda **kw: MinSnapSmoother(**kw))
        smoother = SmootherRegistry.create("minsnap", params=MinSnapParams(...))
    """
    _entries: Dict[str, RegistryEntry] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseSmoother]) -> None:
        """
        Register a smoother factory.

        Args:
            name: unique key (case-insensitive)
            factory: callable that returns a BaseSmoother instance
        """
        key = name.strip().lower()
        if not key:
            raise ValueError("Smoother name cannot be empty")
        cls._entries[key] = RegistryEntry(name=key, factory=factory)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseSmoother:
        """
        Construct a smoother instance by name.

        Raises:
            KeyError if the name is unknown.
        """
        key = name.strip().lower()
        if key not in cls._entries:
            known = ", ".join(sorted(cls._entries.keys()))
            raise KeyError(f"Unknown smoother '{name}'. Known: [{known}]")
        return cls._entries[key].factory(**kwargs)

    @classmethod
    def known(cls) -> list[str]:
        """Return registered smoother keys."""
        return sorted(cls._entries.keys())
