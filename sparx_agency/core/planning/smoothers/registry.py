"""
Smoother registry for name-based construction.

Allows algorithm selection by string key, enabling configuration-driven pipelines.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from sparx_agency.core.planning.interfaces.smoother import BaseSmoother


class SmootherRegistry:
    """
    Registry for smoother factory functions.

    Example:
        >>> SmootherRegistry.register("hermite", lambda **kw: HermiteSmoother(**kw))
        >>> smoother = SmootherRegistry.create("hermite", params=HermiteParams())
    """
    _factories: Dict[str, Callable[..., BaseSmoother]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseSmoother]) -> None:
        """
        Register a smoother factory.

        Args:
            name: Unique key (case-insensitive).
            factory: Callable returning BaseSmoother instance.
        """
        key = name.strip().lower()
        if not key:
            raise ValueError("Smoother name cannot be empty")
        cls._factories[key] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseSmoother:
        """
        Construct smoother by name.

        Args:
            name: Registered smoother key.
            **kwargs: Passed to factory.

        Raises:
            KeyError: If name not registered.
        """
        key = name.strip().lower()
        if key not in cls._factories:
            known = ", ".join(sorted(cls._factories.keys()))
            raise KeyError(f"Unknown smoother '{name}'. Registered: [{known}]")
        return cls._factories[key](**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """Return registered smoother names."""
        return sorted(cls._factories.keys())