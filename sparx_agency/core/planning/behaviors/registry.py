"""
Behavior registry for lookup and management.

This module provides a simple registry pattern for accessing behaviors
by name. The coordinator uses the registry to instantiate and switch
between behaviors dynamically.

Usage:
    >>> registry = BehaviorRegistry.from_behaviors([
    ...     GoToPoseBehavior(),
    ...     ExploreRoomBehavior(),
    ...     WallFollowBehavior(),
    ... ])
    >>> behavior = registry.get("go_to_pose")

See Also:
    - Behavior: Protocol that registered behaviors must implement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from .interfaces.behavior import Behavior


@dataclass
class BehaviorRegistry:
    """
    Registry for looking up behaviors by name.

    Provides a centralized lookup mechanism for behaviors. The coordinator
    can use this to dynamically select behaviors based on task requirements.

    Attributes:
        _by_name: Internal dictionary mapping behavior names to instances.

    Example:
        >>> # Create registry from behavior instances
        >>> registry = BehaviorRegistry.from_behaviors([
        ...     GoToPoseBehavior(),
        ...     ExploreRoomBehavior(),
        ...     EnterPortalBehavior(),
        ... ])
        >>>
        >>> # Look up behavior by name
        >>> behavior = registry.get("explore_room")
        >>> behavior.reset()
        >>> output = behavior.step(ctx, planner=planner)

    Note:
        Behaviors are stored by instance, not class. If you need fresh
        instances, create a new registry or implement a factory pattern.
    """

    _by_name: Dict[str, Behavior] = field(default_factory=dict)

    def get(self, name: str) -> Behavior:
        """
        Retrieve a behavior by name.

        Args:
            name: The behavior's name attribute (e.g., "go_to_pose").

        Returns:
            The registered Behavior instance.

        Raises:
            KeyError: If no behavior with the given name is registered.
        """
        return self._by_name[name]

    def get_optional(self, name: str) -> Optional[Behavior]:
        """
        Retrieve a behavior by name, returning None if not found.

        Args:
            name: The behavior's name attribute.

        Returns:
            The registered Behavior instance, or None if not found.
        """
        return self._by_name.get(name)

    def register(self, behavior: Behavior) -> None:
        """
        Register a behavior instance.

        Args:
            behavior: Behavior instance to register. Must have a `name` attribute.

        Raises:
            ValueError: If a behavior with the same name is already registered.
        """
        if behavior.name in self._by_name:
            raise ValueError(f"Behavior '{behavior.name}' is already registered")
        self._by_name[behavior.name] = behavior

    def names(self) -> Iterable[str]:
        """
        Get all registered behavior names.

        Returns:
            Iterable of behavior name strings.
        """
        return self._by_name.keys()

    def __contains__(self, name: str) -> bool:
        """Check if a behavior name is registered."""
        return name in self._by_name

    def __len__(self) -> int:
        """Return the number of registered behaviors."""
        return len(self._by_name)

    @classmethod
    def from_behaviors(cls, behaviors: Iterable[Behavior]) -> "BehaviorRegistry":
        """
        Create a registry from an iterable of behavior instances.

        Args:
            behaviors: Iterable of Behavior instances to register.

        Returns:
            New BehaviorRegistry containing all provided behaviors.

        Example:
            >>> registry = BehaviorRegistry.from_behaviors([
            ...     GoToPoseBehavior(),
            ...     ExploreRoomBehavior(),
            ... ])
        """
        registry = cls()
        for behavior in behaviors:
            registry.register(behavior)
        return registry