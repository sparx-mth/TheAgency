"""Name -> VLA policy factory.

Copies the ``Factory`` + ``Registry`` + ``default_*_registry()`` idiom already
used by ``core/planning/trackers/registry.py``,
``core/mapping/detection/registry.py`` and ``core/mapping/tracking/registry.py``
-- and specifically the detection registry's habit of doing the heavy import
*inside* the factory closure. That is not a style preference here: a policy's
constructor may pull ``requests``, TensorRT or an external model repo, and
importing this module must stay free (see the package docstring).

Usage::

    registry = default_vla_registry()
    registry.names()                     # ['flownav', 'navdp']
    policy = registry.create("navdp", url="http://127.0.0.1:8888")

Adding a policy is one entry here plus one adapter module; see
``tasks/planning/vlas/README.md`` for the full checklist.

Python 3.8 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class VlaFactory:
    """A named, lazily-constructed VLA policy.

    Attributes:
        name: registry key, e.g. ``"navdp"``.
        create: ``(**kwargs) -> NavigationPolicy``. Must do its heavy imports
            inside the callable, not at module scope.
        goal_modality: short human-readable note on what this policy is told to
            go to (``"point"``, ``"image"``, ``"language"``), so a caller can
            pick one without constructing it.
    """
    name: str
    create: Callable[..., Any]
    goal_modality: str = ""


class VlaRegistry:
    """Look up VLA policies by name."""

    def __init__(self):
        self._factories = {}  # type: Dict[str, VlaFactory]

    def register(self, factory):
        """Add ``factory``.

        Raises:
            ValueError: a policy with that name is already registered. Silently
                overwriting would make which implementation flies depend on
                import order.
        """
        if factory.name in self._factories:
            raise ValueError("VLA %r is already registered" % factory.name)
        self._factories[factory.name] = factory

    def names(self):
        # type: () -> List[str]
        """Registered policy names, sorted."""
        return sorted(self._factories)

    def get(self, name):
        """Return the :class:`VlaFactory` registered under ``name``.

        Raises:
            KeyError: no such policy, listing what is available.
        """
        if name not in self._factories:
            raise KeyError("Unknown VLA %r. Available: %s"
                           % (name, ", ".join(self.names())))
        return self._factories[name]

    def create(self, name, **kwargs):
        """Construct the policy registered under ``name``.

        Args:
            name: registry key.
            **kwargs: forwarded to the factory (``url``, ``timeout_s``, ...).

        Raises:
            KeyError: no such policy.
        """
        return self.get(name).create(**kwargs)


def _navdp(**kwargs):
    """Build the NavDP point-goal policy (import kept inside the closure)."""
    from sparx_agency.core.planning.vlas.navdp.policy import NavDPPolicy
    return NavDPPolicy(**kwargs)


def _flownav(**kwargs):
    """Build the FlowNav image-goal policy (import kept inside the closure)."""
    from sparx_agency.core.planning.vlas.flownav.policy import FlowNavPolicy
    return FlowNavPolicy(**kwargs)


def _internvla_n1(**kwargs):
    """Build the InternVLA-N1 language-goal policy (import kept in the closure)."""
    from sparx_agency.core.planning.vlas.internvla_n1.policy import InternVLAN1Policy
    return InternVLAN1Policy(**kwargs)


def default_vla_registry():
    """The policies that have a :class:`NavigationPolicy` adapter today.

    NavDP, FlowNav and InternVLA-N1 are here because they are driven through this
    interface. OmniVLA and NoMaD ship their ROS-free contract under
    ``core/planning/vlas/`` but no adapter yet -- deliberately, since nothing
    drives them that way. Adding one is ~40 lines plus an entry here; it is not
    a redesign.
    """
    registry = VlaRegistry()
    registry.register(VlaFactory(name="navdp", create=_navdp, goal_modality="point"))
    registry.register(VlaFactory(name="flownav", create=_flownav, goal_modality="image"))
    registry.register(VlaFactory(name="internvla_n1", create=_internvla_n1,
                                 goal_modality="language"))
    return registry
