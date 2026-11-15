"""What a caller of the ObjectNav layer can get wrong, named so the message says which.

Every contract failure here is raised at the boundary where it is detected --
an observation that disagrees with its camera, a category no label mapper
covers, a command the embodiment cannot execute -- instead of being degraded
into a plausible action. On a benchmark a silently wrong action is worse than
a crash: it becomes a failed episode that looks like the method's fault, and it
moves SR and SPL without anyone knowing why.

:class:`ObjNavInternalError` sits on a different base class on purpose. It
means *our* invariant broke, never the caller's input, and a caller who writes
``except ObjNavError`` must not swallow it.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations


class ObjNavError(ValueError):
    """Base for every input-contract failure in the ObjectNav layer.

    A :class:`ValueError`, so a caller may catch the family broadly without
    also catching a bug in this package.
    """


class ObservationError(ObjNavError):
    """An observation breaks its contract.

    Its images disagree with the camera it claims, its target is not the
    episode's, or its step index did not advance.
    """


class UnknownCategoryError(ObjNavError):
    """A dataset category that the label mapper does not cover.

    Raised at episode reset, before the first step, so a missing table entry
    fails the run instead of silently searching for the wrong object.
    """


class CommandError(ObjNavError):
    """A navigation command that is malformed or that the embodiment cannot execute.

    A command that asks to stop and to follow waypoints at once, a non-finite
    waypoint, or a camera-pitch request on a benchmark with no LOOK actions.
    """


class EnvContractError(ObjNavError):
    """An environment broke the :class:`ObjNavEnv` contract.

    Stepping after the episode ended, running past the step budget, reporting
    a step count that disagrees with the actions it was sent, or measuring an
    episode that is still running. Always a bug in an adapter, and always one
    that would otherwise corrupt a score without failing anything.
    """


class AgentContractError(ObjNavError):
    """An agent broke the :class:`ObjNavAgent` contract.

    It returned something that is not an ``AgentDecision``, or an action the
    benchmark does not accept. The runner never passes such an action to the
    environment: it raises, or -- when told to record agent errors -- logs the
    episode as an agent error.
    """


class ObjNavInternalError(RuntimeError):
    """An invariant of this package broke. Our bug, never the caller's input."""
