"""The number checks the converter's geometry refuses bad input with.

Both halves of that geometry -- :mod:`.path_geometry` and :mod:`.action_choice`
-- take plain floats from a policy's command and a simulator's pose. A NaN
there compares false with everything, so it would pick an action silently
instead of failing, and a ``bool`` would pass as the number 1. These three
checks are shared, not copied, so both halves refuse exactly the same inputs
with the same message.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import numbers

from sparx_agency.core.planning.objnav.errors import ObjNavError


def is_finite_real(value) -> bool:
    """Whether ``value`` is a finite real number (a bool is not one).

    Args:
        value: Anything.

    Returns:
        True for a finite int, float or numpy scalar; False for a bool, a NaN,
        an infinity or anything that is not a real number.
    """
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value))


def require_finite(name: str, value) -> None:
    """Raise unless ``value`` is a finite real number.

    Args:
        name: The argument's name, for the message.
        value: The value to check.

    Raises:
        ObjNavError: If ``value`` is not a finite real number.
    """
    if not is_finite_real(value):
        raise ObjNavError("%s must be a finite number, got %r" % (name, value))


def require_positive(name: str, value) -> None:
    """Raise unless ``value`` is a positive finite real number.

    Args:
        name: The argument's name, for the message.
        value: The value to check.

    Raises:
        ObjNavError: If ``value`` is not a positive finite real number.
    """
    if not (is_finite_real(value) and value > 0):
        raise ObjNavError("%s must be a positive finite number, got %r"
                          % (name, value))
