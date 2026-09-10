"""The small predicates every harness type validates with, defined once.

A record, a summary and a score each check that a rate is a fraction, a
length is finite and a count is a count; three private copies of those checks
can drift apart, and then one type accepts what another refuses. Each
predicate also refuses the look-alike a naive check lets through: ``True`` is
an ``int`` and a ``numbers.Real`` in Python, so a bool would pass as a count
or a score unless excluded by name; NaN fails every comparison, so a check
written as ``not value < 0`` would let it through where ``value >= 0`` does
not; and a string of spaces is non-empty.

They only answer. The caller raises its own error, naming the field and what
to do instead, so the message says which value was wrong.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import numbers


def is_real(value) -> bool:
    """Whether ``value`` is a real number that is not a bool.

    NaN and the infinities count as real numbers here; the predicates below
    exclude them where a value must be finite or bounded.

    Args:
        value: Anything.

    Returns:
        True for an ``int``, a ``float`` or any other :class:`numbers.Real`
        (numpy's scalars included); False for a bool, a string or None.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_fraction(value) -> bool:
    """Whether ``value`` is a real number in ``[0, 1]``: a rate, a score, a p-value.

    Args:
        value: Anything.

    Returns:
        True for a real number from 0 to 1 inclusive; False for NaN, for a
        percent such as 46.2, and for anything :func:`is_real` refuses.
    """
    return is_real(value) and 0.0 <= value <= 1.0


def is_length(value) -> bool:
    """Whether ``value`` is a finite, non-negative real number: metres, seconds, a mean of either.

    Args:
        value: Anything.

    Returns:
        True for a finite real number at or above 0; False for a negative
        number, NaN, an infinity, and anything :func:`is_real` refuses.
    """
    return is_real(value) and math.isfinite(value) and value >= 0.0


def is_count(value) -> bool:
    """Whether ``value`` is a non-negative integer that is not a bool: steps, episodes, tallies.

    A whole float such as ``3.0`` is not a count: it means a mean or a ratio
    was stored where a tally belongs.

    Args:
        value: Anything.

    Returns:
        True for an :class:`numbers.Integral` at or above 0 (numpy's integer
        scalars included); False for a bool, a float and a negative integer.
    """
    return (isinstance(value, numbers.Integral) and not isinstance(value, bool)
            and value >= 0)


def is_name(value) -> bool:
    """Whether ``value`` is a string with something besides whitespace in it: an identifier, a key, a source.

    Args:
        value: Anything.

    Returns:
        True for a string holding at least one non-whitespace character;
        False for ``""``, ``"   "`` and anything that is not a string.
    """
    return isinstance(value, str) and bool(value.strip())
