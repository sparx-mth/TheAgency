"""The geometry's number checks accept every real number an adapter hands over and nothing else.

A NaN compares false with everything, so it would pick an action silently; a
``bool`` would pass as the number 1; and a numpy scalar is how every pose read
from an array arrives. Both halves of the converter's geometry refuse input
with these checks, so each is pinned here.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.action_converter.checks import (
    is_finite_real,
    require_finite,
    require_positive,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError


@pytest.mark.parametrize("value", [0, 1.5, -2, np.float32(2.0), np.int64(3)])
def test_finite_real_numbers_numpy_scalars_included_pass(value):
    """Poses and waypoints read from arrays are numpy scalars; refusing them refuses every adapter."""
    assert is_finite_real(value)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"),
                                   -float("inf"), "1", None, [1.0]])
def test_bools_nans_infinities_and_non_numbers_fail(value):
    """A NaN compares false with everything, and ``True`` would pass as 1."""
    assert not is_finite_real(value)


def test_require_finite_names_the_argument_it_refuses():
    """The message must say which of five numbers was the NaN."""
    with pytest.raises(ObjNavError, match="from_arc_m"):
        require_finite("from_arc_m", float("nan"))
    require_finite("x", -1.0)


@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), True])
def test_require_positive_refuses_zero_negatives_and_non_finite(value):
    """A zero turn or tilt would make every error look outside the dead band forever."""
    with pytest.raises(ObjNavError, match="tilt_rad"):
        require_positive("tilt_rad", value)
    require_positive("tilt_rad", 0.5)
