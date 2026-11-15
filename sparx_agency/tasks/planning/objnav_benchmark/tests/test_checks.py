"""The harness's shared predicates, pinned on the look-alikes they exist to refuse.

A bool is an int in Python, NaN fails every comparison, and a string of spaces
is non-empty. A predicate that lets one of them through lets it into every
record, summary and score at once, because they all validate with these.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_fraction,
    is_length,
    is_name,
    is_real,
)

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("value,expected", [
    (0, True), (-3, True), (2.5, True), (INF, True), (NAN, True),
    (np.float32(0.5), True), (np.int64(3), True),
    (True, False), (False, False), ("1", False), (None, False),
    (np.bool_(True), False), (1j, False)])
def test_a_real_number_is_any_real_but_never_a_bool(value, expected):
    """``True`` would otherwise pass as 1 wherever a number is expected."""
    assert is_real(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.0, True), (1, True), (0.5, True), (np.float64(0.25), True),
    (1.0000001, False), (-1e-9, False), (46.2, False), (NAN, False),
    (INF, False), (True, False), ("0.5", False), (None, False)])
def test_a_fraction_lies_in_the_closed_unit_interval_and_is_never_nan(value, expected):
    """46.2 typed for 0.462 is the usual slip, and NaN would slip past a check written the wrong way round."""
    assert is_fraction(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0, True), (3.25, True), (np.float64(2.0), True),
    (-0.5, False), (NAN, False), (INF, False), (-INF, False), (True, False),
    ("1", False), (None, False)])
def test_a_length_is_finite_and_non_negative(value, expected):
    """An infinite or negative length is an adapter bug that would still average into a number."""
    assert is_length(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0, True), (7, True), (np.int64(3), True),
    (-1, False), (3.0, False), (True, False), (False, False), ("3", False),
    (None, False)])
def test_a_count_is_a_non_negative_integer_and_never_a_bool_or_a_float(value, expected):
    """A whole float in a count means a mean was stored where a tally belongs."""
    assert is_count(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("chair", True), (" a ", True),
    ("", False), ("   ", False), ("\t\n", False), (None, False), (5, False),
    (b"chair", False)])
def test_a_name_has_something_besides_whitespace(value, expected):
    """A blank identifier prints as an empty cell and groups episodes under nothing."""
    assert is_name(value) == expected
