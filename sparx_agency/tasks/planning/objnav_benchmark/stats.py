"""Confidence intervals and paired tests for benchmark results, with the standard library alone.

A benchmark number without its uncertainty invites the wrong conclusion: at
500 episodes a success rate carries a 95% interval about +/-4 points wide, so
a two-point gap to a paper is noise, and so may be a two-point gain from a
change. These are the few tests the harness needs, written out so they run in
any Python a simulator ships with (no scipy, no numpy):

* :func:`wilson_interval` for a success rate. The Wald interval
  ``p +/- z * sqrt(p (1 - p) / n)`` collapses to zero width at 0% and 100%
  and spills outside ``[0, 1]`` near them; Wilson does neither.
* :func:`bootstrap_mean_interval` for a mean score such as SPL, whose
  per-episode values are bounded, piled up at 0, and nowhere near normal.
* :func:`mcnemar_exact_p` and :func:`paired_permutation_p` for two agents on
  the same episodes: paired, because the episodes carry most of the variance,
  and exact wherever exactness is cheap.

Every random draw is seeded from a string through ``zlib.crc32``, never
``hash()`` (salted per process for strings), and only ``Random.random()`` is
consumed -- the one method whose stream Python promises to keep across
versions -- so an interval printed today is reproduced bit for bit by another
interpreter later.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import itertools
import math
import random
import zlib
from fractions import Fraction
from statistics import NormalDist
from typing import List, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_real,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError


def _check_confidence(confidence) -> float:
    if not (is_real(confidence) and 0.0 < confidence < 1.0):
        raise HarnessError(
            "confidence must lie strictly between 0 and 1 (0.95 for a 95%% "
            "interval), got %r" % (confidence,))
    return float(confidence)


def _check_int(name: str, value, minimum: int) -> int:
    """``value`` as an int, once it is a count (never a bool) of at least ``minimum`` >= 0."""
    if not (is_count(value) and value >= minimum):
        raise HarnessError(
            "%s must be an integer >= %d, got %r" % (name, minimum, value))
    return int(value)


def _finite_values(name: str, values: Sequence[float]) -> List[float]:
    data = list(values)
    if not data:
        raise HarnessError("%s is empty; a statistic needs values" % name)
    for index, value in enumerate(data):
        if not (is_real(value) and math.isfinite(value)):
            raise HarnessError(
                "%s[%d] must be a finite number, got %r" % (name, index, value))
    return [float(value) for value in data]


def _bools(name: str, values: Sequence[bool]) -> List[bool]:
    data = list(values)
    if not data:
        raise HarnessError("%s is empty; a statistic needs values" % name)
    for index, value in enumerate(data):
        if not isinstance(value, bool):
            raise HarnessError(
                "%s[%d] must be a bool, got %r" % (name, index, value))
    return data


def _check_paired(first: Sequence, second: Sequence) -> None:
    if len(first) != len(second):
        raise HarnessError(
            "a paired test needs one value per episode on each side; first "
            "has %d values and second has %d" % (len(first), len(second)))


def _seeded_rng(seed_key: str) -> random.Random:
    """A generator seeded by ``crc32(seed_key)``: stable across processes, unlike ``hash()``."""
    if not isinstance(seed_key, str):
        raise HarnessError("seed_key must be a string, got %r" % (seed_key,))
    return random.Random(zlib.crc32(seed_key.encode("utf-8")))


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    """The ``q`` quantile of sorted values, interpolated like numpy's default method."""
    position = (len(ordered) - 1) * q
    below = int(math.floor(position))
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def _exact_sign_flip_p(first: Sequence[float], second: Sequence[float]) -> float:
    """The sign-flip p-value by enumerating every pattern, on exact differences.

    Each ``x - y`` is taken exactly (``Fraction``) and scaled by the common
    denominator to an integer, so every pattern's sum is an exact integer and
    a tie is a tie. Integers rather than Fractions because the enumeration
    sums up to ``2 ** n`` patterns, and integer sums are about twenty times
    faster.
    """
    exact = [Fraction(x) - Fraction(y) for x, y in zip(first, second)]
    scale = 1
    for difference in exact:
        scale = scale * difference.denominator // math.gcd(
            scale, difference.denominator)
    diffs = [int(difference * scale) for difference in exact]
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(diffs))
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed)
    return hits / 2 ** len(diffs)


def wilson_interval(successes: int, trials: int,
                    confidence: float = 0.95) -> Tuple[float, float]:
    """The Wilson score interval of a success rate.

    Wilson (1927). It stays inside ``[0, 1]`` and keeps a non-zero width at 0
    and at ``trials`` successes, where the Wald interval collapses; its end
    points are exact there.

    Args:
        successes: Episodes that succeeded.
        trials: Episodes run; at least one.
        confidence: Coverage, in ``(0, 1)``.

    Returns:
        ``(lower, upper)`` bounds of the success rate.

    Raises:
        HarnessError: On ``trials < 1``, ``successes`` outside
            ``[0, trials]``, a non-integer count, or a confidence outside
            ``(0, 1)``.
    """
    n = _check_int("trials", trials, 1)
    k = _check_int("successes", successes, 0)
    if k > n:
        raise HarnessError(
            "successes (%d) cannot exceed trials (%d)" % (k, n))
    z = NormalDist().inv_cdf(0.5 + _check_confidence(confidence) / 2.0)
    rate = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (rate + z2 / (2.0 * n)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / n + z2 / (4.0 * n * n)) / denominator
    lower = 0.0 if k == 0 else centre - half
    upper = 1.0 if k == n else centre + half
    return (lower, upper)


def bootstrap_mean_interval(values: Sequence[float], confidence: float = 0.95,
                            resamples: int = 2000,
                            seed_key: str = "objnav") -> Tuple[float, float]:
    """The percentile-bootstrap interval of a mean.

    Resamples the values with replacement ``resamples`` times, takes each
    resample's mean, and reads the interval off the ``(1 - c) / 2`` and
    ``(1 + c) / 2`` quantiles of those means, interpolated linearly between
    order statistics (numpy's default method). Each draw is
    ``floor(random() * n)`` from ``random.Random(crc32(seed_key))``, so the
    same values and key give the same interval on every Python version.

    Args:
        values: One value per episode, e.g. SPL; at least one.
        confidence: Coverage, in ``(0, 1)``.
        resamples: Bootstrap resamples; at least one.
        seed_key: Names the random stream; give each group its own.

    Returns:
        ``(lower, upper)``; ``(v, v)`` for a single value ``v``.

    Raises:
        HarnessError: On no values, a non-finite value, a confidence outside
            ``(0, 1)``, fewer than one resample, or a non-string key.
    """
    data = _finite_values("values", values)
    level = _check_confidence(confidence)
    count = _check_int("resamples", resamples, 1)
    rng = _seeded_rng(seed_key)
    n = len(data)
    if n == 1:
        return (data[0], data[0])
    draw = rng.random
    means = sorted(math.fsum(data[int(draw() * n)] for _ in range(n)) / n
                   for _ in range(count))
    tail = (1.0 - level) / 2.0
    return (_linear_quantile(means, tail), _linear_quantile(means, 1.0 - tail))


def mcnemar_exact_p(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Two-sided exact McNemar test: do two agents succeed on different episodes?

    Only the discordant episodes (one agent succeeded, the other did not)
    carry information. Under the null each is equally likely to favour
    either agent, so the smaller count is a Binomial(``b + c``, 1/2) draw;
    the p-value doubles its tail, capped at 1, computed with exact integers.

    Args:
        first: The first agent's success per episode.
        second: The second agent's, on the same episodes in the same order.

    Returns:
        The p-value; 1 when no episode is discordant.

    Raises:
        HarnessError: On empty or unequal-length input, or a non-bool entry.
    """
    a = _bools("first", first)
    b = _bools("second", second)
    _check_paired(a, b)
    first_only = sum(1 for x, y in zip(a, b) if x and not y)
    second_only = sum(1 for x, y in zip(a, b) if y and not x)
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i)
               for i in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired_permutation_p(first: Sequence[float], second: Sequence[float],
                         permutations: int = 10000,
                         seed_key: str = "objnav") -> float:
    """Two-sided paired sign-flip test on the mean difference of two agents' scores.

    Under the null each episode's difference is as likely to have either
    sign, so the observed ``|mean difference|`` is compared with those of
    sign-flipped copies. Every one of the ``2 ** n`` sign patterns is
    enumerated when there are no more than ``permutations`` of them;
    otherwise ``permutations`` random patterns are drawn and the p-value is
    ``(hits + 1) / (permutations + 1)``, which never reports an impossible 0.
    The enumeration is exact: every difference ``x - y`` and every pattern's
    sum is computed without rounding, so a pattern that ties the observed
    statistic exactly is a hit, and one that falls short of it by a rounding
    error is not. The Monte-Carlo branch sums with ``math.fsum``, so there a
    tie is decided on correctly rounded float sums: on scores that are not
    binary fractions (thirds, tenths) a pattern within rounding of the
    observed statistic may count either way.

    Args:
        first: The first agent's score per episode.
        second: The second agent's, on the same episodes in the same order.
        permutations: The enumeration limit and the Monte-Carlo sample size.
        seed_key: Names the Monte-Carlo stream.

    Returns:
        The p-value; 1 when every difference is zero.

    Raises:
        HarnessError: On empty or unequal-length input, a non-finite value,
            fewer than one permutation, or a non-string key.
    """
    a = _finite_values("first", first)
    b = _finite_values("second", second)
    _check_paired(a, b)
    count = _check_int("permutations", permutations, 1)
    rng = _seeded_rng(seed_key)
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0.0 for d in diffs):
        return 1.0
    patterns = 2 ** len(diffs)
    if patterns <= count:
        return _exact_sign_flip_p(a, b)
    observed = abs(math.fsum(diffs))
    draw = rng.random
    hits = 0
    for _ in range(count):
        if abs(math.fsum(d if draw() < 0.5 else -d for d in diffs)) >= observed:
            hits += 1
    return (hits + 1) / (count + 1)
