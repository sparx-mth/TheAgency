"""The intervals and tests behind every number the harness prints.

Known values come from closed forms (Wilson, the binomial tail) and from
numpy's own quantile, so a regression shows here rather than in a paper table.
"""
from __future__ import annotations

import itertools
import os
import random
import subprocess
import sys
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.stats import (
    bootstrap_mean_interval,
    mcnemar_exact_p,
    paired_permutation_p,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
NAN = float("nan")
Z95_SQUARED = 1.959963984540054 ** 2


def run_python(code):
    """Run ``code`` in a fresh interpreter at the repo root with another hash seed."""
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          env=env, capture_output=True, text=True)


def spl_like(n, seed):
    """Per-episode SPLs as a benchmark makes them: many zeros, the rest spread over (0, 1)."""
    rng = random.Random(seed)
    return [rng.random() if rng.random() < 0.6 else 0.0 for _ in range(n)]


def pairs(first_only, second_only, both=0, neither=0):
    """Paired successes with the given discordant and concordant counts."""
    first = ([True] * first_only + [False] * second_only + [True] * both
             + [False] * neither)
    second = ([False] * first_only + [True] * second_only + [True] * both
              + [False] * neither)
    return first, second


def exact_sign_flip_p(first, second):
    """The enumeration's p-value over the exact binary inputs, computed independently with Fractions."""
    diffs = [Fraction(x) - Fraction(y) for x, y in zip(first, second)]
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(diffs))
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed)
    return hits / 2 ** len(diffs)


# -- Wilson ----------------------------------------------------------------

def test_wilson_matches_the_textbook_value_for_eight_of_ten():
    """8/10 at 95% is the textbook Wilson example; the closed form must reproduce it."""
    lower, upper = wilson_interval(8, 10)
    assert lower == pytest.approx(0.4902, abs=5e-4)
    assert upper == pytest.approx(0.9433, abs=5e-4)


def test_wilson_keeps_a_nonzero_width_at_zero_and_all_successes():
    """Wald collapses to [0, 0] at 0/10; Wilson's upper end is z^2 / (n + z^2)."""
    assert wilson_interval(0, 10) == pytest.approx(
        (0.0, Z95_SQUARED / (10 + Z95_SQUARED)), abs=1e-12)
    assert wilson_interval(10, 10) == pytest.approx(
        (10 / (10 + Z95_SQUARED), 1.0), abs=1e-12)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_wilson_widens_with_confidence_and_narrows_with_trials():
    """The two monotonicities a reader relies on when reading an interval."""
    def width(interval):
        return interval[1] - interval[0]
    assert width(wilson_interval(46, 100, 0.99)) > width(wilson_interval(46, 100))
    assert width(wilson_interval(460, 1000)) < width(wilson_interval(46, 100))


@pytest.mark.parametrize("successes,trials", [
    (0, 0), (11, 10), (-1, 10), (True, 10), (5, 10.0), (5.0, 10)])
def test_wilson_refuses_impossible_counts(successes, trials):
    """A count that cannot be a count means the caller passed the wrong field."""
    with pytest.raises(HarnessError):
        wilson_interval(successes, trials)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, NAN, "0.95", True])
def test_every_interval_refuses_a_confidence_outside_the_open_unit_interval(
        confidence):
    """95 typed for 0.95 would otherwise produce a meaningless interval."""
    with pytest.raises(HarnessError, match="confidence"):
        wilson_interval(5, 10, confidence)
    with pytest.raises(HarnessError, match="confidence"):
        bootstrap_mean_interval([0.1, 0.2], confidence)


# -- bootstrap -------------------------------------------------------------

def test_bootstrap_is_identical_for_the_same_seed_key_and_differs_for_another():
    """A printed interval must reproduce exactly; different groups must not share draws."""
    values = spl_like(50, seed=1)
    first = bootstrap_mean_interval(values, seed_key="hm3d/val/a/all")
    assert bootstrap_mean_interval(values, seed_key="hm3d/val/a/all") == first
    assert bootstrap_mean_interval(values, seed_key="hm3d/val/a/bed") != first


def test_a_fresh_interpreter_with_another_hash_seed_reproduces_the_bootstrap():
    """Seeds come from crc32, not hash(); a salted string hash would move every interval."""
    values = [0.0, 0.3, 1.0, 0.8, 0.1, 0.0, 0.65]
    code = ("from sparx_agency.tasks.planning.objnav_benchmark.stats "
            "import bootstrap_mean_interval as b; "
            "print(repr(b(%r, resamples=300, seed_key='hm3d/val/a/all')))"
            % (values,))
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == repr(bootstrap_mean_interval(
        values, resamples=300, seed_key="hm3d/val/a/all"))


def test_bootstrap_draws_are_pinned_so_published_intervals_reproduce():
    """The draw scheme is part of every result: crc32 seed, one random() per draw, numpy's quantile."""
    values = [0.0, 0.2, 0.9, 1.0, 0.5, 0.0, 0.7]
    rng = random.Random(zlib.crc32(b"pin"))
    n = len(values)
    means = [sum(values[int(rng.random() * n)] for _ in range(n)) / n
             for _ in range(500)]
    expected = np.quantile(means, [0.05, 0.95])
    got = bootstrap_mean_interval(values, confidence=0.9, resamples=500,
                                  seed_key="pin")
    assert got == pytest.approx(tuple(float(v) for v in expected), abs=1e-12)


def test_bootstrap_of_one_value_or_a_constant_is_that_value():
    """There is no spread to resample; the interval must not invent one."""
    assert bootstrap_mean_interval([0.4]) == (0.4, 0.4)
    assert bootstrap_mean_interval([0.7] * 20, resamples=50) == (0.7, 0.7)


def test_bootstrap_interval_contains_the_mean_and_shrinks_with_more_episodes():
    """Four times the episodes should roughly halve the width; that is the point of running more."""
    def width_around_mean(n):
        values = spl_like(n, seed=7)
        lower, upper = bootstrap_mean_interval(values, resamples=1000)
        assert lower <= sum(values) / n <= upper
        return upper - lower
    assert width_around_mean(400) < 0.6 * width_around_mean(100)


@pytest.mark.parametrize("values,resamples", [
    ([], 100), ([0.1, NAN], 100), ([0.1, float("inf")], 100),
    ([0.1, 0.2], 0), ([0.1, 0.2], 2.5)])
def test_bootstrap_refuses_empty_non_finite_and_bad_resamples(values,
                                                             resamples):
    """A NaN SPL in the input would silently propagate into the interval."""
    with pytest.raises(HarnessError):
        bootstrap_mean_interval(values, resamples=resamples)


def test_seeds_refuse_anything_but_a_string():
    """An int key would be tempting to hash(); refusing it keeps seeding in one place."""
    with pytest.raises(HarnessError, match="seed_key"):
        bootstrap_mean_interval([0.1, 0.2], seed_key=7)


# -- McNemar ---------------------------------------------------------------

@pytest.mark.parametrize("first_only,second_only,expected", [
    (0, 5, 2 / 32), (1, 9, 22 / 1024), (2, 8, 112 / 1024), (5, 5, 1.0),
    (3, 4, 1.0)])
def test_mcnemar_matches_the_binomial_tail(first_only, second_only, expected):
    """Exact values of the doubled Binomial(n, 1/2) tail (scipy's binomtest agrees)."""
    assert mcnemar_exact_p(*pairs(first_only, second_only)) == pytest.approx(
        expected, abs=1e-15)


def test_mcnemar_ignores_concordant_episodes():
    """Episodes both agents solve, or both fail, say nothing about which is better."""
    assert mcnemar_exact_p(*pairs(1, 9, both=40, neither=30)) == \
        mcnemar_exact_p(*pairs(1, 9))


def test_mcnemar_is_symmetric_in_its_sides():
    """A two-sided test must not care which agent is called first."""
    first, second = pairs(2, 8, both=3)
    assert mcnemar_exact_p(first, second) == mcnemar_exact_p(second, first)


def test_mcnemar_with_no_discordant_episodes_is_one():
    """Identical outcomes are no evidence of a difference."""
    assert mcnemar_exact_p(*pairs(0, 0, both=5, neither=5)) == 1.0


def test_mcnemar_handles_a_full_benchmark_without_overflow():
    """2 ** 2000 overflows a float; the exact integer tail must not."""
    p = mcnemar_exact_p(*pairs(900, 1100))
    assert 0.0 < p < 1e-4


@pytest.mark.parametrize("first,second", [
    ([], []), ([True], [True, False]), ([1, 0], [0, 1]),
    ([True, "false"], [True, True])])
def test_mcnemar_refuses_unpaired_or_non_bool_input(first, second):
    """A string 'false' is truthy; a paired test needs one bool per episode on each side."""
    with pytest.raises(HarnessError):
        mcnemar_exact_p(first, second)


# -- paired permutation ----------------------------------------------------

def test_permutation_enumerates_every_sign_flip_when_it_can():
    """[1, 2, 3]: only all-plus and all-minus reach |6| of the 8 patterns, so p = 2/8."""
    assert paired_permutation_p([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.25
    assert paired_permutation_p([1.0] * 5, [0.0] * 5) == 2 / 32


@pytest.mark.parametrize("first,second,expected", [
    ([0.9, 0.5, 0.5, 0.4, 0.5],
     [0.6666666666666666, 0.8, 0.0, 0.3333333333333333, 0.25], 0.3125),
    ([0.1, 0.7, 0.2, 0.2, 0.75, 1.0],
     [0.6666666666666666, 0.5, 0.0, 0.75, 0.4, 0.7], 0.90625),
], ids=["near_tie_counted", "exact_tie_lost"])
def test_the_enumeration_decides_ties_on_exact_differences(first, second, expected):
    """Rounded float differences once counted a near-tie as a hit (0.375) and lost an exact one (0.875)."""
    assert exact_sign_flip_p(first, second) == expected
    assert paired_permutation_p(first, second) == expected
    assert paired_permutation_p(second, first) == expected


def test_permutation_exact_and_monte_carlo_agree():
    """The Monte-Carlo branch must estimate the same p the enumeration computes."""
    rng = random.Random(11)
    first = [rng.random() for _ in range(16)]
    second = [x - 0.1 + rng.gauss(0.0, 0.3) for x in first]
    exact = paired_permutation_p(first, second, permutations=2 ** 16)
    sampled = paired_permutation_p(first, second, permutations=20000)
    assert 0.02 < exact < 0.98
    assert sampled == pytest.approx(exact, abs=0.015)


def test_permutation_monte_carlo_never_reports_zero():
    """(hits + 1) / (permutations + 1): a p of exactly 0 is impossible to observe."""
    assert paired_permutation_p([1.0] * 40, [0.0] * 40, permutations=999) == (
        1 / 1000)


def test_permutation_is_deterministic_and_symmetric_in_its_sides():
    """Sign flips make A-vs-B and B-vs-A the same test, draw for draw."""
    first, second = spl_like(20, seed=2), spl_like(20, seed=3)
    p = paired_permutation_p(first, second, permutations=2000)
    assert paired_permutation_p(first, second, permutations=2000) == p
    assert paired_permutation_p(second, first, permutations=2000) == p


def test_permutation_with_identical_scores_is_one():
    """Zero differences everywhere are no evidence of a difference."""
    assert paired_permutation_p([0.3, 0.0, 1.0], [0.3, 0.0, 1.0]) == 1.0


@pytest.mark.parametrize("first,second,permutations", [
    ([], [], 100), ([0.1], [0.1, 0.2], 100), ([NAN], [0.1], 100),
    ([0.1], [0.2], 0), ([True], [0.2], 100)])
def test_permutation_refuses_unpaired_or_non_finite_input(first, second,
                                                         permutations):
    """Unpaired lengths mean the two sides were not joined on episode id."""
    with pytest.raises(HarnessError):
        paired_permutation_p(first, second, permutations=permutations)
