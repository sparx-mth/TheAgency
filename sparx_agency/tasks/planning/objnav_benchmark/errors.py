"""What can go wrong in the harness itself, named so the message says which.

Every one of these stops a run instead of letting it write a number. A
benchmark harness has one job -- produce a figure someone will put in a table
next to a paper's -- and the failures it guards against all produce a
*plausible* figure: two experiments appended to one results file, a path length
measured in the wrong frame, an episode counted twice after a resume.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations


class HarnessError(ValueError):
    """Base for every failure of the benchmark harness."""


class ScoringError(HarnessError):
    """A measurement that cannot be scored as the benchmark's protocol demands.

    A success without STOP where STOP is required, an observed path length
    that disagrees with the environment's, or a native metric the harness
    recomputes differently. Each is an adapter bug that would otherwise move
    SPL silently.
    """


class ResultsError(HarnessError):
    """Results that cannot be written or read without corrupting them.

    Refusing to overwrite a run, resuming with a different configuration,
    an episode logged twice, a row from another schema version, or a record
    whose fields contradict each other.
    """
