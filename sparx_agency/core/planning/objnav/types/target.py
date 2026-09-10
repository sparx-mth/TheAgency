"""A dataset category, translated into our vision vocabulary.

A benchmark names its goals in its own words -- ``tv_monitor`` in HM3D,
``Television`` in RoboTHOR, ``chest_of_drawers`` in MP3D. Our perception stack
needs three different things from that one word:

* a **query** a language model understands (``"tv"``), because the room
  probabilities come from an LLM and an underscore token reaches it verbatim;
* **detector prompts**, because an open-vocabulary detector only finds what it
  is asked for, and nothing adds the target to its prompt list for you;
* an **accept set**: the detector labels that *count* as the target. Exact
  membership after :func:`~sparx_agency.core.common.label_match.normalize_label`,
  never a fuzzy match -- a false accept is a false STOP and a failed episode.

:class:`LabelSpec` is one hand-written row of a dataset's table;
:class:`TargetLabels` is what the label mapper hands the search policy for one
episode, already normalised.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from typing import FrozenSet, Tuple

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.planning.objnav.errors import ObjNavError


def _strings(values, what: str, ordered: bool) -> Tuple[str, ...]:
    """``values`` as a tuple of non-blank strings, or raise naming ``what``.

    ``ordered`` values must arrive as a list or tuple: their order is kept
    (primary prompt first), and a set's order changes from one process to
    the next -- a run and its resume would prompt the detector differently.
    """
    if ordered and (isinstance(values, str)
                    or not isinstance(values, collections.abc.Sequence)):
        raise ObjNavError(
            "%s must be a list or tuple of strings (a bare string would "
            "become letters, a set an order that changes between runs), got "
            "%r" % (what, values))
    if isinstance(values, str):
        raise ObjNavError(
            "%s must be a collection of strings, not the bare string %r"
            % (what, values))
    result = tuple(values)
    for value in result:
        if not isinstance(value, str) or not normalize_label(value):
            raise ObjNavError(
                "%s must hold non-blank strings, got %r" % (what, value))
    return result


@dataclass(frozen=True)
class LabelSpec:
    """One row of a dataset's label table, written by a person in plain words.

    The mapper normalises it; nothing here needs to be lowercase already.

    Attributes:
        query: What a person would call the object, for the LLM prompts
            (``"tv"``, ``"potted plant"``).
        prompts: Open-vocabulary detector prompts that should fire on it,
            primary first: a list or tuple, never a set. At least one.
        accept: Further detector labels that count as this category when a
            detector emits them (a closed-vocabulary detector's synonyms), as
            a list or tuple. The prompts are always accepted as well.

    Raises:
        ObjNavError: On a blank query, no prompts, prompts or accept labels
            that are not a list or tuple (a set's order changes between runs),
            or a blank entry.
    """

    query: str
    prompts: Tuple[str, ...]
    accept: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not normalize_label(self.query):
            raise ObjNavError(
                "LabelSpec.query must be a non-blank string, got %r"
                % (self.query,))
        prompts = _strings(self.prompts, "LabelSpec.prompts", ordered=True)
        if not prompts:
            raise ObjNavError("LabelSpec.prompts needs at least one prompt")
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "accept", _strings(
            self.accept, "LabelSpec.accept", ordered=True))


@dataclass(frozen=True)
class TargetLabels:
    """The goal of one episode, in our vision vocabulary.

    Every string is already normalised. Build it through a label mapper, not
    by hand, so the dataset's table stays the one source of truth.

    Attributes:
        category: The dataset's category, verbatim (``"tv_monitor"``) -- the
            key results are reported under.
        query: Natural-language phrase for the LLM prompts.
        detector_prompts: Detector prompts, primary first, without repeats:
            a list or tuple, stored as a tuple.
        accept_labels: Detector labels that count as the target, in any
            collection (their order means nothing), stored as a frozenset.
            Always contains every prompt.

    Raises:
        ObjNavError: On a blank category or query, detector prompts that are
            not a list or tuple, a string that is not normalised, a repeated
            or missing prompt, or a prompt that is not accepted.
    """

    category: str
    query: str
    detector_prompts: Tuple[str, ...]
    accept_labels: FrozenSet[str]

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ObjNavError(
                "TargetLabels.category must be a non-blank string, got %r"
                % (self.category,))
        if not isinstance(self.query, str) or not normalize_label(self.query):
            raise ObjNavError(
                "TargetLabels.query must be a non-blank string -- it is what "
                "the LLM is asked about -- got %r" % (self.query,))
        prompts = _strings(self.detector_prompts,
                           "TargetLabels.detector_prompts", ordered=True)
        accept = frozenset(_strings(self.accept_labels,
                                    "TargetLabels.accept_labels",
                                    ordered=False))
        if not prompts:
            raise ObjNavError("TargetLabels.detector_prompts is empty")
        if len(set(prompts)) != len(prompts):
            raise ObjNavError(
                "TargetLabels.detector_prompts repeats a prompt: %r"
                % (prompts,))
        for value in (self.query,) + prompts + tuple(accept):
            if not isinstance(value, str) or value != normalize_label(value):
                raise ObjNavError(
                    "TargetLabels strings must be normalised; %r is not (it "
                    "normalises to %r)" % (value, normalize_label(value)))
        unaccepted = [p for p in prompts if p not in accept]
        if unaccepted:
            raise ObjNavError(
                "TargetLabels would prompt the detector for %r but not "
                "accept what it finds" % (unaccepted,))
        object.__setattr__(self, "detector_prompts", prompts)
        object.__setattr__(self, "accept_labels", accept)

    def accepts(self, detector_label: str) -> bool:
        """Whether a detector label counts as this target.

        Exact membership after normalisation. Deliberately not
        :func:`~sparx_agency.core.common.label_match.label_matches`: its
        shared-token rule accepts ``"car"`` for ``"car keys"``, and on a
        benchmark a false accept ends the episode with a false STOP.
        """
        return normalize_label(detector_label) in self.accept_labels
