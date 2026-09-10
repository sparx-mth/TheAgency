"""A dataset's label table, checked once and served as ready-made episode targets.

A benchmark names its goals in its own words, and our perception stack needs
each one as an LLM query, detector prompts and an accept set (see
:mod:`sparx_agency.core.planning.objnav.types.target`). :class:`TableLabelMapper`
turns one hand-written table of :class:`LabelSpec` rows into those
:class:`TargetLabels`, and refuses, when it is built, the table mistakes that
would otherwise surface only as a wrong score:

* **one detector label accepted by two categories** -- whichever of the two an
  episode asks for, a detection of the other can end it with a false STOP;
* **two categories that normalise to one string** (``"tv_monitor"`` and
  ``"TV Monitor"``) -- one object typed twice, so the table is ambiguous about
  which row a result belongs to;
* **a context vocabulary given as a bare string or a set** -- the first would
  become single letters, the second a prompt order that changes from run to
  run.

Lookups are verbatim on purpose. :meth:`TableLabelMapper.target_labels` does
not normalise the category it is asked for: an adapter that respells the
dataset's category has a bug, and normalising here would hide it until two
categories collided.

Everything is built in the constructor, so a bad table fails when the mapper is
created -- before the first episode, not at the three-hundredth.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec, TargetLabels


def _unique_in_order(values: Iterable[str]) -> Tuple[str, ...]:
    """``values`` without repeats, each kept at its first position."""
    return tuple(dict.fromkeys(values))


def _table_rows(name: str, table) -> List[Tuple[str, LabelSpec]]:
    """The rows of ``table`` in its own order, every key and value checked.

    Copied into a list so that changing the caller's mapping afterwards cannot
    change the mapper.
    """
    if not isinstance(table, collections.abc.Mapping):
        raise ObjNavError(
            "label mapper %r needs a mapping from dataset category to "
            "LabelSpec, got a %s" % (name, type(table).__name__))
    rows = list(table.items())
    if not rows:
        raise ObjNavError(
            "label mapper %r has an empty table; give it one LabelSpec row per "
            "dataset category" % (name,))
    for category, spec in rows:
        if not isinstance(category, str) or not normalize_label(category):
            raise ObjNavError(
                "label mapper %r: every category must be a non-blank string, "
                "spelled as the dataset spells it, got %r" % (name, category))
        if not isinstance(spec, LabelSpec):
            raise ObjNavError(
                "label mapper %r: category %r maps to %r; write the row as "
                "LabelSpec(query, prompts, accept)" % (name, category, spec))
    return rows


def _check_distinct_categories(name: str, categories: Sequence[str]) -> None:
    """Raise when two categories are one object typed twice."""
    first_spelling: Dict[str, str] = {}
    clashes = []
    for category in categories:
        key = normalize_label(category)
        if key in first_spelling:
            clashes.append("%r and %r both normalise to %r"
                           % (first_spelling[key], category, key))
        else:
            first_spelling[key] = category
    if clashes:
        raise ObjNavError(
            "label mapper %r has two rows for one object: %s. Keep one row "
            "per dataset category, spelled as the dataset spells it"
            % (name, "; ".join(clashes)))


def _build_target(category: str, spec: LabelSpec) -> TargetLabels:
    """The normalised :class:`TargetLabels` of one table row."""
    prompts = _unique_in_order(normalize_label(p) for p in spec.prompts)
    accept = frozenset(prompts + tuple(normalize_label(a) for a in spec.accept))
    return TargetLabels(category=category, query=normalize_label(spec.query),
                        detector_prompts=prompts, accept_labels=accept)


def _check_unambiguous(name: str, targets: Sequence[TargetLabels]) -> None:
    """Raise when one detector label counts as two different categories.

    Every clash is reported at once, in table order, so one fix-and-rerun
    cycle is enough however many rows collide.
    """
    owner: Dict[str, str] = {}
    clashes = []
    for target in targets:
        for label in sorted(target.accept_labels):
            first = owner.setdefault(label, target.category)
            if first != target.category:
                clashes.append("%r is accepted by both %r and %r"
                               % (label, first, target.category))
    if clashes:
        raise ObjNavError(
            "label mapper %r is ambiguous: %s. A detection of such a label is "
            "a guaranteed false STOP for one of its categories; keep every "
            "label (prompts included) in exactly one row"
            % (name, "; ".join(clashes)))


def _context_entries(name: str, values) -> Tuple[str, ...]:
    """The context vocabulary, normalised, without repeats, order kept."""
    if isinstance(values, str) or not isinstance(values, collections.abc.Sequence):
        raise ObjNavError(
            "label mapper %r: context_vocabulary must be a list or tuple of "
            "strings (a bare string would become letters, a set an order that "
            "changes between runs), got %r" % (name, values))
    entries = []
    for value in values:
        if not isinstance(value, str) or not normalize_label(value):
            raise ObjNavError(
                "label mapper %r: context_vocabulary must hold non-blank "
                "strings, got %r" % (name, value))
        entries.append(normalize_label(value))
    return _unique_in_order(entries)


class TableLabelMapper(LabelMapper):
    """A :class:`LabelMapper` backed by one dataset's hand-written table.

    Every :class:`TargetLabels` is built and cross-checked in the
    constructor; :meth:`target_labels` hands out the prebuilt object.

    Args:
        name: The registry key, e.g. ``"hm3d"``.
        table: Dataset category, verbatim, to its :class:`LabelSpec`. Its
            iteration order is the dataset's order, which :meth:`categories`
            and :meth:`vocabulary` keep. Copied: changing the mapping
            afterwards does not change the mapper.
        context_vocabulary: Further detector prompts that are not goals but
            tell rooms apart (``"sink"``, ``"refrigerator"``). Normalised.

    Raises:
        ObjNavError: On a blank name; a table that is not a non-empty mapping
            from non-blank strings to :class:`LabelSpec`; two categories that
            normalise to one string; a detector label accepted by two
            categories; or a context vocabulary that is not a list or tuple
            of non-blank strings.
    """

    def __init__(self, name: str, table: Mapping[str, LabelSpec],
                 context_vocabulary: Sequence[str] = ()) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ObjNavError(
                "TableLabelMapper.name must be a non-blank string (the "
                "registry key), got %r" % (name,))
        rows = _table_rows(name, table)
        _check_distinct_categories(name, [category for category, _ in rows])
        targets = [_build_target(category, spec) for category, spec in rows]
        _check_unambiguous(name, targets)
        context = _context_entries(name, context_vocabulary)
        self._name = name
        self._targets: Dict[str, TargetLabels] = {t.category: t for t in targets}
        self._categories = tuple(t.category for t in targets)
        self._vocabulary = _unique_in_order(
            [prompt for t in targets for prompt in t.detector_prompts]
            + list(context))

    @property
    def name(self) -> str:
        """The registry key, e.g. ``"hm3d"``."""
        return self._name

    def categories(self) -> Tuple[str, ...]:
        """The dataset's goal categories, verbatim, in table order."""
        return self._categories

    def target_labels(self, category: str) -> TargetLabels:
        """The prebuilt goal ``category`` in our vision vocabulary.

        Args:
            category: A dataset category, verbatim. Not normalised: a
                respelled category is the adapter's bug, reported here.

        Returns:
            The same :class:`TargetLabels` object on every call.

        Raises:
            UnknownCategoryError: ``category`` is not one of
                :meth:`categories`. The message names this mapper, lists its
                categories, and points at the table's spelling when the two
                differ only in case or separators.
        """
        target = self._targets.get(category) if isinstance(category, str) else None
        if target is None:
            raise UnknownCategoryError(self._unknown_category_message(category))
        return target

    def vocabulary(self) -> Tuple[str, ...]:
        """Every category's prompts in table order, then the context vocabulary.

        Normalised and without repeats; the accept-only labels are not here,
        because nothing asks the detector for them.
        """
        return self._vocabulary

    def _unknown_category_message(self, category) -> str:
        """Why ``category`` is not covered, and what to pass instead."""
        message = ("label mapper %r has no category %r; it covers %s"
                   % (self._name, category,
                      ", ".join(repr(c) for c in self._categories)))
        if isinstance(category, str):
            wanted = normalize_label(category)
            for known in self._categories:
                if normalize_label(known) == wanted:
                    message += (
                        ". %r differs from the table's %r only in case or "
                        "separators; categories are matched verbatim, so pass "
                        "the dataset's own spelling" % (category, known))
        return message

    def __repr__(self) -> str:
        return ("TableLabelMapper(name=%r, categories=%r, vocabulary=%d prompts)"
                % (self._name, self._categories, len(self._vocabulary)))
