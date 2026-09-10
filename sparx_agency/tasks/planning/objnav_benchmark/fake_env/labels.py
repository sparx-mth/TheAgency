"""The fake building's label table: its object categories, in plain words.

A **test rig, not a benchmark.** The fake has no appearance and its oracle
reads no detector, so the rows only have to cover the demo world's categories
-- but they are a real :class:`TableLabelMapper`, so the headless agent maps
every episode's category through it exactly as it maps HM3D's, and a missing
row fails here as it would there: at reset, before the first step.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper
from sparx_agency.core.planning.objnav.types.target import LabelSpec

#: The mapper's name, as a registry would key it.
FAKE_LABEL_MAPPER_NAME = "fake"


def fake_label_mapper() -> TableLabelMapper:
    """The label mapper of the fake building's categories: chair, bed, toilet.

    Returns:
        A new :class:`TableLabelMapper` named ``"fake"``.
    """
    return TableLabelMapper(FAKE_LABEL_MAPPER_NAME, {
        "chair": LabelSpec(query="chair", prompts=("chair",),
                           accept=("armchair",)),
        "bed": LabelSpec(query="bed", prompts=("bed",)),
        "toilet": LabelSpec(query="toilet", prompts=("toilet",)),
    })
