"""Dataset goal categories, translated into our vision vocabulary.

* :class:`TableLabelMapper` -- one dataset's hand-written table of
  :class:`~sparx_agency.core.planning.objnav.types.target.LabelSpec` rows,
  checked once and served as normalised ``TargetLabels``.
* :class:`LabelMapperRegistry` -- selects a dataset's mapper by name; the
  benchmark branches add their tables under ``labels/datasets`` and register
  them in :func:`default_label_mapper_registry`.

Python 3.8 syntax, standard library only.
"""
from sparx_agency.core.planning.objnav.labels.registry import (
    LabelMapperFactory,
    LabelMapperRegistry,
    default_label_mapper_registry,
)
from sparx_agency.core.planning.objnav.labels.table_mapper import TableLabelMapper

__all__ = [
    "TableLabelMapper",
    "LabelMapperFactory",
    "LabelMapperRegistry",
    "default_label_mapper_registry",
]
