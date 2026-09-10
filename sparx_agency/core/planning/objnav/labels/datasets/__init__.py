"""One label table per dataset vocabulary, added by the benchmark branches.

Each module here defines a
:class:`~sparx_agency.core.planning.objnav.labels.table_mapper.TableLabelMapper`
for one dataset (HM3D, MP3D, Gibson, RoboTHOR) and is registered by name in
:func:`~sparx_agency.core.planning.objnav.labels.registry.default_label_mapper_registry`,
whose factory imports the module lazily inside its closure.

Python 3.8 syntax, standard library only.
"""
