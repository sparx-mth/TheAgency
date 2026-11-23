"""Zero-shot ObjectNav: observed RGB-D, VLM-scored frontiers, RPT*, A* and target closing.

A :class:`~sparx_agency.core.planning.objnav.interfaces.policy.SearchPolicy`
for the ObjectNav contract in :mod:`sparx_agency.core.planning.objnav`. The
contract layer is the seam -- types, interfaces, the action converter -- and
this package is the method that plugs into it. It is simulator-free and
service-free: a detector, an optional segmenter and a frontier-scoring VLM are
injected behind the protocols in :mod:`.interfaces`, and the benchmark
adapters and HTTP clients live with the runtime under
``tasks/planning/objnav_benchmark_runtime/``. See README.md.

Importing this package loads nothing heavy; import the policy by module path:
``from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy``.
"""
