"""HTTP clients for the services the method is given: the detector, the segmenter and the room LLM.

Each implements a protocol of
:mod:`sparx_agency.core.planning.object_search.interfaces` over a wire format
owned by the serving side (``tasks/mapping/scene_graph/serve/``); importing
this package loads no model.
"""
