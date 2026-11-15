"""The four contracts of the ObjectNav layer.

* :class:`ObjNavEnv` -- a simulator, behind one face (one adapter per
  benchmark branch).
* :class:`ObjNavAgent` -- what the harness drives: observation in, action out.
* :class:`SearchPolicy` -- where a search method plugs into the agent.
* :class:`LabelMapper` -- a dataset's categories, in our vision vocabulary.

Python 3.8 syntax, standard library only.
"""
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.interfaces.label_mapper import LabelMapper
from sparx_agency.core.planning.objnav.interfaces.policy import SearchPolicy

__all__ = ["ObjNavEnv", "ObjNavAgent", "SearchPolicy", "LabelMapper"]
