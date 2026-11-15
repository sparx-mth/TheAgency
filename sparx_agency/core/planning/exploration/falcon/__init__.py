"""Bounded FALCON planar adaptation; heavy host-side modules load explicitly.

The parent exploration facade deliberately does not import these scipy modules,
so existing Python 3.8/Noetic drone adapters do not gain a new dependency.
"""
from sparx_agency.core.planning.exploration.falcon.params import FalconParams, SOURCE

__all__ = ["FalconParams", "SOURCE"]

