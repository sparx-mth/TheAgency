"""Shared, ROS-free math utilities (functions, not types).

These helpers are cross-cutting (localization, planning adapters, robots) and
deliberately stateless. Data *types* live in :mod:`sparx_agency.core.common.types`;
this package holds the algorithmic helpers that operate on them.
"""
