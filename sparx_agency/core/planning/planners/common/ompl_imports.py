"""OMPL library imports and availability check."""
from __future__ import annotations

try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
    OMPL_ERROR = None
except ImportError as e:
    ob = None
    og = None
    OMPL_AVAILABLE = False
    OMPL_ERROR = str(e)

__all__ = ["ob", "og", "OMPL_AVAILABLE", "OMPL_ERROR"]
