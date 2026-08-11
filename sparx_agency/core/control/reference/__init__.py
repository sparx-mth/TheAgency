"""What the plan says, held in one place for every control backend.

See ``feed.py`` for why the trajectory lives here rather than inside a
controller, and ``diagnosis.py`` for why the gap from the plan is decomposed the
way it is.
"""
from sparx_agency.core.control.reference.diagnosis import decompose_error
from sparx_agency.core.control.reference.feed import TrajectoryFeed
from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.control.reference.types import ReferenceSample

__all__ = [
    "TrajectoryFeed",
    "ReferenceParams",
    "ReferenceSample",
    "decompose_error",
]
