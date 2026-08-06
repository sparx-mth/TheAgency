"""Specific thrust to throttle, with the scale learned in flight."""
from sparx_agency.core.control.thrust_model.model import ThrustModel, specific_force_along
from sparx_agency.core.control.thrust_model.params import ThrustModelParams

__all__ = ["ThrustModel", "ThrustModelParams", "specific_force_along"]
