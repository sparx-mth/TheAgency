"""The imports this recorder cannot take for granted, resolved in one place.

A node that runs inside a container it does not own has two import problems, and
both of them are the difference between a degraded recording and no recording:

**Vendor message types.** ``fcu_driver_interfaces``, ``sphera_common_interfaces``
and ``rooster_manager_interfaces`` are built only inside the ``it`` container.
Missing ones disable one stream each and land in :data:`MISSING`, which the run's
manifest carries, so the reason is readable after the flight instead of guessed.

**The shared on-disk contract.** ``nav_debug/schema.py`` is documented as pure
stdlib precisely so the Foxy recorder can import it, but importing it the
ordinary way first runs ``nav_debug/__init__.py``, which re-exports the offline
player and therefore pulls in numpy and everything below it. That package is a
host-side analysis tool; the recorder is flight-side and must not inherit its
dependency set. So the ordinary import is tried first -- it is the normal case,
and it keeps both halves on the same module object -- and ``schema.py`` is loaded
straight off disk only when that fails.

Python 3.8 compatible: this runs under ROS 2 Foxy in the vendor container.
"""
from __future__ import annotations

import importlib
import importlib.util
import os

#: Stream name -> why its vendor message type could not be imported here.
MISSING = {}

_ROBOT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(_ROBOT_DIR)),
                            "tasks", "planning", "nav_debug", "schema.py")


def _message(stream, module_name, class_name):
    """Import one vendor message class, recording the reason if it is absent."""
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        MISSING[stream] = "{}.{}: {}".format(module_name, class_name, exc)
        return None


def _load_schema():
    """Return the nav-debug schema module, by import or straight off disk."""
    try:
        from sparx_agency.tasks.planning.nav_debug import schema as imported
        return imported
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "sparx_agency_nav_debug_schema", _SCHEMA_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


#: The on-disk contract every writer and reader of a run folder shares.
schema = _load_schema()

#: The final ManualControl the airframe acts on.
ManualControl = _message("manual", "fcu_driver_interfaces.msg", "ManualControl")
#: Sphera's own pawn state (published BEST_EFFORT).
SpheraPawnState = _message("sphera", "sphera_common_interfaces.msg",
                           "SpheraPawnState")
#: The Rooster manager's view of the airframe: battery, armed, flight mode.
RoosterState = _message("state", "rooster_manager_interfaces.msg", "RoosterState")
