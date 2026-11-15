"""The headless agent: a search policy, a label mapper and an action converter behind one step API.

* :mod:`.headless_agent` -- :class:`HeadlessObjNavAgent`, the
  :class:`~sparx_agency.core.planning.objnav.interfaces.ObjNavAgent` every
  method is benchmarked through.
* :mod:`.episode_checks` -- its boundary checks: the mapped target, and each
  frame against its episode.
* :mod:`.episode_log` -- what it records: each step's reasons and each
  episode's counters and report.
* :mod:`.params` -- :class:`HeadlessAgentParams`, its tuning.

Python 3.8 syntax; numpy arrives only through the observation type.
"""
from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams

__all__ = ["HeadlessObjNavAgent", "HeadlessAgentParams"]
