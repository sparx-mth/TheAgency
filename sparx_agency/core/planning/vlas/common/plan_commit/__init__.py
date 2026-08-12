"""Commit to a policy's prediction and fly it, instead of replacing it per frame.

A VLA answers per frame; an aircraft flies a route. This package is the piece
between the two: it anchors one prediction in the world, follows it with pure
pursuit, and says when enough of it has been flown to be worth asking again.

* :mod:`~sparx_agency.core.planning.vlas.common.plan_commit.progress` -- arc
  length along a polyline and distance from it.
* :mod:`~sparx_agency.core.planning.vlas.common.plan_commit.committed_plan` --
  the anchored prediction and its commit point.
* :mod:`~sparx_agency.core.planning.vlas.common.plan_commit.executor` -- the
  commitment itself: when it stands, when it is over, and what to fly meanwhile.

Policy-agnostic on purpose: nothing here knows what produced the trajectory, so
NavDP, FlowNav and anything after them share it. See ``README.md``.
"""
from .committed_plan import CommittedPlan, anchor_plan, commit_index_for
from .executor import CommitSpec, CommitTick, PlanCommitExecutor
from .progress import cumulative_arc, project

__all__ = [
    "CommitSpec",
    "CommitTick",
    "CommittedPlan",
    "PlanCommitExecutor",
    "anchor_plan",
    "commit_index_for",
    "cumulative_arc",
    "project",
]
