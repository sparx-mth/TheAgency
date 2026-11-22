"""Which frontier node is asked about, and when: once, when it has a picture worth asking about -- and never again.

The bookkeeping of the rule the node valuation runs under (decided
2026-11-22), kept apart from whoever does the asking:

* a node is **scored once**: on the first action it has a usable view
  (``selection.NodeViews.usable``) -- the action it appears, when a frame
  already shows it, or a later one. Its answer is kept under its sticky id;
* a node already scored is **never asked about again**, however the frames
  that show it improve: only new nodes cost a question;
* a node that is **still there after it was visited** -- walked to, looked
  into -- is not re-scored either: something keeps it from resolving (a
  boundary under furniture, a gap the body cannot pass), and it is marked
  **blocked**.

The ledger holds no probability and asks no model. It answers which nodes
are due this action (:meth:`ScoreLedger.due`), is told what was done
(:meth:`ScoreLedger.mark_scored`, :meth:`ScoreLedger.mark_visited`) and
keeps, per node, the steps it was first and last seen, scored and blocked
at, and the frames it was scored on -- which is all an offline replay needs
to count the questions an episode would have cost.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

#: A node's states. PENDING: seen, not scored -- no usable view yet. SCORED: asked about, answer kept.
#: BLOCKED: still there well after it was visited; not asked about (again).
PENDING, SCORED, BLOCKED = "pending", "scored", "blocked"


@dataclass
class NodeScore:
    """What the ledger knows of one node.

    Attributes:
        node_id: The node's sticky id.
        first_step: The step the node first appeared at.
        last_step: The latest step it was present at.
        state: :data:`PENDING`, :data:`SCORED` or :data:`BLOCKED`.
        scored_step: The step it was scored at, or None.
        frames: The names of the frames it was scored on.
        visited_step: The step it was first visited at, or None.
        blocked_step: The step it was found still there after a visit, or None.
    """

    node_id: int
    first_step: int
    last_step: int
    state: str = PENDING
    scored_step: Optional[int] = None
    frames: Tuple[str, ...] = ()
    visited_step: Optional[int] = None
    blocked_step: Optional[int] = None

    @property
    def wait_steps(self) -> Optional[int]:
        """Steps from the node's first appearance to its scoring; None while unscored."""
        return None if self.scored_step is None else int(self.scored_step) - int(self.first_step)


class ScoreLedger:
    """The score-once rule over an episode's frontier nodes.

    Args:
        settle_steps: A visited node still present this many steps after
            the visit is blocked. Twelve by default: a full look-around at
            the benchmark's 30-degree turn, the longest a visit takes to
            show the camera what is there.
        min_age_steps: A node is not offered for scoring until it has been
            there this many steps. 0 by default -- the rule as decided: a
            node is scored the action it appears, when it has a view. The
            first replay found nearly a quarter of the questions asked that
            way (21 of 91) spent on nodes gone two actions later (the edge of
            the camera's own field of view is a frontier until the next turn
            of a look-around); a small age buys those back at the price of
            ordering that many actions without the node's number.

    Raises:
        ValueError: On a ``settle_steps`` that is not a positive integer, or a negative ``min_age_steps``.
    """

    def __init__(self, settle_steps: int = 12, min_age_steps: int = 0) -> None:
        if type(settle_steps) is not int or settle_steps < 1:
            raise ValueError("settle_steps must be a positive integer, got %r" % (settle_steps,))
        if type(min_age_steps) is not int or min_age_steps < 0:
            raise ValueError("min_age_steps must be a non-negative integer, got %r" % (min_age_steps,))
        self.settle_steps = settle_steps
        self.min_age_steps = min_age_steps
        self.nodes: Dict[int, NodeScore] = {}
        self._present: Tuple[int, ...] = ()
        self._step = 0

    def observe(self, step: int, node_ids: Iterable[int]) -> List[int]:
        """Note the nodes present at ``step``; returns the ids that became blocked on it.

        Raises:
            ValueError: If ``step`` is older than a step already observed for one of the nodes.
        """
        step = int(step)
        blocked = []
        present = []
        for node_id in node_ids:
            node_id = int(node_id)
            record = self.nodes.get(node_id)
            if record is None:
                record = self.nodes[node_id] = NodeScore(node_id, first_step=step, last_step=step)
            elif step < record.last_step:
                raise ValueError("node %d was seen at step %d; step %d is older" % (node_id, record.last_step, step))
            record.last_step = step
            present.append(node_id)
            if (record.state != BLOCKED and record.visited_step is not None
                    and step - record.visited_step >= self.settle_steps):
                record.state, record.blocked_step = BLOCKED, step
                blocked.append(node_id)
        self._present, self._step = tuple(present), step
        return blocked

    def pending(self) -> List[int]:
        """The nodes to look for views of: present at the last observed step, old enough, never scored and not blocked."""
        return [node_id for node_id in self._present
                if self.nodes[node_id].state == PENDING
                and self._step - self.nodes[node_id].first_step >= self.min_age_steps]

    def due(self, views: Mapping[int, object]) -> List[int]:
        """The pending nodes with a usable view in ``views`` (``{node_id: NodeViews}``): the ones to ask about now."""
        return [node_id for node_id in self.pending()
                if node_id in views and getattr(views[node_id], "usable", False)]

    def mark_scored(self, node_id: int, step: int, frames: Iterable[str]) -> None:
        """Record that ``node_id`` was asked about at ``step`` on ``frames``.

        Raises:
            KeyError: For a node never observed.
            ValueError: For a node already scored or blocked -- the rule is once.
        """
        record = self.nodes[int(node_id)]
        if record.state != PENDING:
            raise ValueError("node %d is %s (step %s); a node is scored once"
                             % (record.node_id, record.state, record.scored_step if record.state == SCORED
                                else record.blocked_step))
        record.state, record.scored_step, record.frames = SCORED, int(step), tuple(str(name) for name in frames)

    def mark_visited(self, node_id: int, step: int) -> None:
        """Record that the agent visited ``node_id`` at ``step``.

        The first visit counts: "still there" is measured from the action
        the agent first got there, so a node the agent goes on standing
        beside cannot keep postponing its own verdict.

        Raises:
            KeyError: For a node never observed.
        """
        record = self.nodes[int(node_id)]
        if record.visited_step is None:
            record.visited_step = int(step)

    def counts(self) -> Dict[str, int]:
        """How many nodes are in each state, and how many were ever seen."""
        out = {PENDING: 0, SCORED: 0, BLOCKED: 0}
        for record in self.nodes.values():
            out[record.state] += 1
        out["nodes"] = len(self.nodes)
        return out
