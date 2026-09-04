"""The knobs, all of them, and why each default is where it is.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RptStarParams:
    """Tuning for one solve.

    Attributes:
        epsilon: The sub-optimality factor for F-RPT*. ``None`` -- the default
            -- runs the exact search of Section IV-B; a number runs the focal
            search of Section IV-D and guarantees a route within
            ``(1 + epsilon)`` of optimal.

            **Exact is the default because a small epsilon buys nothing.** The
            paper's own suggested value of 0.01 (p.12) was measured here
            against exact search across belief shapes and problem sizes and
            returned the same routes, slightly more slowly, while downgrading
            the guarantee from optimal to bounded. Its band is simply too
            narrow to admit states that exact search would not have expanded
            anyway.

            Epsilon becomes worth setting at around ``0.5``, and only in a
            narrow window: a *concentrated* belief over roughly eighteen to
            twenty-four places, where it triples how often the search finishes
            inside its budget. Below that window exact already wins; above it,
            nothing helps and the fallback route takes over. See the README.
        max_expansions: Stop after this many state expansions. Deterministic,
            so it is what tests use. ``None`` for no limit.
        time_budget_s: Stop after this long. Wall-clock, checked every
            :data:`BUDGET_CHECK_INTERVAL` expansions so the clock is not read
            in the inner loop. ``None`` for no limit. Finite by default because
            the search is exponential in the worst case and the caller is
            usually an aircraft that is airborne.
        require_triangle_inequality: Whether to reject a cost matrix in which
            a detour beats going direct. Dominance pruning is unsound without
            it and the result would silently not be optimal, so this defaults
            to on. Shortest-path costs satisfy it by construction; see
            :func:`~sparx_agency.core.planning.routing.rpt_star.costs.metric_closure`
            for the repair.
        verify_reconstruction: Re-score the returned route from scratch and
            check it against the cost the search terminated on. Linear in the
            route length against an exponential search, and it catches the
            entire family of bugs where the state machine and the objective
            drift apart. On by default; there is no good reason to turn it off.
    """

    epsilon: Optional[float] = None
    max_expansions: Optional[int] = None
    time_budget_s: Optional[float] = 5.0
    require_triangle_inequality: bool = True
    verify_reconstruction: bool = True

    def __post_init__(self):
        # type: () -> None
        """Refuse an epsilon outside the range the paper defines.

        The paper puts it in ``[0, infinity)`` (p.6). A negative value does not
        corrupt the route -- the band ``(1 + epsilon) * f_min`` admits nothing,
        FOCAL stays empty and the search quietly degrades to exact A* -- but
        the result would still be stamped as bounded by a factor smaller than
        one, which is a promise no route can keep and which a caller checking
        ``cost <= (1 + epsilon) * lower_bound`` would read as a failure. Every
        other out-of-domain input to this package raises; this one should too.

        Raises:
            ValueError: If epsilon is negative or not a number.
        """
        if self.epsilon is None:
            return
        epsilon = float(self.epsilon)
        if epsilon != epsilon:                  # NaN
            raise ValueError("epsilon is NaN; use None for an exact search")
        if epsilon < 0.0:
            raise ValueError(
                "epsilon must be at least 0.0 (the paper's range is [0, inf), "
                "p.6), got %r. Use None for an exact search." % (self.epsilon,))


#: How often the wall clock is read, in expansions. Reading it every expansion
#: costs more than the search does on easy instances.
BUDGET_CHECK_INTERVAL = 256

#: How far the reconstructed cost may differ from the search's own g-value
#: before the tripwire fires, relative to the cost itself.
RECONSTRUCTION_TOLERANCE = 1e-9
