"""Routing: what order to visit places in, over an abstract cost graph.

Distinct from ``core/planning/planners/``, which finds a geometric path from
one pose to another through a grid. Nothing here knows what a pose, a frame or
a metre is: a routing problem is a set of places, a number for how much it
costs to get from each to each, and whatever else the particular problem needs
to weigh -- for RPT*, a probability that the search ends at each place.

Its consumers are the layers that have already decided *where* the candidate
places are and now have to decide the order: a room search choosing which room
to look in next, an object mission choosing which of several catalogued objects
to fly to first.

Contents:

* :mod:`~sparx_agency.core.planning.routing.rpt_star` -- the Hamiltonian path
  with probabilistic terminals: the visiting order that minimises the expected
  cost of finding something.

Planned second inhabitant: a builder that turns an occupancy grid and a set of
places into the cost matrix these solvers consume, one graph search per place
rather than one per pair. Until it exists, ``rpt_star.costs`` accepts a row
callback so a caller can do exactly that in a few lines.
"""
