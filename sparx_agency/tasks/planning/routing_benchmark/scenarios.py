"""The set of scenarios, chosen rather than sampled.

A benchmark can be large or it can be interpretable, and the usual failure is
to be neither: ten thousand random instances drawn from one distribution, which
answer one question very precisely and every other question not at all.

So this is a **factorial design**. Four building shapes, four sizes, five
oracle regimes, several seeds each -- every combination, none omitted. That
buys two things a big random sample does not:

* every cell is populated, so "RPT* wins on ring-shaped buildings when the
  oracle is misleading" is a question with an answer rather than a subgroup of
  eleven instances;
* the axes are independent by construction, so a difference between two
  planners can be attributed to the factor that actually varied.

A few hundred scenarios is the right size for this. Each one is a distinct,
meaningful configuration rather than another draw from the same urn, and the
metrics are exact expectations rather than sampled, so there is no statistical
noise for extra repetitions to average away.

Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import random
import zlib

from sparx_agency.tasks.planning.routing_benchmark.buildings import (
    TOPOLOGIES,
    Building,
    generate_building,
)
from sparx_agency.tasks.planning.routing_benchmark.oracle import (
    ORACLE_MODELS,
    Belief,
    make_belief,
)

#: Room counts. Chosen to straddle the point where exact search stops being
#: affordable, because where that point falls is one of the things the
#: benchmark is for. Eight is comfortable, twenty is not.
SIZES = (8, 12, 16, 20)

#: Repeats per cell. Small on purpose: the metrics are exact expectations, so
#: repeats vary the *building and belief*, not the measurement.
SEEDS = (0, 1, 2, 3, 4)

#: How fast the aircraft flies, for the mission-time figure.
SPEED_MPS = 1.2

#: How long searching one room takes once inside it. Real, and large enough to
#: matter: at these distances a thirty-second sweep is comparable to the flight
#: between rooms, which changes which planner looks best.
DWELL_S = 30.0


@dataclass(frozen=True)
class Scenario:
    """One building, one belief about it, and the identity to group results by.

    Attributes:
        key: Unique name, e.g. ``"ring-16-misleading-s2"``.
        topology: Which building shape.
        n_rooms: How many rooms.
        oracle: Which oracle regime produced the belief.
        seed: Which repeat.
        building: The floor plan and its distance matrix.
        belief: The truth and what the oracle said.
    """

    key: str
    topology: str
    n_rooms: int
    oracle: str
    seed: int
    building: Building
    belief: Belief

    @property
    def distance(self):
        """Walking distances between rooms, entrance last."""
        return self.building.distance

    @property
    def entrance(self):
        """Index of the entrance in the distance matrix."""
        return self.n_rooms


def build_scenario(topology, n_rooms, oracle_model, seed):
    # type: (str, int, object, int) -> Scenario
    """Construct one scenario deterministically from its coordinates.

    The seed is mixed with every other factor, so the same building shape at
    two different sizes is genuinely a different building rather than a prefix
    of the same one.

    Args:
        topology: One of :data:`~...buildings.TOPOLOGIES`.
        n_rooms: How many rooms.
        oracle_model: An
            :class:`~...oracle.OracleModel`.
        seed: The repeat index.

    Returns:
        The scenario.
    """
    stamp = "%s-%d-%s-s%d" % (topology, n_rooms, oracle_model.name, seed)
    # crc32, not hash(): Python randomises string hashing per process, so hash()
    # here would silently make every run a different benchmark.
    rng = random.Random(zlib.crc32(stamp.encode("utf-8")))
    building = generate_building(topology, n_rooms, rng)
    belief = make_belief(n_rooms, oracle_model, rng,
                         distance=building.distance, start=n_rooms)
    return Scenario(key=stamp, topology=topology, n_rooms=n_rooms,
                    oracle=oracle_model.name, seed=seed,
                    building=building, belief=belief)


def all_scenarios(sizes=SIZES, seeds=SEEDS, topologies=TOPOLOGIES,
                  oracle_models=ORACLE_MODELS):
    # type: (Sequence[int], Sequence[int], Sequence[str], Sequence[object]) -> Iterator[Scenario]
    """Every cell of the factorial design, in a stable order.

    Yields:
        Each scenario once. The default sweep is
        ``len(topologies) * len(sizes) * len(oracle_models) * len(seeds)``
        scenarios -- 400 with the defaults.
    """
    for topology in topologies:
        for n_rooms in sizes:
            for model in oracle_models:
                for seed in seeds:
                    yield build_scenario(topology, n_rooms, model, seed)


def count(sizes=SIZES, seeds=SEEDS, topologies=TOPOLOGIES,
          oracle_models=ORACLE_MODELS):
    # type: (Sequence[int], Sequence[int], Sequence[str], Sequence[object]) -> int
    """How many scenarios a sweep will produce, without building them."""
    return len(topologies) * len(sizes) * len(oracle_models) * len(seeds)
