"""Where the thing actually is, and what a language model thinks about it.

Two distributions over the same rooms, and keeping them apart is the whole
point of the benchmark:

* **the truth** -- where the object really is. Generated from room semantics: an
  apple is likely in a kitchen, a scalpel in an operating theatre. The
  benchmark samples the object's real location from this and it is never shown
  to any planner.
* **the belief** -- what the oracle reports, which is what the planners
  actually get. It is the truth seen through a model that may be sharp, blurry,
  or confidently wrong.

A planner optimises against the belief and is judged against the truth. That
gap is the only interesting question here: RPT* provably minimises expected
cost *under the distribution it is given*, so measuring it against its own
belief would be measuring nothing. The paper's most useful result is exactly
this comparison (its Tables II and III, accurate prior versus misleading
prior), and it is the one its abstract does not mention.

**One dial, and its sign matters.** :attr:`OracleModel.skill` blends the truth
into the belief in log space. At ``1.0`` the oracle reproduces the truth; at
``0.0`` it knows nothing and reports noise; at ``-1.0`` it is systematically
inverted -- confidently pointing at the emptiest rooms, which is what a
language model reasoning from a wrong premise about a building actually does.
Negative skill is not "random", and that distinction is the one that separates
the planners.

Standard library only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Room kinds, and how plausible the target is in each. Real numbers rather
#: than a uniform list so the truth has structure to find: most rooms are
#: unlikely, a few are worth searching, which is what makes ordering matter.
ROOM_KINDS = (
    ("kitchen", 8.0),
    ("store", 5.0),
    ("office", 3.0),
    ("ward", 2.0),
    ("lab", 2.0),
    ("bathroom", 0.6),
    ("plant", 0.3),
    ("stairwell", 0.2),
)

#: Guards the logarithm when a room's true probability is zero.
FLOOR = 1e-9


@dataclass(frozen=True)
class OracleModel:
    """How good the language model is at guessing, as three numbers.

    Attributes:
        name: The regime's name, for grouping results.
        skill: How much of the truth reaches the belief, in log space. ``1.0``
            is a perfect oracle, ``0.0`` knows nothing, and negative values are
            confidently wrong.
        noise: Standard deviation of the per-room error added on top, in log
            space. Even a skilled oracle is not exact.
        temperature: How peaked the reported distribution is. Above one it
            flattens towards uniform; below one it sharpens, which makes the
            oracle more confident without making it more right.
        decoy_mass: How much probability to move onto one wrong room that is
            far from the entrance, after the rest of the belief is formed.
            This is a *different* way to be wrong from low skill, and it is
            the one the paper actually tests: its misleading prior is a
            mixture of two peaks, one on the target and a second far away
            (p.13). An oracle can be largely right and still send the robot on
            one expensive detour, which is what a language model confidently
            naming the wrong room looks like.
    """

    name: str
    skill: float
    noise: float
    temperature: float = 1.0
    decoy_mass: float = 0.0


#: The regimes the benchmark sweeps. The first is a control -- a planner given
#: the truth -- and the last is the case the paper cares about and the one a
#: language model reasoning about an unfamiliar building will actually produce.
ORACLE_MODELS = (
    OracleModel("perfect", skill=1.0, noise=0.0),
    OracleModel("accurate", skill=1.0, noise=0.45),
    OracleModel("noisy", skill=0.6, noise=1.0),
    OracleModel("uninformative", skill=0.0, noise=0.4),
    OracleModel("decoy", skill=1.0, noise=0.45, decoy_mass=0.45),
    OracleModel("adversarial", skill=-0.8, noise=0.5),
)


@dataclass(frozen=True)
class Belief:
    """The truth, and what the oracle said about it.

    Attributes:
        kinds: Each room's semantic kind, as the oracle would label it.
        truth: The real probability the object is in each room. Sums to one.
        belief: What the oracle reported. Sums to one. This is what a planner
            is given.
        model: Which regime produced the belief.
    """

    kinds: Tuple[str, ...]
    truth: Tuple[float, ...]
    belief: Tuple[float, ...]
    model: OracleModel

    @property
    def true_room(self):
        # type: () -> int
        """The room the truth most favours -- for labelling a drawing only."""
        return max(range(len(self.truth)), key=lambda i: self.truth[i])

    def agreement(self):
        # type: () -> float
        """How much belief mass sits where the truth is, in ``[0, 1]``.

        The overlap of the two distributions. One means the oracle is exactly
        right; near zero means it is pointing somewhere else entirely. Reported
        alongside every result so a reader can see *why* a planner did well
        rather than only that it did.
        """
        return sum(min(t, b) for t, b in zip(self.truth, self.belief))


def make_belief(n_rooms, model, rng, distance=None, start=None,
                concentration=1.0):
    # type: (int, OracleModel, random.Random, object, object, float) -> Belief
    """Assign room kinds, derive the truth from them, then blur it into a belief.

    Args:
        n_rooms: How many rooms the building has.
        model: The oracle regime to simulate.
        rng: Randomness, so a scenario is reproducible from its seed.
        distance: Walking distances, needed only when the regime plants a
            decoy -- the decoy has to be somewhere expensive to reach, or it
            costs nothing to fall for.
        start: Index of the entrance in that matrix.
        concentration: How peaked the *truth* is. One leaves the room-kind
            weights as they are; above one the object is far more likely to be
            in the few most plausible rooms; below one it could be nearly
            anywhere.

            This is a property of the world rather than of the oracle, and it
            sets a ceiling on what any planner can win. If the object really is
            equally likely everywhere then no ordering beats any other, however
            perfect the belief -- so it is worth varying on its own axis, and
            not confusing with how good the oracle is.

    Returns:
        The pair of distributions.
    """
    kinds = [rng.choice(ROOM_KINDS) for _ in range(n_rooms)]
    weights = [(weight * rng.uniform(0.6, 1.4)) ** concentration
               for _, weight in kinds]
    truth = _normalise(weights)

    scores = []                                 # type: List[float]
    for probability in truth:
        scores.append(model.skill * math.log(probability + FLOOR)
                      + rng.gauss(0.0, model.noise))
    belief = _softmax(scores, model.temperature)
    if model.decoy_mass > 0.0:
        belief = _plant_decoy(belief, truth, model.decoy_mass, distance, start)
    return Belief(kinds=tuple(kind for kind, _ in kinds),
                  truth=truth, belief=belief, model=model)


def _plant_decoy(belief, truth, mass, distance, start):
    # type: (Sequence[float], Sequence[float], float, object, object) -> Tuple[float, ...]
    """Move probability onto one wrong room that is a long way off.

    The room chosen is the one furthest from the entrance among those the truth
    thinks unlikely -- an expensive mistake rather than a cheap one. The rest
    of the belief is scaled down to make room, so it stays a distribution and
    stays otherwise well-informed. That is the point: this oracle is not
    stupid, it is confidently wrong about one thing.
    """
    count = len(belief)
    unlikely = sorted(range(count), key=lambda r: truth[r])[:max(1, count // 2)]
    if distance is None or start is None:
        decoy = unlikely[0]
    else:
        decoy = max(unlikely, key=lambda r: distance[start][r])
    scaled = [value * (1.0 - mass) for value in belief]
    scaled[decoy] += mass
    return _normalise(scaled)


def sample_true_room(belief, rng):
    # type: (Belief, random.Random) -> int
    """Draw where the object actually is, from the truth.

    Only needed for a narrative walk-through; the headline metrics integrate
    over the whole truth distribution exactly rather than sampling it, which
    removes Monte-Carlo noise from every number the benchmark reports.
    """
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(belief.truth):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(belief.truth) - 1


# -- small maths ----------------------------------------------------------

def _normalise(weights):
    # type: (Sequence[float]) -> Tuple[float, ...]
    """Scale non-negative weights to sum to one."""
    total = float(sum(weights))
    if total <= 0.0:
        uniform = 1.0 / len(weights)
        return tuple(uniform for _ in weights)
    return tuple(float(weight) / total for weight in weights)


def _softmax(scores, temperature):
    # type: (Sequence[float], float) -> Tuple[float, ...]
    """Exponentiate and normalise, shifted for numerical safety."""
    scaled = [score / temperature for score in scores]
    ceiling = max(scaled)
    exponentiated = [math.exp(score - ceiling) for score in scaled]
    return _normalise(exponentiated)
