"""How good does the oracle have to be before using it is worth anything?

The factorial sweep answers "which planner wins" for six named oracle regimes.
This answers the question underneath that, which is the one that actually
decides whether to build the thing: **there is a level of oracle quality below
which paying attention to the belief makes the search worse, and it is useful
to know where that level is.**

The experiment is a continuous sweep of the oracle's skill from confidently
wrong to perfect, holding everything else fixed, plotting the expected distance
of a planner that uses the belief against one that ignores it. Where the two
curves cross is the break-even point, expressed in the one quantity that can be
measured on a real oracle without knowing the answer in advance: the *overlap*
between what it says and where things really are.

    python -m sparx_agency.tasks.planning.routing_benchmark.crossover

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import zlib
from typing import Dict, List, Sequence

from sparx_agency.tasks.planning.routing_benchmark.buildings import (
    TOPOLOGIES,
    generate_building,
)
from sparx_agency.tasks.planning.routing_benchmark.metrics import (
    clairvoyant_distance,
    evaluate,
)
from sparx_agency.tasks.planning.routing_benchmark.oracle import (
    OracleModel,
    make_belief,
)
from sparx_agency.tasks.planning.routing_benchmark.planners import (
    plan_greedy,
    plan_nearest_2opt,
    plan_rpt_star,
)

#: Where results go by default.
DEFAULT_OUT = pathlib.Path.home() / "rpt_star_benchmark" / "crossover.json"

#: The oracles swept, as ``(skill, noise)`` pairs. Both are varied together on
#: purpose: skill decides whether the belief points the right way, noise decides
#: how sharply it points at all, and a planner benefits only when both are
#: favourable. Holding noise fixed would cap the best achievable oracle well
#: short of a real one and understate the whole method. The pairs are chosen to
#: spread the resulting agreement roughly evenly from 0.2 to 1.0.
ORACLES = (
    (-1.00, 0.50), (-0.75, 0.50), (-0.50, 0.50), (-0.25, 0.50),
    (0.00, 0.50), (0.25, 0.50), (0.40, 0.45), (0.55, 0.40),
    (0.70, 0.30), (0.85, 0.20), (1.00, 0.10), (1.00, 0.00),
)

#: Repeats per skill level. Larger than the factorial sweep's, because here the
#: quantity being estimated is a curve and its shape matters more than any one
#: point.
REPEATS = 80

#: Rooms per building. Small enough that exact RPT* always finishes, so the
#: curve measures the belief and not the budget.
N_ROOMS = 11


def run(oracles=ORACLES, repeats=REPEATS, n_rooms=N_ROOMS):
    # type: (Sequence[tuple], int, int) -> Dict
    """Sweep oracle quality and record what each planner achieves.

    Args:
        oracles: ``(skill, noise)`` pairs to try.
        repeats: Buildings and beliefs per level.
        n_rooms: Rooms per building.

    Returns:
        One entry per oracle, with the median outcome of each planner and the
        median agreement that oracle produced. Agreement, not skill, is the
        x-axis worth plotting: it is the one quantity measurable on a real
        oracle by comparing what it said against where things turned out to be.
    """
    points = []                                 # type: List[Dict]
    for skill, noise in oracles:
        model = OracleModel("skill%+.2f" % skill, skill=skill, noise=noise)
        gathered = {"rpt_star": [], "nearest_2opt": [], "greedy": [],
                    "agreement": [], "clairvoyant": []}
        # PAIRED, and that is the point: both planners see the same building
        # and the same belief, so the per-instance ratio cancels out how hard
        # that particular instance happened to be. Comparing two medians of
        # separately-varying quantities would need many times the repeats to
        # resolve the same effect.
        ratio_vs_nearest = []                   # type: List[float]
        ratio_vs_greedy = []                    # type: List[float]
        for repeat in range(repeats):
            stamp = "cross-%.3f-%.3f-%d" % (skill, noise, repeat)
            rng = random.Random(zlib.crc32(stamp.encode("utf-8")))
            topology = TOPOLOGIES[repeat % len(TOPOLOGIES)]
            building = generate_building(topology, n_rooms, rng)
            belief = make_belief(n_rooms, model, rng,
                                 distance=building.distance, start=n_rooms)
            gathered["agreement"].append(belief.agreement())
            gathered["clairvoyant"].append(clairvoyant_distance(
                belief.truth, building.distance, n_rooms))
            scored = {}
            for name, planner in (("rpt_star", plan_rpt_star),
                                  ("nearest_2opt", plan_nearest_2opt),
                                  ("greedy", plan_greedy)):
                plan = planner(belief.belief, building.distance)
                scored[name] = evaluate(
                    plan.order, belief.truth, building.distance).distance
                gathered[name].append(scored[name])
            if scored["nearest_2opt"] > 0.0:
                ratio_vs_nearest.append(scored["rpt_star"]
                                        / scored["nearest_2opt"])
            if scored["greedy"] > 0.0:
                ratio_vs_greedy.append(scored["rpt_star"] / scored["greedy"])
        points.append({
            "skill": skill,
            "noise": noise,
            "agreement": statistics.median(gathered["agreement"]),
            "clairvoyant_m": statistics.median(gathered["clairvoyant"]),
            "rpt_star_m": statistics.median(gathered["rpt_star"]),
            "nearest_2opt_m": statistics.median(gathered["nearest_2opt"]),
            "greedy_m": statistics.median(gathered["greedy"]),
            # The headline: below 1.0, RPT* flies less than the planner that
            # ignores the belief entirely.
            "ratio_vs_nearest": statistics.median(ratio_vs_nearest),
            "ratio_vs_greedy": statistics.median(ratio_vs_greedy),
            "beats_nearest_pct": 100.0 * sum(1 for r in ratio_vs_nearest
                                             if r < 1.0) / len(ratio_vs_nearest),
        })
    return {"points": points, "repeats": repeats, "n_rooms": n_rooms,
            "break_even": _break_even(points)}


def _break_even(points):
    # type: (Sequence[Dict]) -> Dict
    """Where using the belief stops helping, by linear interpolation.

    Walks the curve from confidently-wrong upwards and reports the first place
    the belief-using planner overtakes the one that ignores it.

    Returns:
        The interpolated skill and agreement at the crossing, or ``None`` values
        if the curves never cross within the sweep.
    """
    for earlier, later in zip(points, points[1:]):
        before = earlier["ratio_vs_nearest"] - 1.0
        after = later["ratio_vs_nearest"] - 1.0
        if before > 0.0 >= after:
            span = before - after
            fraction = before / span if span else 0.0
            return {
                "skill": earlier["skill"]
                + fraction * (later["skill"] - earlier["skill"]),
                "agreement": earlier["agreement"]
                + fraction * (later["agreement"] - earlier["agreement"]),
            }
    return {"skill": None, "agreement": None}


#: How peaked the world is, for the second sweep. One is the default world;
#: four means the object is strongly concentrated in a few plausible rooms.
CONCENTRATIONS = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0)


def run_concentration(concentrations=CONCENTRATIONS, repeats=REPEATS,
                      n_rooms=N_ROOMS, skill=1.0, noise=0.15):
    # type: (Sequence[float], int, int, float, float) -> Dict
    """What a good oracle is worth, as the world becomes more predictable.

    The oracle is held near-perfect throughout, so this isolates the *other*
    ceiling on the method. Even a flawless belief buys nothing if the object is
    genuinely equally likely anywhere: there is no ordering to get right. What
    varies here is the world, not the knowledge of it.

    Returns:
        One entry per concentration, with the paired ratio against a planner
        that ignores the belief entirely.
    """
    model = OracleModel("sharp", skill=skill, noise=noise)
    points = []                                 # type: List[Dict]
    for concentration in concentrations:
        ratios = []                             # type: List[float]
        peaks = []                              # type: List[float]
        scores = {"rpt": [], "near": [], "greedy": []}
        for repeat in range(repeats):
            stamp = "conc-%.3f-%d" % (concentration, repeat)
            rng = random.Random(zlib.crc32(stamp.encode("utf-8")))
            topology = TOPOLOGIES[repeat % len(TOPOLOGIES)]
            building = generate_building(topology, n_rooms, rng)
            belief = make_belief(n_rooms, model, rng,
                                 distance=building.distance, start=n_rooms,
                                 concentration=concentration)
            peaks.append(max(belief.truth))
            one = {}
            for name, planner in (("rpt", plan_rpt_star),
                                  ("near", plan_nearest_2opt),
                                  ("greedy", plan_greedy)):
                plan = planner(belief.belief, building.distance)
                one[name] = evaluate(plan.order, belief.truth,
                                     building.distance).distance
                scores[name].append(one[name])
            if one["near"] > 0.0:
                ratios.append(one["rpt"] / one["near"])
        points.append({
            "concentration": concentration,
            "peak_true_prob": statistics.median(peaks),
            "rpt_star_m": statistics.median(scores["rpt"]),
            "nearest_2opt_m": statistics.median(scores["near"]),
            "greedy_m": statistics.median(scores["greedy"]),
            "ratio_vs_nearest": statistics.median(ratios),
            "beats_nearest_pct": 100.0 * sum(1 for r in ratios if r < 1.0)
            / len(ratios),
        })
    return {"points": points, "repeats": repeats, "n_rooms": n_rooms}


def main(argv=None):
    # type: (list) -> int
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    results = run()
    results["concentration"] = run_concentration()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("skill noise  agree     rpt*  nearest   greedy | rpt/near  wins%")
    for point in results["points"]:
        print("%+5.2f %5.2f  %5.2f  %7.1f  %7.1f  %7.1f | %8.3f  %4.0f%%"
              % (point["skill"], point["noise"], point["agreement"],
                 point["rpt_star_m"],
                 point["nearest_2opt_m"], point["greedy_m"],
                 point["ratio_vs_nearest"], point["beats_nearest_pct"]))
    crossing = results["break_even"]
    if crossing["agreement"] is None:
        print("\nthe curves never cross in this sweep")
    else:
        print("\nbreak-even: skill %+.2f, oracle agreement %.2f"
              % (crossing["skill"], crossing["agreement"]))
    print()
    print("peakP  concentr    rpt*  nearest   greedy | rpt/near  wins%")
    for point in results["concentration"]["points"]:
        print("%5.2f  %8.1f  %6.1f  %7.1f  %7.1f | %8.3f  %4.0f%%"
              % (point["peak_true_prob"], point["concentration"],
                 point["rpt_star_m"], point["nearest_2opt_m"],
                 point["greedy_m"], point["ratio_vs_nearest"],
                 point["beats_nearest_pct"]))
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
