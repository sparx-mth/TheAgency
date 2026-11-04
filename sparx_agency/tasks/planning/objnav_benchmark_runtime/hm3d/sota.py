"""The published HM3D ObjectNav numbers we compare against, and their caveats.

Every row is **transcribed from a paper**, never measured here. Each carries the
table it was printed in, because two papers print different numbers for the same
baseline on the same split -- L3MVN's HM3D-v1 SR is 50.4 in ApexNav's Table I
and 48.7 in SG-Nav's Table 1 -- and a figure whose source is not recorded cannot
be checked later.

The protocols behind these numbers are **not identical**, and the differences
all push in the same direction:

* **Success radius.** habitat-lab's ``objectnav_hm3d`` uses 0.1 m to the nearest
  goal view point. SG-Nav's released HM3D config keeps 0.1; ApexNav's released
  configs set 0.2, which can only raise SR and SPL, never lower them. Our own
  run is scored at 0.1 and re-scored at 0.2 so both rows exist.
* **Navmesh.** Not a difference, despite the habitat-lab version gap: the
  recompute lives in **habitat-sim**, whose ``Simulator._config_pathfinder``
  rebuilds the mesh at the agent's radius and height whenever the shipped one
  disagrees. Both papers run a habitat-sim that does this, and so do we.
Methods that both papers evaluate only on other benchmarks are simply absent
here -- CoW, SemExp and PONI report MP3D or Gibson and no HM3D number, and a
table row is not the place to park a figure from a different benchmark.

* **Episode set.** ApexNav and SG-Nav run the full published val splits
  (2000 for v1, 1000 for v2). OSG runs 400 self-sampled episodes with no
  published ids, no step cap, RGB only and no STOP action, so its HM3D row is
  recorded for context and is **not** a like-for-like comparison.

Python 3.8 syntax, standard library only; no numpy, no simulator.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult

APEXNAV = "arXiv:2504.14478v3, Table I, p.6"
SGNAV = "arXiv:2410.08189v1, Table 1, p.7"
SGNAV_ABLATION = "arXiv:2410.08189v1, Table 2, p.8"
OSG = "arXiv:2508.04678v1, Table 1, p.12"

#: ApexNav's own disclosure of what it measured rather than quoted (p.6):
#: "As only InstructNav reported HM3Dv2 results, we re-evaluated several
#: open-source methods under our settings for fairness."
APEXNAV_MEASURED = "re-evaluated by ApexNav's authors under ApexNav's settings"
APEXNAV_RADIUS = "success radius 0.2 m (habitat-lab's objectnav_hm3d uses 0.1 m)"
SGNAV_RADIUS = "success radius 0.1 m, habitat-lab 0.2.1"

#: ApexNav's re-evaluation disclosure covers **HM3D-v2 only**. Its HM3D-v1
#: baseline rows are quotes from the originating papers, produced under those
#: authors' own settings -- several of them at 0.1 m, not ApexNav's 0.2 m. So a
#: v1 row is not "ApexNav's number for that method"; it is ApexNav's citation.
APEXNAV_QUOTED = ("quoted by ApexNav from the originating paper; the success "
                  "radius and stack behind it are those authors', and ApexNav "
                  "does not state them")

#: ``(method, SR %, SPL %, SoftSPL % or None, source, notes)``, in the order the
#: paper prints them. Percentages, exactly as printed; converted below.
HM3D_V1 = (
    ("ZSON", 25.5, 12.6, None, APEXNAV, "not zero-shot; " + APEXNAV_QUOTED),
    ("ESC", 39.2, 22.3, None, APEXNAV, APEXNAV_QUOTED),
    ("L3MVN", 50.4, 23.1, None, APEXNAV, APEXNAV_QUOTED),
    ("OpenFMNav", 54.9, 24.4, None, APEXNAV, APEXNAV_QUOTED),
    ("VLFM", 52.5, 30.4, None, APEXNAV,
     "ApexNav marks VLFM's zero-shot claim as disputed; " + APEXNAV_QUOTED),
    ("VLFM* (shortest-path planner)", 50.9, 23.6, None, APEXNAV,
     APEXNAV_MEASURED + "; " + APEXNAV_RADIUS),
    ("TriHelper", 56.5, 25.3, None, APEXNAV, APEXNAV_QUOTED),
    ("SG-Nav (ApexNav's quote of SG-Nav-GPT)", 54.0, 24.9, None, APEXNAV,
     APEXNAV_QUOTED + "; the same figure appears below from its own paper"),
    ("ApexNav", 59.6, 33.0, None, APEXNAV, "the paper's own result; " + APEXNAV_RADIUS),
    ("ProcTHOR", 54.4, 31.8, None, SGNAV, "trained on ObjectNav; " + SGNAV_RADIUS),
    ("ProcTHOR-ZS", 13.2, 7.7, None, SGNAV, SGNAV_RADIUS),
    ("L3MVN (SG-Nav's table)", 48.7, 23.0, None, SGNAV,
     "same method, different number from ApexNav's 50.4/23.1; " + SGNAV_RADIUS),
    ("OpenFMNav (SG-Nav's table)", 52.5, 24.1, None, SGNAV, SGNAV_RADIUS),
    ("VLFM (SG-Nav's table)", 52.4, 30.3, None, SGNAV, SGNAV_RADIUS),
    ("SG-Nav-LLaMA", 53.9, 24.8, 33.8, SGNAV,
     "SoftSPL from " + SGNAV_ABLATION + "; " + SGNAV_RADIUS),
    ("SG-Nav-GPT", 54.0, 24.9, None, SGNAV, "the paper's own result; " + SGNAV_RADIUS),
)

HM3D_V2 = (
    ("L3MVN", 36.3, 15.7, None, APEXNAV, APEXNAV_MEASURED + "; " + APEXNAV_RADIUS),
    ("InstructNav", 58.0, 20.9, None, APEXNAV,
     "the only HM3D-v2 number ApexNav quotes rather than re-runs; " + APEXNAV_QUOTED),
    ("VLFM", 63.6, 32.5, None, APEXNAV, APEXNAV_MEASURED + "; " + APEXNAV_RADIUS),
    ("VLFM* (shortest-path planner)", 56.9, 27.5, None, APEXNAV,
     APEXNAV_MEASURED + "; " + APEXNAV_RADIUS),
    ("SG-Nav", 49.6, 25.5, None, APEXNAV, APEXNAV_MEASURED + "; " + APEXNAV_RADIUS),
    ("ApexNav", 76.2, 38.0, None, APEXNAV, "the paper's own result; " + APEXNAV_RADIUS),
)

#: OSG's HM3D rows. Recorded, but excluded from the default comparison: a
#: 400-episode self-sampled subset of HM3D-Semantics v0.1 with no published
#: episode ids, no step cap, RGB-only observations, velocity commands and no
#: STOP action. Its DTG is defined only as "average distance of the agent from
#: the goal instance at the end of each episode", with no statement of geodesic
#: versus Euclidean and no treatment of unreachable goals.
OSG_CONTEXT = (
    ("FBE-GT (OSG's protocol)", 62.6, 31.0, 2.677),
    ("LGX-GT (OSG's protocol)", 27.5, 8.0, 5.078),
    ("LFG-GT (OSG's protocol)", 67.5, 38.9, 2.411),
    ("OSG-Nav-GT (OSG's protocol)", 77.5, 38.0, 1.702),
    ("OSG-Nav (OSG's protocol)", 69.3, 28.3, 2.338),
)

OSG_NOTE = ("400 self-sampled HM3D-Semantics v0.1 episodes, no published ids, no "
            "step cap, RGB only, velocity commands, no STOP action, DTG "
            "under-specified -- context, NOT a like-for-like comparison")


def reported_results(benchmark, split="val", include_osg=False):
    """The paper rows for one HM3D benchmark key, as :class:`ReportedResult`.

    Args:
        benchmark: ``"hm3d_v1"`` or ``"hm3d_v2"``.
        split: The split the run is on; the rows are validation numbers.
        include_osg: Also emit OSG's rows. They are on a different protocol and
            are off by default so a table cannot imply a comparison the papers
            do not support. Only HM3D-v1 has OSG rows (it is v0.1 scenes).

    Returns:
        A tuple of :class:`ReportedResult`, in table order.

    Raises:
        KeyError: An unknown benchmark key.
    """
    rows = {"hm3d_v1": HM3D_V1, "hm3d_v2": HM3D_V2}
    if benchmark not in rows:
        raise KeyError("No published HM3D numbers for %r; known: %s"
                       % (benchmark, ", ".join(sorted(rows))))
    results = [ReportedResult(method, benchmark, split, sr / 100.0, spl / 100.0,
                              source, soft_spl=None if soft is None else soft / 100.0,
                              notes=notes)
               for method, sr, spl, soft, source, notes in rows[benchmark]]
    if include_osg and benchmark == "hm3d_v1":
        results.extend(ReportedResult(method, benchmark, split, sr / 100.0,
                                      spl / 100.0, OSG, distance_to_goal_m=dtg,
                                      notes=OSG_NOTE)
                       for method, sr, spl, dtg in OSG_CONTEXT)
    return tuple(results)
