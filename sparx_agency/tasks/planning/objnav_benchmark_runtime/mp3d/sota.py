"""The published MP3D ObjectNav numbers we compare against, and their caveats.

Every row is **transcribed from a paper**, never measured here. Each carries the
table it was printed in, because two papers print different numbers for the same
baseline on the same split -- VLFM's MP3D row is 36.2/15.9 in SG-Nav's Table 1
and 36.4/17.5 in ApexNav's Table I -- and a figure whose source is not recorded
cannot be checked later. Both are kept, under their own sources, rather than
averaged or picked.

The protocols behind these numbers are **not identical**:

* **Success radius.** habitat-lab's ``objectnav_mp3d`` uses 0.1 m to the nearest
  goal view point. SG-Nav's released config keeps 0.1; ApexNav's released
  configs set 0.2, which can only raise SR and SPL, never lower them. Our own
  run is scored at 0.1 and re-scored at 0.2 so both rows exist (``report.py``).
* **Episode set.** Both papers state the same split: the 2021 Habitat
  challenge's MP3D validation, 11 scenes, 21 goal categories, 2,195 episodes.
* **Navmesh.** Not a difference: habitat-sim recomputes the shipped mesh at the
  agent's radius and height for every habitat-lab run, and so do we.

Python 3.8 syntax, standard library only; no numpy, no simulator.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult

SGNAV = "arXiv:2410.08189v1, Table 1, p.7"
APEXNAV = "arXiv:2504.14478v1, Table I, p.6"

SGNAV_RADIUS = "success radius 0.1 m, habitat-lab 0.2.1"
APEXNAV_RADIUS = "success radius 0.2 m (habitat-lab's objectnav_mp3d uses 0.1 m)"
#: SG-Nav's MP3D baseline rows are quotes from the originating papers (its
#: Table 1 marks its own rows only); the stacks and radii behind the quoted
#: rows are those authors', and SG-Nav does not restate them.
SGNAV_QUOTED = ("quoted by SG-Nav from the originating paper; the success "
                "radius and stack behind it are those authors'")

#: ``(method, SR %, SPL %, SoftSPL % or None, source, notes)``, in the order the
#: papers print them. Percentages, exactly as printed; converted below.
MP3D_V1 = (
    ("SemExp", 36.0, 14.4, None, SGNAV, "trained ObjectNav policy; not zero-shot; " + SGNAV_QUOTED),
    ("PONI", 31.8, 12.1, None, SGNAV, "trained ObjectNav policy; not zero-shot; " + SGNAV_QUOTED),
    ("ZSON", 15.3, 4.8, None, SGNAV, "unsupervised pretraining; not zero-shot; " + SGNAV_QUOTED),
    ("CoW", 7.4, 3.7, None, SGNAV, "zero-shot; " + SGNAV_QUOTED),
    ("ESC", 28.7, 14.2, None, SGNAV, "zero-shot; " + SGNAV_QUOTED),
    ("L3MVN", 34.9, 14.5, None, SGNAV, "zero-shot; " + SGNAV_QUOTED),
    ("OpenFMNav", 37.2, 15.7, None, SGNAV, "zero-shot; " + SGNAV_QUOTED),
    ("VLFM (SG-Nav's table)", 36.2, 15.9, None, SGNAV,
     "zero-shot; ApexNav prints 36.4/17.5 for the same method; " + SGNAV_QUOTED),
    ("SG-Nav-LLaMA", 40.1, 16.0, None, SGNAV, "the paper's own result; " + SGNAV_RADIUS),
    ("SG-Nav-GPT", 40.2, 16.0, None, SGNAV, "the paper's own result; " + SGNAV_RADIUS),
    ("VLFM (ApexNav's table)", 36.4, 17.5, None, APEXNAV,
     "zero-shot; SG-Nav prints 36.2/15.9 for the same method; " + APEXNAV_RADIUS),
    ("VLFM* (shortest-path planner)", 32.5, 15.9, None, APEXNAV,
     "PointNav module replaced by a shortest-path planner; " + APEXNAV_RADIUS),
    ("ApexNav", 39.2, 17.8, None, APEXNAV, "the paper's own result; " + APEXNAV_RADIUS),
)


def reported_results(benchmark, split="val"):
    """The paper rows for the MP3D benchmark key, as :class:`ReportedResult`.

    Args:
        benchmark: ``"mp3d_v1"``.
        split: The split the run is on; the rows are validation numbers.

    Raises:
        KeyError: An unknown benchmark key.
    """
    rows = {"mp3d_v1": MP3D_V1}
    if benchmark not in rows:
        raise KeyError("No published MP3D numbers for %r; known: %s"
                       % (benchmark, ", ".join(sorted(rows))))
    return tuple(ReportedResult(method, benchmark, split, sr / 100.0, spl / 100.0,
                                source, soft_spl=None if soft is None else soft / 100.0,
                                notes=notes)
                 for method, sr, spl, soft, source, notes in rows[benchmark])
