"""What the papers report on RoboTHOR ObjectNav, and which rows may sit together.

Every number here was read from the table that first printed it, not copied
from a downstream paper's comparison. That matters more on RoboTHOR than on
any other ObjectNav benchmark, because the copied chain carries a specific,
widespread error:

    **ESC's Table 1 and SG-Nav's Table 1 both place ProcTHOR's 65.2/28.8 and
    ProcTHOR-ZS's 55.0/23.7 in a column headed "RoboTHOR", beside their own
    1,800-episode validation numbers. Those two ProcTHOR figures are test-set
    numbers, measured on the 2,040-episode challenge test split.**

ProcTHOR's own Table 2 is the giveaway: its EmbCLIP row reads 47.0/0.200,
which is exactly the archived leaderboard's ``test_success`` 0.4701 and
``test_spl`` 0.2004, while EmbCLIP's *validation* figures are 52.2/26.0. The
Embodied AI workshop retrospective independently calls 0.2884 "test SPL". So
those rows compare a validation agent against a test-set agent, and the
difference is not small.

This module therefore keeps the groups apart and only exposes the validation
ones to :func:`~sparx_agency.tasks.planning.objnav_benchmark.comparison.comparison_table`,
which already refuses a row from another split. :data:`TEST_SPLIT_RESULTS` is
kept so the test numbers can be quoted as test numbers, deliberately, rather
than being silently unavailable and quietly re-copied from a secondary source.

Two further cautions, both recorded in the rows' ``notes``:

* **ESC does not reproduce.** ESC reports 38.1/22.2; two later papers, both
  using what they call the official implementation, report 34.5/18.2. ESC's
  code is not public ("cannot be publicly released due to company policy"), so
  the discrepancy cannot be resolved. ESC is the anchor baseline in nearly
  every later RoboTHOR table, so this uncertainty propagates.
* **Three rows exist only as third-party re-runs.** L3MVN, VLFM and OpenFMNav
  never published a RoboTHOR number themselves; their RoboTHOR figures appear
  only inside SG-Nav's Table 1.

TF and NM are **not metrics**. They are the method-property flags of OSG
Navigator's Table 2 ("TF / NM denote training-free/non-metric approaches"):
training-free, and non-metric. They cannot be computed from episode telemetry
and are hand-authored per method. :data:`METHOD_PROPERTIES` carries them, and
:func:`properties_row` renders ours -- which is training-free (no ObjectNav
training of any kind) but **metric**: the search builds a 2.5-D occupancy map
and plans through it with weighted A*.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult

BENCHMARK = "robothor"
VAL = "val"

_THIRD_PARTY = ("RoboTHOR figure published only inside SG-Nav's Table 1; the "
                "method's own paper reports no RoboTHOR result")

#: Published results on the 1,800-episode RoboTHOR **validation** split.
#: Mutually comparable: same episodes, same rule, same budget.
VALIDATION_RESULTS = (
    ReportedResult(
        method="CoW (OWL B/32 + post)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.267, spl=0.169,
        source="arXiv:2203.10421 Table 1, p.6",
        notes="The row downstream papers cite as 'CoW'. CoW's own best is "
              "OWL B/16 + post at 27.5/17.2. CoW uses a four-action space "
              "with no LookUp/LookDown, so it is locked at the episode's "
              "initial 30-degree downward horizon throughout"),
    ReportedResult(
        method="CoW (OWL B/16 + post)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.275, spl=0.172,
        source="arXiv:2203.10421 Table 1, p.6",
        notes="CoW's best breed; four-action space as above"),
    ReportedResult(
        method="LGX (YOLO -> LLM + GLIP)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.350, spl=0.219,
        source="arXiv:2303.03480 Table III",
        notes="No public code"),
    ReportedResult(
        method="GoW (FBE + GLIP)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.332, spl=0.203,
        source="arXiv:2303.03480 Table III",
        notes="LGX's own frontier baseline"),
    ReportedResult(
        method="VLTNet", benchmark=BENCHMARK, split=VAL,
        success_rate=0.332, spl=0.171,
        source="arXiv:2410.18570 Table 1",
        notes="The paper calls its efficiency metric SWPL"),
    ReportedResult(
        method="ESC", benchmark=BENCHMARK, split=VAL,
        success_rate=0.381, spl=0.222,
        source="arXiv:2301.13166 Table 1, p.6",
        notes="DOES NOT REPRODUCE: two independent papers using what they "
              "describe as the official implementation report 34.5/18.2 "
              "(arXiv:2410.21926 Table 2, arXiv:2410.21037 Table 1). ESC's "
              "code is not public. ESC also notes RoboTHOR gives its agent no "
              "GPS, unlike its own MP3D/HM3D configurations"),
    ReportedResult(
        method="ESC (independent reproduction)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.345, spl=0.182,
        source="arXiv:2410.21926 Table 2",
        notes="Marked in that table as reproduced with the official "
              "implementation; 3.6 SR and 4.0 SPL below ESC's own figure"),
    ReportedResult(
        method="DEFA+CDM", benchmark=BENCHMARK, split=VAL,
        success_rate=0.368, spl=0.223,
        source="arXiv:2410.21037 Table 1",
        notes="That paper's Section 5.1 states a 0.1 m success threshold "
              "where RoboTHOR's is 1.0 m; almost certainly a typo, but if "
              "read literally the row is not comparable"),
    ReportedResult(
        method="Unlu et al.", benchmark=BENCHMARK, split=VAL,
        success_rate=0.352, spl=0.183,
        source="arXiv:2410.21926 Table 2"),
    ReportedResult(
        method="L3MVN", benchmark=BENCHMARK, split=VAL,
        success_rate=0.412, spl=0.225,
        source="SG-Nav, arXiv:2410.08189 Table 1, p.7", notes=_THIRD_PARTY),
    ReportedResult(
        method="VLFM", benchmark=BENCHMARK, split=VAL,
        success_rate=0.423, spl=0.230,
        source="SG-Nav, arXiv:2410.08189 Table 1, p.7", notes=_THIRD_PARTY),
    ReportedResult(
        method="OpenFMNav", benchmark=BENCHMARK, split=VAL,
        success_rate=0.441, spl=0.233,
        source="SG-Nav, arXiv:2410.08189 Table 1, p.7", notes=_THIRD_PARTY),
    ReportedResult(
        method="SG-Nav (LLaMA)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.473, spl=0.237,
        source="arXiv:2410.08189 Table 1, p.7",
        notes="Released code covers MP3D only; the RoboTHOR figure is not "
              "reproducible from the release. The paper does not say whether "
              "all 1,800 episodes were run"),
    ReportedResult(
        method="SG-Nav (GPT-4)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.475, spl=0.240,
        source="arXiv:2410.08189 Table 1, p.7", notes="As above"),
    ReportedResult(
        method="CogNav", benchmark=BENCHMARK, split=VAL,
        success_rate=0.546, spl=0.243,
        source="arXiv:2412.10439 Table 1",
        notes="Strongest published zero-shot RoboTHOR result on both SR and "
              "SPL. Released code covers HM3D only"),
    ReportedResult(
        method="EmbCLIP (trained, 200M steps)", benchmark=BENCHMARK, split=VAL,
        success_rate=0.522, spl=0.260,
        source="RoboTHOR ObjectNav leaderboard, archived 2022-04-22, "
               "val_success / val_spl",
        notes="TRAINED, not zero-shot, and measured on these same 1,800 "
              "episodes -- the honest trained reference. No ObjectNav paper "
              "cites it, because they all compare against ProcTHOR's test "
              "numbers instead. Zero-shot has overtaken it on SR but not SPL"),
)

#: Published results on the 2,040-episode challenge **test** split. Kept
#: separate on purpose: these are the rows the literature mislabels as
#: RoboTHOR validation. Never pass them to ``comparison_table`` beside a
#: validation run -- it refuses them, which is the point.
TEST_SPLIT_RESULTS = (
    ReportedResult(
        method="EmbCLIP (2021 challenge winner, trained)", benchmark=BENCHMARK,
        split="test", success_rate=0.4701, spl=0.2004,
        source="RoboTHOR ObjectNav leaderboard rank 1, archived 2022-04-22; "
               "also arXiv:2111.09888 Table 1"),
    ReportedResult(
        method="ProcTHOR-ZS (pretrained, no RoboTHOR data)", benchmark=BENCHMARK,
        split="test", success_rate=0.550, spl=0.237,
        source="arXiv:2206.06994 Table 2, p.9",
        notes="Scene-zero-shot but navigation-trained. Widely reprinted in "
              "'RoboTHOR' columns beside validation numbers"),
    ReportedResult(
        method="ProcTHOR + fine-tune (trained)", benchmark=BENCHMARK,
        split="test", success_rate=0.652, spl=0.2884,
        source="arXiv:2206.06994 Table 2, p.9",
        notes="Fine-tuned 29M steps on the 60 RoboTHOR training scenes. "
              "Corroborated as a test figure by arXiv:2210.06849 3.1.3. CoW's "
              "Table 1 quotes 66.4/27.4 for the same agent, unexplained"),
)

#: ``method -> (training-free, non-metric)``, OSG Navigator's Table 2 flags.
#: Not metrics: they describe the method, and are set by hand.
METHOD_PROPERTIES = {
    "CoW (OWL B/32 + post)": (True, False),
    "CoW (OWL B/16 + post)": (True, False),
    "LGX (YOLO -> LLM + GLIP)": (True, False),
    "GoW (FBE + GLIP)": (True, False),
    "VLTNet": (True, False),
    "ESC": (True, False),
    "ESC (independent reproduction)": (True, False),
    "DEFA+CDM": (True, False),
    "Unlu et al.": (True, False),
    "L3MVN": (True, False),
    "VLFM": (True, False),
    "OpenFMNav": (True, False),
    "SG-Nav (LLaMA)": (True, False),
    "SG-Nav (GPT-4)": (True, False),
    "CogNav": (True, False),
    "EmbCLIP (trained, 200M steps)": (False, False),
}

#: Ours. Training-free -- no ObjectNav training of any kind, the detector and
#: the LLM are used off the shelf. **Metric**: the search integrates depth into
#: a robot-height 2.5-D occupancy grid and plans through it with weighted A*,
#: which is exactly what OSG Navigator's NM column means by metric.
OUR_PROPERTIES = (True, False)


def properties_row(method, properties=None):
    """The TF / NM cells for one method, as OSG Navigator prints them."""
    flags = (METHOD_PROPERTIES.get(method) if properties is None
             else properties)
    if flags is None:
        return ("?", "?")
    return tuple("yes" if flag else "no" for flag in flags)


def validation_results(include_trained=True):
    """The validation rows, optionally zero-shot only."""
    if include_trained:
        return VALIDATION_RESULTS
    return tuple(r for r in VALIDATION_RESULTS
                 if METHOD_PROPERTIES.get(r.method, (False, False))[0])
