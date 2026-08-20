"""Tests for the optimization report renderer.

The stats objects here are hand-rolled duck types rather than the real
``bench.latency.LatencyStats``: the report contract is "anything with
``.mean_ms`` and ``.hz``", and testing against a stub is what proves it.
"""
from __future__ import annotations

import json
import re

import pytest

from sparx_agency.tasks.common.trt_optimizer.bench.report import (
    ComponentRow,
    OptimizationReport,
    QualityRow,
    render_markdown,
    summarize,
    write_report,
)
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, Component, Verdict

REQUIRED_HEADINGS = (
    "## Hardware and build",
    "## Components",
    "## Deliberately not converted",
    "## Quality",
    "## Memory",
    "## Warnings",
)


class _Stats(object):
    """Minimal LatencyStats stand-in: mean latency plus derived throughput."""

    def __init__(self, mean_ms):
        self.mean_ms = float(mean_ms)
        self.n = 200

    @property
    def hz(self):
        return 1000.0 / self.mean_ms


def _rows():
    return [
        ComponentRow(name="model.denoiser", params=31_000_000,
                     cadence=Cadence.PER_STEP, calls_per_decision=20.0,
                     before_ms=24.0, after_ms=6.0, action="trt_fp16",
                     why="dominates the budget and exports clean"),
        ComponentRow(name="model.vision_tower", params=86_000_000,
                     cadence=Cadence.PER_FRAME, calls_per_decision=1.0,
                     before_ms=48.0, after_ms=12.0, action="trt_fp32",
                     why="precision-sensitive ViT trunk, strongly-typed FP32"),
        ComponentRow(name="model.language_head", params=1_300_000_000,
                     cadence=Cadence.PER_PLAN, calls_per_decision=0.125,
                     before_ms=9.0, after_ms=None, action="llm_runtime",
                     why="autoregressive KV-cache loop, never an ONNX graph"),
        ComponentRow(name="model.text_encoder", params=63_000_000,
                     cadence=Cadence.ONCE_PER_EPISODE, calls_per_decision=0.0,
                     before_ms=0.4, after_ms=None, action="cache_output",
                     why="instruction is fixed for the episode, cache it"),
        ComponentRow(name="model.tokenizer", params=0,
                     cadence=Cadence.ONCE_PER_EPISODE, calls_per_decision=0.0,
                     before_ms=0.1, after_ms=None, action="leave_in_torch",
                     why="0.1 percent of the budget; conversion cannot repay"),
    ]


def _report(quality=None, **kwargs):
    defaults = dict(
        model="navdp",
        target_tag="nvidiageforcertx_sm120",
        gpu_name="NVIDIA GeForce RTX 5070 Laptop",
        trt_version="11.1.0.106",
        precision="fp16",
        before=_Stats(82.6),
        after=_Stats(24.3),
        components=_rows(),
        quality=[
            QualityRow(metric="trajectory_l2", reference=0.0, measured=0.031,
                       threshold=0.05, passed=True, note="mean over 500 frames"),
            QualityRow(metric="cosine_sim", reference=1.0, measured=0.998,
                       threshold=0.995, passed=True, note=""),
        ] if quality is None else quality,
        memory={"required_bytes": 2_147_483_648,
                "available_bytes": 8_549_990_400,
                "engines": 3},
        warnings=["SM clock was not locked; rerun with nvidia-smi -lgc."],
        notes=["Baseline taken on a thermally settled device."],
    )
    defaults.update(kwargs)
    return OptimizationReport(**defaults)


def _find(report, name):
    """The row for one component name (the report exposes no lookup helper)."""
    return [row for row in report.components if row.name == name][0]


def _component_table_names(markdown):
    """Names in the order the component table lists them."""
    lines = markdown.splitlines()
    start = lines.index("## Components")
    names = []
    for line in lines[start:]:
        if line.startswith("| model."):
            names.append(line.split("|")[1].strip())
        elif names and not line.startswith("|"):
            break
    return names


def test_render_contains_every_required_section():
    markdown = render_markdown(_report())
    for heading in REQUIRED_HEADINGS:
        assert heading in markdown
    assert markdown.startswith("# TensorRT optimization report -- navdp")


def test_headline_reports_rates_and_pass():
    markdown = render_markdown(_report())
    headline = markdown.splitlines()[2]
    assert "**PASS**" in headline
    assert "12.1 Hz -> 41.2 Hz" in headline
    assert "3.40x" in headline
    assert "NOT ACCEPTED" not in markdown


def test_headline_says_failed_and_not_accepted_when_one_quality_row_fails():
    report = _report()
    report.quality[1].passed = False
    markdown = render_markdown(report)
    headline = markdown.splitlines()[2]
    assert "**FAILED**" in headline
    assert "**NOT ACCEPTED**" in headline
    assert "1 of 2 quality check(s) failed" in headline
    assert report.passed is False


def test_empty_quality_gate_is_not_a_pass():
    report = _report(quality=[])
    assert report.passed is False
    markdown = render_markdown(report)
    assert "**FAILED**" in markdown
    assert "unverified" in markdown


def test_not_converted_section_lists_every_kept_component_with_its_why():
    markdown = render_markdown(_report())
    section = markdown.split("## Deliberately not converted")[1]
    section = section.split("## Quality")[0]
    assert "model.tokenizer" in section
    assert "leave_in_torch" in section
    assert "0.1 percent of the budget; conversion cannot repay" in section
    assert "model.language_head" in section and "llm_runtime" in section
    assert "model.text_encoder" in section and "cache_output" in section
    assert "model.vision_tower" not in section


def test_not_converted_section_is_present_even_when_nothing_was_kept():
    rows = [r for r in _rows() if r.action.startswith("trt_")]
    markdown = render_markdown(_report(components=rows))
    assert "## Deliberately not converted" in markdown
    assert "Nothing was skipped" in markdown


def test_component_rows_sorted_by_before_ms_descending():
    markdown = render_markdown(_report())
    assert _component_table_names(markdown) == [
        "model.vision_tower", "model.denoiser", "model.language_head",
        "model.text_encoder", "model.tokenizer",
    ]


def test_unmeasured_component_sorts_last_and_has_no_share():
    rows = _rows()
    rows.append(ComponentRow(name="model.unprofiled", before_ms=None))
    report = _report(components=rows)
    markdown = render_markdown(report)
    assert _component_table_names(markdown)[-1] == "model.unprofiled"
    assert _find(report, "model.unprofiled").share_before is None


def test_shares_are_assigned_by_the_renderer_and_sum_to_one():
    report = _report()
    assert all(row.share_before is None for row in report.components)
    render_markdown(report)
    shares = [row.share_before for row in report.components]
    assert all(s is not None for s in shares)
    assert sum(shares) == pytest.approx(1.0)
    biggest = max(report.components, key=lambda r: r.share_before)
    assert biggest.name == "model.vision_tower"
    assert biggest.share_before == pytest.approx(48.0 / 81.5)


def test_component_and_report_speedups():
    report = _report()
    row = _find(report, "model.vision_tower")
    assert row.speedup == pytest.approx(4.0)
    assert _find(report, "model.tokenizer").speedup is None
    assert report.speedup == pytest.approx(82.6 / 24.3)


def test_speedup_is_none_for_a_zero_after_measurement():
    assert ComponentRow(name="x", before_ms=5.0, after_ms=0.0).speedup is None


def test_numbers_are_kept_to_three_significant_figures():
    markdown = render_markdown(_report())
    assert "| 4.00x" in markdown or "4.00x" in markdown
    assert not re.search(r"\d\.\d{4,}", markdown)


def test_memory_section_shows_required_available_and_headroom():
    markdown = render_markdown(_report())
    section = markdown.split("## Memory")[1].split("## Warnings")[0]
    assert "required" in section and "2048 MiB" in section
    assert "available" in section
    assert "headroom" in section and "fits" in section
    assert "engines" in section


def test_memory_section_flags_an_over_budget_build():
    report = _report(memory={"required_mib": 9000, "available_mib": 8151})
    section = render_markdown(report).split("## Memory")[1]
    assert "OVER BUDGET" in section


def test_as_dict_round_trips_through_json():
    report = _report()
    render_markdown(report)
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["model"] == "navdp"
    assert payload["passed"] is True
    assert payload["speedup"] == pytest.approx(82.6 / 24.3)
    assert payload["before"]["mean_ms"] == pytest.approx(82.6)
    assert payload["after"]["hz"] == pytest.approx(1000.0 / 24.3)
    assert len(payload["components"]) == 5
    kept = [c for c in payload["components"] if c["action"] == "leave_in_torch"]
    assert kept[0]["why"].startswith("0.1 percent")
    assert kept[0]["share_before"] == pytest.approx(0.1 / 81.5)
    assert payload["quality"][0]["passed"] is True
    assert payload["memory"]["engines"] == 3


def test_write_report_creates_both_files(tmp_path):
    out_dir = tmp_path / "engines" / "run01"
    md_path, json_path = write_report(_report(), out_dir, stem="navdp_report")
    assert md_path.name == "navdp_report.md" and md_path.is_file()
    assert json_path.name == "navdp_report.json" and json_path.is_file()
    assert "## Deliberately not converted" in md_path.read_text()
    assert json.loads(json_path.read_text())["target_tag"] == \
        "nvidiageforcertx_sm120"


def test_summarize_one_line_format():
    line = summarize(_report())
    assert line == ("3.40x (12.1 -> 41.2 Hz) on nvidiageforcertx_sm120, fp16, "
                    "quality PASS")
    assert "\n" not in line


def test_summarize_reports_a_failed_gate():
    report = _report()
    report.quality[0].passed = False
    assert summarize(report).endswith("quality FAIL")


def test_a_one_sided_report_raises_instead_of_rendering(tmp_path):
    report = _report(after=None)
    with pytest.raises(ValueError) as excinfo:
        render_markdown(report)
    assert "before and an after" in str(excinfo.value)
    with pytest.raises(ValueError):
        summarize(report)
    with pytest.raises(ValueError):
        write_report(report, tmp_path)
    assert not list(tmp_path.iterdir())


def test_row_from_spec_component_folds_cadence_into_the_budget():
    component = Component(name="model.denoiser", params=31_000_000,
                          cadence=Cadence.PER_STEP, calls_per_decision=20.0,
                          latency_ms=1.2, dtype="float16")
    verdict = Verdict(component="model.denoiser", action="reduce_calls",
                      why="20 denoise steps is the lever, not the kernel")
    row = ComponentRow.from_component(component, verdict, after_ms=6.0)
    assert row.before_ms == pytest.approx(24.0)
    assert row.action == "reduce_calls" and row.why.startswith("20 denoise")
    assert row.speedup == pytest.approx(4.0)


def test_empty_inventory_does_not_claim_full_coverage():
    """An empty component list means the plan was not passed, not that nothing
    was skipped. Claiming the latter would hide the report's most useful half."""
    from sparx_agency.tasks.common.trt_optimizer.bench import markdown
    lines = markdown._not_converted_lines([])
    text = "\n".join(lines)
    assert "No component inventory was supplied" in text
    assert "Nothing was skipped" not in text
