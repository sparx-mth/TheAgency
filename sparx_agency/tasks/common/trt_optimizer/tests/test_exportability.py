"""Tests for the source-level exportability classifier.

torch is deliberately absent from this interpreter, which is the point: the
classifier has to reach a verdict on a live module by duck-typing
``named_modules()``, so the module fakes below stand in for ``nn.Module`` and
the whole suite runs in the pure-numpy venv.

The two sample tables are coverage guards as much as fixtures: a marker added
to :data:`HOSTILE_MARKERS` or :data:`PATCHABLE_MARKERS` without a snippet that
actually trips it fails ``test_every_marker_has_a_sample``, so no marker can
ship undetectable.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer.exportability import (
    HOSTILE_MARKERS,
    PATCHABLE_MARKERS,
    _PATCH_ORDER,
    classify_module,
    classify_source,
    patch_plan,
    scan,
)
from sparx_agency.tasks.common.trt_optimizer.spec import Exportability

# One snippet per hostile marker, written the way the real model writes it.
HOSTILE_SAMPLES = {
    "generate(": "out = self.lm.generate(ids, max_new_tokens=32)",
    "past_key_values": "hidden, past_key_values = self.decode(x, past)",
    "DynamicCache": "cache = DynamicCache()",
    "use_cache": "cfg = dict(use_cache=True)",
    "flash_attn": "from flash_attn import flash_attn_func",
    "flash_attention_2": 'model = load(name, attn="flash_attention_2")',
    "attn_implementation": 'model = load(name, attn_implementation="sdpa")',
    "xformers": "import xformers.ops as xops",
    "memory_efficient_attention": "y = xops.memory_efficient_attention(q, k, v)",
    ".item()": "n_tokens = mask.sum().item()",
    "torch.nonzero": "idx = torch.nonzero(mask)",
    "masked_scatter": "x = x.masked_scatter(mask, image_embeds)",
    "cu_seqlens": "y = attn(q, k, v, cu_seqlens=cu_seqlens, max_len=m)",
    "tensor_dependent_branch": "if mask.any():\n    x = x + 1\n",
    "tensor_length_loop": "for i in range(image_grid_thw.shape[0]):\n    pass\n",
}

# One snippet per patchable marker.
PATCHABLE_SAMPLES = {
    "gradient_checkpointing": "self.gradient_checkpointing = True",
    "guidance_scale_one": "guidance_scale = 1.0",
    "MemoryEfficientSwish": "self.act = MemoryEfficientSwish()",
    "AdaptiveAvgPool1d": "self.pool = nn.AdaptiveAvgPool1d(1)",
    "view_as_complex": "freqs = torch.view_as_complex(pairs)",
    "interpolate_pos_encoding": "pe = self.interpolate_pos_encoding(x, w, h)",
    "bicubic": 'pos_embed = interpolate(pe, size=s, mode="bicubic")',
    "scaled_dot_product_attention": "y = F.scaled_dot_product_attention(q, k, v)",
    "MultiheadAttention": "self.attn = nn.MultiheadAttention(768, 12)",
}

CLEAN_SOURCE = """
class TinyEncoder(object):
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        return self.head(y.flatten(1))
"""


class FakeModule(object):
    """Duck-typed stand-in for ``nn.Module``: only ``named_modules()`` matters."""

    def __init__(self, children=None):
        self._children = list(children or [])

    def named_modules(self):
        yield "", self
        for name, child in self._children:
            yield name, child


class MultiheadAttention(FakeModule):
    """A submodule recognised by class name alone, as a torch layer would be."""


class DynamicCache(FakeModule):
    """A hostile submodule, used to prove the depth limit really cuts off."""


class KVDecoder(FakeModule):
    """A class whose own source carries a hostile marker."""

    def forward(self, x, past_key_values=None):
        return self.decode(x, past_key_values)


def test_every_marker_has_a_sample():
    assert set(HOSTILE_SAMPLES) == set(HOSTILE_MARKERS)
    assert set(PATCHABLE_SAMPLES) == set(PATCHABLE_MARKERS)


def test_patch_order_covers_every_patchable_marker():
    assert set(_PATCH_ORDER) == set(PATCHABLE_MARKERS)


@pytest.mark.parametrize("marker", sorted(HOSTILE_MARKERS))
def test_each_hostile_marker_classifies_hostile(marker):
    verdict, reason = classify_source(HOSTILE_SAMPLES[marker])
    assert verdict == Exportability.HOSTILE
    assert marker in reason


@pytest.mark.parametrize("marker", sorted(PATCHABLE_MARKERS))
def test_each_patchable_marker_needs_a_patch(marker):
    verdict, reason = classify_source(PATCHABLE_SAMPLES[marker])
    assert verdict == Exportability.NEEDS_PATCH
    assert marker in reason


@pytest.mark.parametrize("marker", sorted(PATCHABLE_MARKERS))
def test_no_patchable_sample_trips_a_hostile_marker(marker):
    """A patch is only advice if the component is otherwise exportable."""
    kinds = {f["kind"] for f in scan(PATCHABLE_SAMPLES[marker])}
    assert kinds == {"patchable"}


def test_bicubic_pos_embed_is_found_across_a_wrapped_call():
    """The real DINOv2-style resize spans two lines; a line-bound match misses."""
    source = (
        "pos_embed = nn.functional.interpolate(self.pe, size=(w, h),\n"
        '                                      mode="bicubic")\n'
    )
    verdict, _ = classify_source(source)
    assert verdict == Exportability.NEEDS_PATCH
    assert [f["marker"] for f in scan(source)] == ["bicubic"]


def test_a_bare_bicubic_image_resize_is_not_a_pos_embed():
    """Preprocessing outside the graph must not be reported as a patch."""
    assert scan('img = resize(img, (224, 224), mode="bicubic")') == []


@pytest.mark.parametrize("source", [
    "guidance_scale = 1.0",
    "guidance_scale = 1",
    "guidance_scale: float = 1.00",
    "if guidance_scale == 1.0:",
    "pipe(prompt, guidance_scale=1.0, steps=20)",
])
def test_guidance_scale_of_exactly_one_is_patchable(source):
    assert classify_source(source)[0] == Exportability.NEEDS_PATCH


@pytest.mark.parametrize("source", [
    "guidance_scale = 1.5",
    "guidance_scale = 1.05",
    "guidance_scale = 7.5",
    "guidance_scale = 10.0",
    "guidance_scale = 11",
])
def test_a_live_guidance_scale_is_left_alone(source):
    """The dangerous false positive: at any scale but 1.0 the null branch is
    real, and deleting it would silently change what the model outputs."""
    assert scan(source) == []


def test_clean_source_is_clean():
    verdict, reason = classify_source(CLEAN_SOURCE)
    assert verdict == Exportability.CLEAN
    assert scan(CLEAN_SOURCE) == []
    assert "absence of evidence" in reason


def test_hostile_wins_over_patchable():
    source = (
        "y = F.scaled_dot_product_attention(q, k, v)\n"
        "self.gradient_checkpointing = True\n"
        "hidden, past_key_values = self.decode(x, past)\n"
    )
    verdict, reason = classify_source(source)
    assert verdict == Exportability.HOSTILE
    assert "past_key_values" in reason
    assert "scaled_dot_product_attention" not in reason


def test_scan_returns_every_finding_not_just_the_first():
    """All four markers, reported in marker-table order rather than source order.

    Table order is deterministic under reformatting of the model source, and it
    is what puts the blockers above the chores.
    """
    source = (
        "cache = DynamicCache()\n"
        "n = mask.sum().item()\n"
        "y = F.scaled_dot_product_attention(q, k, v)\n"
        "self.pool = nn.AdaptiveAvgPool1d(1)\n"
    )
    markers = [f["marker"] for f in scan(source)]
    assert markers == ["DynamicCache", ".item()",
                       "AdaptiveAvgPool1d", "scaled_dot_product_attention"]


def test_scan_orders_hostile_findings_first():
    source = "self.pool = nn.AdaptiveAvgPool1d(1)\nidx = torch.nonzero(mask)\n"
    kinds = [f["kind"] for f in scan(source)]
    assert kinds == ["hostile", "patchable"]


def test_scan_findings_carry_marker_kind_and_why():
    finding = scan("idx = torch.nonzero(mask)")[0]
    assert set(finding) == {"marker", "kind", "why"}
    assert finding["why"] == HOSTILE_MARKERS["torch.nonzero"]


def test_reason_caps_the_marker_list_and_points_at_scan():
    source = "\n".join(HOSTILE_SAMPLES.values())
    verdict, reason = classify_source(source)
    assert verdict == Exportability.HOSTILE
    assert "scan()" in reason
    assert len(scan(source)) > 3


def test_classify_source_rejects_a_non_string():
    with pytest.raises(TypeError):
        classify_source(FakeModule())


def test_scan_rejects_an_object_without_named_modules():
    with pytest.raises(TypeError):
        scan(object())


def test_classify_module_reads_submodule_class_names():
    module = FakeModule([("blocks.0.attn", MultiheadAttention())])
    verdict, reason = classify_module(module)
    assert verdict == Exportability.NEEDS_PATCH
    assert "MultiheadAttention" in reason


def test_classify_module_reads_the_class_source_of_its_modules():
    verdict, reason = classify_module(FakeModule([("decoder", KVDecoder())]))
    assert verdict == Exportability.HOSTILE
    assert "past_key_values" in reason


def test_classify_module_on_a_plain_module_is_clean():
    assert classify_module(FakeModule())[0] == Exportability.CLEAN


def test_max_depth_bounds_the_walk():
    deep = FakeModule([("a.b.c.d", DynamicCache())])
    assert classify_module(deep, max_depth=3)[0] == Exportability.CLEAN
    assert classify_module(deep, max_depth=4)[0] == Exportability.HOSTILE


def test_negative_max_depth_raises():
    with pytest.raises(ValueError):
        classify_module(FakeModule(), max_depth=-1)


def test_scan_accepts_a_module_as_well_as_source():
    markers = [f["marker"] for f in scan(FakeModule([("a", KVDecoder())]))]
    assert "past_key_values" in markers


def test_patch_plan_is_ordered_non_empty_and_numbered():
    source = (
        "y = F.scaled_dot_product_attention(q, k, v)\n"
        "self.gradient_checkpointing = True\n"
        "pe = self.interpolate_pos_encoding(x, w, h)\n"
    )
    steps = patch_plan(scan(source))
    assert len(steps) == 3
    assert steps[0].startswith("1. [gradient_checkpointing]")
    assert steps[1].startswith("2. [interpolate_pos_encoding]")
    assert steps[2].startswith("3. [scaled_dot_product_attention]")
    assert PATCHABLE_MARKERS["gradient_checkpointing"] in steps[0]


def test_patch_plan_follows_the_declared_order_not_the_find_order():
    findings = scan("y = F.scaled_dot_product_attention(q, k, v)")
    findings += scan("self.gradient_checkpointing = True")
    assert "[gradient_checkpointing]" in patch_plan(findings)[0]


def test_patch_plan_ignores_hostile_findings():
    steps = patch_plan(scan("cache = DynamicCache()"))
    assert steps == []


def test_patch_plan_deduplicates_repeated_markers():
    finding = scan("self.gradient_checkpointing = True")[0]
    assert len(patch_plan([finding, dict(finding)])) == 1


def test_patch_plan_on_nothing_is_empty():
    assert patch_plan([]) == []


def test_patch_plan_raises_on_an_unknown_marker():
    with pytest.raises(ValueError):
        patch_plan([{"marker": "no_such_marker", "kind": "patchable",
                     "why": ""}])


def test_patch_plan_raises_on_a_malformed_finding():
    with pytest.raises(ValueError):
        patch_plan([{"marker": "gradient_checkpointing"}])


def test_patch_plan_raises_on_an_unknown_kind():
    with pytest.raises(ValueError):
        patch_plan([{"marker": "gradient_checkpointing", "kind": "maybe"}])
