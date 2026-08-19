"""Tests for the architecture dissector, run against a hand-built fake tree.

The point of the fakes is that the accounting invariant -- ``sum(component
params) == total_params(model)`` -- is a property of the *walk*, not of torch,
so it can and should be covered in the pure-numpy venv that has no torch at all.
The fakes implement exactly the duck-type :mod:`...dissect` documents:
``named_children()`` / ``named_parameters()`` / ``parameters()`` /
``named_buffers()``, over objects with ``.numel()`` and ``.dtype``.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer.dissect import (
    check_accounting, describe, dtype_histogram, inventory, total_buffers,
    total_params)
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, Exportability


class FakeParam(object):
    """Stand-in for a tensor: only ``numel()`` and ``dtype`` are used."""

    def __init__(self, numel, dtype="float32"):
        self._numel = int(numel)
        self.dtype = dtype

    def numel(self):
        return self._numel


class FakeModule(object):
    """Stand-in for ``nn.Module`` exposing only the walker's duck-type."""

    def __init__(self, params=None, children=None, buffers=None):
        self._params = list(params or [])
        self._children = list(children or [])
        self._buffers = list(buffers or [])

    def named_children(self):
        for name, child in self._children:
            yield name, child

    def named_parameters(self):
        for name, param in self._params:
            yield name, param
        for child_name, child in self._children:
            for name, param in child.named_parameters():
                yield "%s.%s" % (child_name, name), param

    def parameters(self):
        for _name, param in self.named_parameters():
            yield param

    def named_buffers(self):
        for name, buf in self._buffers:
            yield name, buf
        for child_name, child in self._children:
            for name, buf in child.named_buffers():
                yield "%s.%s" % (child_name, name), buf


def _leaf(numel, dtype="float32"):
    return FakeModule(params=[("weight", FakeParam(numel, dtype))])


def _deep_tree():
    """root -> backbone -> stage -> block -> leaf, 1000 parameters at the bottom."""
    block = FakeModule(children=[("0", _leaf(600)), ("1", _leaf(400))])
    stage = FakeModule(children=[("block", block)])
    backbone = FakeModule(children=[("stage", stage)])
    return FakeModule(children=[("backbone", backbone)])


def _two_branch():
    """root -> trunk -> {big: 1000, tiny: 5}, plus 7 parameters held on trunk."""
    trunk = FakeModule(params=[("bias", FakeParam(7))],
                       children=[("big", _leaf(1000)), ("tiny", _leaf(5))])
    return FakeModule(children=[("trunk", trunk)])


# --------------------------------------------------------------------------
# frontier
# --------------------------------------------------------------------------

def test_depth_one_collapses_a_deep_tree_to_one_component():
    model = _deep_tree()
    components = inventory(model, max_depth=1)
    assert [c.name for c in components] == ["backbone"]
    assert components[0].params == 1000


def test_depth_two_stops_at_the_frontier_not_the_leaves():
    model = _deep_tree()
    components = inventory(model, max_depth=2)
    assert [c.name for c in components] == ["backbone.stage"]
    assert components[0].params == 1000


def test_deeper_max_depth_expands_further():
    model = _deep_tree()
    names = [c.name for c in inventory(model, max_depth=4)]
    assert names == ["backbone.stage.block.0", "backbone.stage.block.1"]


def test_childless_node_above_the_frontier_is_emitted_where_it_sits():
    model = FakeModule(children=[("head", _leaf(12)),
                                 ("trunk", FakeModule(
                                     children=[("a", _leaf(30))]))])
    names = [c.name for c in inventory(model, max_depth=3)]
    assert names == ["head", "trunk.a"]


def test_childless_root_becomes_a_single_component_named_for_its_class():
    components = inventory(_leaf(42), max_depth=2)
    assert [c.name for c in components] == ["FakeModule"]
    assert components[0].params == 42


def test_max_depth_below_one_raises():
    with pytest.raises(ValueError):
        inventory(_deep_tree(), max_depth=0)


def test_non_module_object_raises_type_error():
    with pytest.raises(TypeError):
        inventory(object())


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_depth", [1, 2, 3, 4, 9])
def test_parameter_accounting_is_exact_at_every_depth(max_depth):
    model = _two_branch()
    components = inventory(model, max_depth=max_depth)
    assert sum(c.params for c in components) == total_params(model) == 1012
    check_accounting(components, model)


def test_check_accounting_raises_on_a_truncated_inventory():
    model = _two_branch()
    components = inventory(model, max_depth=3)
    with pytest.raises(ValueError):
        check_accounting(components[:-1], model)


def test_tied_parameter_is_counted_once():
    shared = FakeParam(64)
    model = FakeModule(children=[("a", FakeModule(params=[("w", shared)])),
                                 ("b", FakeModule(params=[("w", shared)]))])
    components = inventory(model, max_depth=1)
    assert total_params(model) == 64
    assert [(c.name, c.params) for c in components] == [("a", 64), ("b", 0)]
    check_accounting(components, model)


# --------------------------------------------------------------------------
# the '<parent>.other' fold-in
# --------------------------------------------------------------------------

def test_min_params_folds_a_dropped_subtree_into_parent_other():
    model = _two_branch()
    components = inventory(model, max_depth=3, min_params=100)
    by_name = dict((c.name, c.params) for c in components)
    assert by_name == {"trunk.big": 1000, "trunk.other": 12}
    check_accounting(components, model)


def test_parameters_held_directly_on_a_parent_land_in_other():
    model = _two_branch()
    components = inventory(model, max_depth=3)
    by_name = dict((c.name, c.params) for c in components)
    assert by_name == {"trunk.big": 1000, "trunk.tiny": 5, "trunk.other": 7}


def test_root_level_other_is_prefixed_with_the_root_class_name():
    model = FakeModule(params=[("bias", FakeParam(3))],
                       children=[("big", _leaf(500))])
    names = [c.name for c in inventory(model, max_depth=1)]
    assert names == ["big", "FakeModule.other"]


def test_other_bucket_is_hostile_and_never_offered_for_export():
    model = _two_branch()
    other = [c for c in inventory(model, max_depth=3) if c.name.endswith(".other")]
    assert len(other) == 1
    assert other[0].exportability == Exportability.HOSTILE
    assert "not exportable" in other[0].reason


def test_min_params_folds_at_the_highest_level_that_is_too_small():
    """A whole branch under ``min_params`` folds into its own parent's bucket."""
    model = _two_branch()
    components = inventory(model, max_depth=3, min_params=10 ** 6)
    assert [c.name for c in components] == ["FakeModule.other"]
    assert components[0].params == 1012
    check_accounting(components, model)


# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------

def test_dtype_histogram_counts_elements_per_dtype():
    model = FakeModule(children=[("a", _leaf(100, "bfloat16")),
                                 ("b", _leaf(30, "float32")),
                                 ("c", _leaf(70, "bfloat16"))])
    assert dtype_histogram(model) == {"bfloat16": 170, "float32": 30}


def test_component_dtype_is_the_most_common_in_its_subtree():
    subtree = FakeModule(children=[("a", _leaf(100, "bfloat16")),
                                   ("b", _leaf(30, "float32"))])
    model = FakeModule(children=[("trunk", subtree)])
    assert inventory(model, max_depth=1)[0].dtype == "bfloat16"


def test_dtype_ties_resolve_to_the_first_one_seen():
    subtree = FakeModule(children=[("a", _leaf(50, "float16")),
                                   ("b", _leaf(50, "float32"))])
    model = FakeModule(children=[("trunk", subtree)])
    assert inventory(model, max_depth=1)[0].dtype == "float16"


def test_torch_prefixed_dtype_strings_are_stripped():
    model = FakeModule(children=[("trunk", _leaf(8, "torch.bfloat16"))])
    assert inventory(model, max_depth=1)[0].dtype == "bfloat16"
    assert dtype_histogram(model) == {"bfloat16": 8}


def test_zero_parameter_component_reports_the_default_dtype():
    model = FakeModule(children=[("relu", FakeModule())])
    component = inventory(model, max_depth=1)[0]
    assert component.params == 0
    assert component.dtype == "float32"


# --------------------------------------------------------------------------
# cadences and exportability
# --------------------------------------------------------------------------

def test_cadences_match_by_exact_name_and_by_prefix():
    model = FakeModule(children=[
        ("text_encoder", _leaf(10)),
        ("vision_tower", FakeModule(children=[("blocks", _leaf(20)),
                                              ("norm", _leaf(5))]))])
    cadences = {"text_encoder": Cadence.ONCE_PER_EPISODE,
                "vision_tower": Cadence.PER_FRAME,
                "vision_tower.norm": Cadence.PER_STEP}
    by_name = dict((c.name, c.cadence)
                   for c in inventory(model, max_depth=2, cadences=cadences))
    assert by_name["text_encoder"] == Cadence.ONCE_PER_EPISODE
    assert by_name["vision_tower.blocks"] == Cadence.PER_FRAME  # prefix
    assert by_name["vision_tower.norm"] == Cadence.PER_STEP     # exact wins


def test_longest_matching_prefix_wins():
    model = FakeModule(children=[("policy", FakeModule(
        children=[("head", _leaf(4)), ("denoiser", _leaf(6))]))])
    cadences = {"policy": Cadence.PER_PLAN,
                "policy.denoiser": Cadence.PER_STEP}
    by_name = dict((c.name, c.cadence)
                   for c in inventory(model, max_depth=2, cadences=cadences))
    assert by_name == {"policy.head": Cadence.PER_PLAN,
                       "policy.denoiser": Cadence.PER_STEP}


def test_unmatched_component_defaults_to_per_frame():
    model = FakeModule(children=[("mystery", _leaf(9))])
    component = inventory(model, max_depth=1, cadences={"other": Cadence.PER_PLAN})[0]
    assert component.cadence == Cadence.PER_FRAME


def test_unknown_cadence_value_raises():
    with pytest.raises(ValueError):
        inventory(_deep_tree(), cadences={"backbone": "sometimes"})


def test_default_exportability_is_clean_with_no_reason():
    component = inventory(_deep_tree(), max_depth=1)[0]
    assert component.exportability == Exportability.CLEAN
    assert component.reason == ""


def test_exportability_fn_result_is_used():
    calls = []

    def probe(module):
        calls.append(module)
        return Exportability.NEEDS_PATCH, "SDPA math fallback"

    component = inventory(_deep_tree(), max_depth=1, exportability_fn=probe)[0]
    assert component.exportability == Exportability.NEEDS_PATCH
    assert component.reason == "SDPA math fallback"
    assert len(calls) == 1


def test_exportability_fn_returning_garbage_raises():
    with pytest.raises(ValueError):
        inventory(_deep_tree(), max_depth=1,
                  exportability_fn=lambda m: ("mostly_fine", "hmm"))


def test_exportability_fn_is_not_called_for_the_other_bucket():
    model = _two_branch()
    seen = []

    def probe(module):
        seen.append(module)
        return Exportability.CLEAN, ""

    components = inventory(model, max_depth=3, exportability_fn=probe)
    assert len(seen) == 2                      # trunk.big and trunk.tiny only
    assert len(components) == 3                # plus trunk.other


# --------------------------------------------------------------------------
# buffers and rendering
# --------------------------------------------------------------------------

def test_buffers_are_excluded_from_params_and_reported_separately():
    model = FakeModule(children=[("bn", FakeModule(
        params=[("weight", FakeParam(16))],
        buffers=[("running_mean", FakeParam(16)),
                 ("running_var", FakeParam(16))]))])
    components = inventory(model, max_depth=1)
    assert components[0].params == 16
    assert total_params(model) == 16
    assert total_buffers(model) == 32


def test_describe_renders_every_row_plus_a_total():
    text = describe(inventory(_two_branch(), max_depth=3))
    assert "trunk.big" in text and "trunk.other" in text
    assert "1.01K" in text          # 1012 parameters, human units
    assert "TOTAL" in text
    assert "per_frame" in text
    assert len(text.splitlines()) == 7   # header, rule, 3 rows, rule, total


def test_describe_on_an_empty_inventory():
    assert describe([]) == "(no components)"


# --------------------------------------------------------------------------
# real torch, when the interpreter has it
#
# These skip in the pure-numpy venv (which has no torch) and run in the
# ``navdp`` conda env. They are the proof that the duck-type above is the real
# ``nn.Module`` API and not a shape invented for the fakes.
# --------------------------------------------------------------------------

def test_real_torch_resnet_frontier_and_accounting():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None)

    top = inventory(model, max_depth=1)
    assert [c.name for c in top][:2] == ["conv1", "bn1"]
    assert sum(c.params for c in top) == total_params(model) == 11689512
    check_accounting(top, model)

    deeper = inventory(model, max_depth=2, min_params=10000)
    by_name = dict((c.name, c.params) for c in deeper)
    assert by_name["layer4.1"] == 4720640
    assert by_name["ResNet.other"] == 9536      # conv1 + bn1, folded not lost
    check_accounting(deeper, model)
    assert "layer4.1" in describe(deeper)


def test_real_torch_mixed_precision_and_buffers():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    model = torch.nn.Sequential()
    model.add_module("vision_tower",
                     torchvision.models.resnet18(weights=None).half())
    model.add_module("head", torch.nn.Linear(1000, 4))

    components = inventory(
        model, max_depth=1,
        cadences={"vision_tower": Cadence.PER_FRAME, "head": Cadence.PER_STEP},
        exportability_fn=lambda m: (Exportability.NEEDS_PATCH, "SDPA math"))
    by_name = dict((c.name, c) for c in components)
    assert by_name["vision_tower"].dtype == "float16"
    assert by_name["head"].dtype == "float32"
    assert by_name["vision_tower"].weight_bytes() == 11689512 * 2
    assert by_name["head"].cadence == Cadence.PER_STEP
    check_accounting(components, model)

    assert dtype_histogram(model) == {"float16": 11689512, "float32": 4004}
    assert total_buffers(model) == 9620          # BN stats, excluded from params


def test_real_torch_tied_embedding_is_counted_once():
    torch = pytest.importorskip("torch")
    shared = torch.nn.Embedding(100, 8)
    model = torch.nn.Module()
    model.add_module("enc", torch.nn.Sequential(shared))
    model.add_module("dec", torch.nn.Sequential(shared))

    components = inventory(model, max_depth=1)
    assert total_params(model) == 800
    assert [(c.name, c.params) for c in components] == [("enc", 800), ("dec", 0)]
    check_accounting(components, model)


def test_real_torch_bare_module_has_no_children():
    torch = pytest.importorskip("torch")
    components = inventory(torch.nn.Linear(4, 3), max_depth=2)
    assert [(c.name, c.params) for c in components] == [("Linear", 15)]
