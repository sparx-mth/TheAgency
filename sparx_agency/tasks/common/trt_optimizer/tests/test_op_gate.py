"""Tests for the ONNX op gate, run without ``onnx`` installed.

The gate reads exactly three things off a model: ``graph.node[*].op_type`` /
``.domain`` and the ``graph.input``/``graph.output`` dimension protos. That is a
small enough surface to stand in for, and standing in for it is what lets the
whole policy live in the repo ``.venv`` -- where there is no ``onnx``, and where
a real ``ModelProto`` with a genuinely dynamic dimension could not be built
anyway without also exporting a model.

The fakes below mimic the protobuf accessors the module actually calls,
including ``HasField('dim_value')``, which is the *only* way to tell a symbolic
dimension from a zero-valued one. Every test passes ``check_model=False`` so the
gate never reaches for ``onnx.checker``.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.trt_optimizer.export import op_gate


# --------------------------------------------------------------------------
# fake ModelProto
# --------------------------------------------------------------------------

class _FakeDim(object):
    """One tensor dimension: a fixed value, or a symbolic (dynamic) one."""

    def __init__(self, value=None, param=""):
        self._has_value = value is not None
        self.dim_value = int(value) if value is not None else 0
        self.dim_param = param

    def HasField(self, field):  # noqa: N802 (protobuf spelling)
        """Mimic the protobuf oneof probe the gate uses."""
        if field == "dim_value":
            return self._has_value
        if field == "dim_param":
            return bool(self.dim_param)
        raise ValueError("unknown field %r" % (field,))


class _FakeShape(object):
    def __init__(self, dims):
        self.dim = list(dims)


class _FakeTensorType(object):
    def __init__(self, shape):
        self.shape = shape


class _FakeType(object):
    def __init__(self, tensor_type):
        self.tensor_type = tensor_type


class _FakeValueInfo(object):
    """A graph input or output with the nested ``.type.tensor_type.shape.dim``."""

    def __init__(self, name, dims):
        self.name = name
        self.type = _FakeType(_FakeTensorType(_FakeShape(dims)))


class _FakeNode(object):
    def __init__(self, op_type, domain=""):
        self.op_type = op_type
        self.domain = domain


class _FakeGraph(object):
    def __init__(self, nodes, inputs, outputs):
        self.node = list(nodes)
        self.input = list(inputs)
        self.output = list(outputs)


class _FakeModel(object):
    """The whole stand-in: ``_load`` accepts anything exposing ``.graph``."""

    def __init__(self, graph):
        self.graph = graph


def _dims(spec):
    """Build dimension protos: an int is static, a str is a symbolic dim."""
    out = []
    for entry in spec:
        if isinstance(entry, str):
            out.append(_FakeDim(param=entry))
        else:
            out.append(_FakeDim(value=entry))
    return out


def _value(name, shape):
    return _FakeValueInfo(name, _dims(shape))


def _model(ops, inputs=(("rgb", (1, 3, 224, 224)),),
           outputs=(("embed", (1, 768)),)):
    """A fake model from ``ops`` (each an op type or an (op, domain) pair)."""
    nodes = []
    for entry in ops:
        if isinstance(entry, tuple):
            nodes.append(_FakeNode(entry[0], entry[1]))
        else:
            nodes.append(_FakeNode(entry))
    return _FakeModel(_FakeGraph(
        nodes,
        [_value(n, s) for n, s in inputs],
        [_value(n, s) for n, s in outputs],
    ))


# --------------------------------------------------------------------------
# op counting
# --------------------------------------------------------------------------

def test_op_counts_tally_every_node_by_type():
    model = _model(["MatMul", "Add", "MatMul", "Softmax", "MatMul", "Add"])
    result = op_gate.gate(model, check_model=False)

    assert result.op_counts == {"MatMul": 3, "Add": 2, "Softmax": 1}
    assert result.ok is True


def test_node_count_sums_the_op_counts_and_summary_reports_both():
    model = _model(["Conv", "Relu", "Conv", "GlobalAveragePool"])
    result = op_gate.gate(model, key="backbone", check_model=False)

    assert result.node_count == 4
    assert len(result.op_counts) == 3
    assert result.summary() == "backbone: 4 nodes, 3 op types, PASS"


def test_summary_says_fail_when_the_gate_failed():
    model = _model(["MultiHeadAttention"])
    result = op_gate.gate(model, key="vit", check_model=False)

    assert result.ok is False
    assert result.summary() == "vit: 1 nodes, 1 op types, FAIL"


def test_an_empty_graph_counts_zero_nodes():
    result = op_gate.gate(_model([]), check_model=False)

    assert result.node_count == 0
    assert result.op_counts == {}


# --------------------------------------------------------------------------
# the hard rules: fused attention, non-standard domains
# --------------------------------------------------------------------------

@pytest.mark.parametrize("op_type", [
    "Attention", "MultiHeadAttention", "GroupQueryAttention",
    "ScaledDotProductAttention",
])
def test_any_fused_attention_node_fails_the_default_policy(op_type):
    result = op_gate.gate(_model(["MatMul", op_type, "Add"]), check_model=False)

    assert result.ok is False
    assert result.forbidden_found == [op_type]
    assert any("forbidden ops survived" in m for m in result.messages)


def test_the_fused_attention_message_names_the_patch_that_did_not_apply():
    result = op_gate.gate(_model(["Attention"]), check_model=False)

    message = " ".join(result.messages)
    assert "SDPA" in message and "fast-path" in message


@pytest.mark.parametrize("domain", list(op_gate.NONSTANDARD_DOMAINS))
def test_a_nonstandard_operator_domain_fails(domain):
    model = _model(["MatMul", ("Gelu", domain)])
    result = op_gate.gate(model, check_model=False)

    assert result.ok is False
    assert result.forbidden_found == ["Gelu (domain %s)" % domain]
    # The op is still counted: the inventory is of the graph, not of the verdict.
    assert result.op_counts["Gelu"] == 1


def test_the_standard_domain_and_the_empty_domain_both_pass():
    model = _model([("Conv", ""), ("Relu", "ai.onnx")])
    result = op_gate.gate(model, check_model=False)

    assert result.ok is True
    assert result.forbidden_found == []


def test_repeated_forbidden_ops_are_reported_once_and_sorted():
    model = _model(["Attention", "MultiHeadAttention", "Attention"])
    result = op_gate.gate(model, check_model=False)

    assert result.forbidden_found == ["Attention", "MultiHeadAttention"]
    assert result.op_counts == {"Attention": 2, "MultiHeadAttention": 1}


# --------------------------------------------------------------------------
# policy is per model: Resize
# --------------------------------------------------------------------------

def test_the_default_policy_allows_resize():
    result = op_gate.gate(_model(["Conv", "Resize", "Conv"]), check_model=False)

    assert result.ok is True
    assert result.forbidden_found == []


def test_vit_policy_rejects_resize():
    result = op_gate.gate(_model(["Conv", "Resize", "Conv"]),
                          policy=op_gate.vit_policy(), check_model=False)

    assert result.ok is False
    assert result.forbidden_found == ["Resize"]


def test_vit_policy_keeps_every_other_default_rule():
    policy = op_gate.vit_policy()

    assert policy.require_static is True
    assert policy.forbidden_suffixes == (op_gate.FUSED_ATTENTION_SUFFIX,)
    assert policy.suspect == op_gate.DATA_DEPENDENT_OPS
    assert tuple(policy.forbidden_domains) == op_gate.NONSTANDARD_DOMAINS


def test_vit_policy_still_catches_a_fused_attention_node():
    result = op_gate.gate(_model(["Attention"]), policy=op_gate.vit_policy(),
                          check_model=False)

    assert result.forbidden_found == ["Attention"]


# --------------------------------------------------------------------------
# data-dependent ops: reported, never fatal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("op_type", sorted(op_gate.DATA_DEPENDENT_OPS))
def test_data_dependent_ops_are_suspect_but_do_not_fail_the_gate(op_type):
    result = op_gate.gate(_model(["MatMul", op_type]), check_model=False)

    assert result.ok is True
    assert result.suspect_found == [op_type]
    assert result.forbidden_found == []
    assert any("data-dependent ops present" in m for m in result.messages)


def test_the_suspect_message_explains_that_tracing_froze_one_branch():
    result = op_gate.gate(_model(["If", "Where"]), check_model=False)

    assert result.suspect_found == ["If", "Where"]
    message = " ".join(result.messages)
    assert "froze" in message and "Not fatal" in message


def test_enforce_returns_the_result_when_only_suspect_ops_are_present():
    result = op_gate.enforce(_model(["TopK"]), check_model=False)

    assert result.ok is True
    assert result.suspect_found == ["TopK"]


def test_a_policy_may_clear_the_suspect_set_entirely():
    policy = op_gate.OpGatePolicy(suspect=frozenset())
    result = op_gate.gate(_model(["Where"]), policy=policy, check_model=False)

    assert result.suspect_found == []
    assert result.messages == []


# --------------------------------------------------------------------------
# static shapes
# --------------------------------------------------------------------------

def test_a_symbolic_input_dimension_fails_the_static_requirement():
    model = _model(["Conv"], inputs=(("rgb", ("batch", 3, 224, 224)),))
    result = op_gate.gate(model, check_model=False)

    assert result.ok is False
    assert result.dynamic_tensors == ["rgb"]
    assert any("dynamic shapes on rgb" in m for m in result.messages)


def test_a_zero_valued_dimension_is_dynamic_too():
    model = _model(["Conv"], inputs=(("rgb", (0, 3, 224, 224)),))
    result = op_gate.gate(model, check_model=False)

    assert result.ok is False
    assert result.dynamic_tensors == ["rgb"]


def test_a_dynamic_output_dimension_is_caught_as_well():
    model = _model(["Conv"], outputs=(("embed", (1, "tokens")),))
    result = op_gate.gate(model, check_model=False)

    assert result.ok is False
    assert result.dynamic_tensors == ["embed"]


def test_every_dynamic_tensor_is_named_once_and_sorted():
    model = _model(
        ["Conv"],
        inputs=(("rgb", ("b", 3, "h", "w")), ("goal", (1, 3))),
        outputs=(("embed", ("b", 768)),),
    )
    result = op_gate.gate(model, check_model=False)

    assert result.dynamic_tensors == ["embed", "rgb"]


def test_require_static_false_skips_the_shape_check_entirely():
    model = _model(["Conv"], inputs=(("rgb", ("batch", 3, 224, 224)),))
    policy = op_gate.OpGatePolicy(require_static=False)
    result = op_gate.gate(model, policy=policy, check_model=False)

    assert result.ok is True
    assert result.dynamic_tensors == []
    assert result.messages == []


def test_a_fully_static_graph_reports_no_dynamic_tensors():
    result = op_gate.gate(_model(["Conv"]), check_model=False)

    assert result.dynamic_tensors == []
    assert result.ok is True


# --------------------------------------------------------------------------
# enforce()
# --------------------------------------------------------------------------

def test_enforce_passes_a_clean_graph_through_unchanged():
    model = _model(["Conv", "Relu", "MatMul"])
    result = op_gate.enforce(model, key="clean", check_model=False)

    assert result.ok is True
    assert result.key == "clean"
    assert result.node_count == 3


def test_enforce_raises_carrying_every_message_it_collected():
    model = _model(
        ["Conv", "Resize", "Where", ("Gelu", "com.microsoft"), "Attention"],
        inputs=(("rgb", ("batch", 3, 224, 224)),),
    )
    with pytest.raises(RuntimeError) as excinfo:
        op_gate.enforce(model, policy=op_gate.vit_policy(), key="vit_trunk",
                        check_model=False)

    text = str(excinfo.value)
    assert "ONNX op gate failed for vit_trunk" in text
    # every one of the three message kinds survived into the raise
    assert "forbidden ops survived" in text
    assert "dynamic shapes on rgb" in text
    assert "data-dependent ops present: Where" in text
    # and the forbidden list is complete, not just the first hit
    assert "Attention" in text and "Resize" in text
    assert "Gelu (domain com.microsoft)" in text


def test_enforce_names_the_key_it_was_given():
    with pytest.raises(RuntimeError) as excinfo:
        op_gate.enforce(_model(["Attention"]), key="navdp_encoder",
                        check_model=False)

    assert "navdp_encoder" in str(excinfo.value)


def test_an_in_memory_model_defaults_its_key_to_graph():
    result = op_gate.gate(_model(["Conv"]), check_model=False)

    assert result.key == "graph"
