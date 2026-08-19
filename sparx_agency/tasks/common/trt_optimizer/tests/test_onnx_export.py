"""Tests for the ONNX export stage: the artifact, the gates and the manifest.

These need a real torch (and, through the op gate, a real ``onnx``), so run them
with the conda interpreter:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        /home/nadavc/miniconda3/envs/navdp/bin/python -m pytest \\
        sparx_agency/tasks/common/trt_optimizer/tests/test_onnx_export.py -q

The models are deliberately tiny. What is under test is not a network, it is the
*contract*: the spec decides the filename, the tensor names and the shapes; the
opset floor and the static-shape rule are refusals rather than warnings; and the
manifest is the record that ties an engine back to the exact graph it came from,
so its hash, shapes and cadence are checked field by field.

``slim=False`` almost everywhere -- onnxslim is a separate concern with its own
Jetson caveat, and one test covers it explicitly.
"""
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn                                        # noqa: E402
import torch.nn.functional as F                              # noqa: E402

from sparx_agency.tasks.common.trt_optimizer.export import (  # noqa: E402
    onnx_export, op_gate,
)
from sparx_agency.tasks.common.trt_optimizer.spec import (
    ShapeProfile,  # noqa: E402
    Cadence, GraphSpec, ShapeProfile,
)


# --------------------------------------------------------------------------
# export wrappers
# --------------------------------------------------------------------------

class _Tiny(nn.Module):
    """One input, one output, and a LayerNorm so opset 17 has something to do."""

    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.norm(self.fc(x))


class _TwoInTwoOut(nn.Module):
    """Two inputs consumed in the spec's declared order, two outputs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, rgb, goal):
        embed = self.fc(rgb)
        return embed, embed + goal


class _Upsampler(nn.Module):
    """Emits a ``Resize`` node -- what an un-baked pos-embed leaves behind."""

    def forward(self, x):
        return F.interpolate(x, size=(8, 8), mode="bicubic",
                             align_corners=False)


def _spec(key="tiny", **kwargs):
    """A valid single-input GraphSpec, overridable field by field."""
    fields = dict(inputs={"x": (1, 8)}, outputs=["y"], component="tiny")
    fields.update(kwargs)
    return GraphSpec(key=key, **fields)


@pytest.fixture(autouse=True)
def _quiet_exporter_warnings():
    """The TorchScript exporter is loud on torch 2.11; the noise is not news."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------
# sha256
# --------------------------------------------------------------------------

def test_sha256_matches_hashlib_over_the_file_bytes(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"engine bytes")

    assert onnx_export.sha256(path) == hashlib.sha256(b"engine bytes").hexdigest()


# --------------------------------------------------------------------------
# example_inputs
# --------------------------------------------------------------------------

def test_example_inputs_match_the_declared_shapes_and_order():
    spec = _spec(inputs={"rgb": (1, 3, 8, 8), "goal": (1, 3), "mask": (1, 1, 4)},
                 outputs=["a"])

    args = onnx_export.example_inputs(spec)

    assert isinstance(args, tuple)
    assert [tuple(t.shape) for t in args] == [(1, 3, 8, 8), (1, 3), (1, 1, 4)]
    assert spec.input_names() == ["rgb", "goal", "mask"]


def test_example_inputs_are_deterministic_for_a_fixed_seed():
    spec = _spec(inputs={"a": (2, 3), "b": (1, 4)}, outputs=["y"])

    first = onnx_export.example_inputs(spec, seed=7)
    second = onnx_export.example_inputs(spec, seed=7)

    assert len(first) == len(second) == 2
    for lhs, rhs in zip(first, second):
        assert torch.equal(lhs, rhs)


def test_example_inputs_differ_for_a_different_seed():
    spec = _spec(inputs={"a": (4, 4)}, outputs=["y"])

    assert not torch.equal(onnx_export.example_inputs(spec, seed=0)[0],
                           onnx_export.example_inputs(spec, seed=1)[0])


def test_example_inputs_do_not_disturb_the_global_rng():
    spec = _spec(inputs={"a": (4, 4)}, outputs=["y"])
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    onnx_export.example_inputs(spec, seed=99)

    assert torch.equal(torch.randn(4), expected)


def test_example_inputs_are_float32_on_cpu():
    args = onnx_export.example_inputs(_spec())

    assert args[0].dtype is torch.float32
    assert args[0].device.type == "cpu"


# --------------------------------------------------------------------------
# export_graph: the artifact
# --------------------------------------------------------------------------

def test_export_graph_writes_the_file_and_returns_its_path(tmp_path):
    path = onnx_export.export_graph(_spec(), _Tiny(), tmp_path, slim=False)

    assert path == tmp_path / "tiny.onnx"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_export_graph_creates_a_missing_output_directory(tmp_path):
    out_dir = tmp_path / "engines" / "onnx"
    assert not out_dir.exists()

    path = onnx_export.export_graph(_spec(), _Tiny(), out_dir, slim=False)

    assert path.parent == out_dir
    assert path.is_file()


def test_export_graph_names_the_io_tensors_from_the_spec(tmp_path):
    onnx = pytest.importorskip("onnx")
    spec = _spec(key="two", inputs={"rgb": (1, 8), "goal": (1, 4)},
                 outputs=["embed", "sum"])

    path = onnx_export.export_graph(spec, _TwoInTwoOut(), tmp_path, slim=False)

    model = onnx.load(str(path))
    assert [i.name for i in model.graph.input] == ["rgb", "goal"]
    assert [o.name for o in model.graph.output] == ["embed", "sum"]


def test_export_graph_emits_static_shapes_the_gate_accepts(tmp_path):
    onnx = pytest.importorskip("onnx")

    path = onnx_export.export_graph(_spec(), _Tiny(), tmp_path, slim=False)

    result = op_gate.gate(onnx.load(str(path)), check_model=False)
    assert result.dynamic_tensors == []
    assert result.ok is True


def test_export_graph_at_opset_17_emits_one_layernormalization_node(tmp_path):
    onnx = pytest.importorskip("onnx")

    path = onnx_export.export_graph(_spec(), _Tiny(), tmp_path, slim=False)

    ops = [n.op_type for n in onnx.load(str(path)).graph.node]
    assert ops.count("LayerNormalization") == 1
    assert "ReduceMean" not in ops


def test_export_graph_accepts_explicit_example_inputs(tmp_path):
    args = (torch.ones(1, 8),)

    path = onnx_export.export_graph(_spec(), _Tiny(), tmp_path, inputs=args,
                                    slim=False)

    assert path.is_file()


def test_export_graph_runs_the_onnxslim_pass_when_asked(tmp_path):
    pytest.importorskip("onnxslim")

    path = onnx_export.export_graph(_spec(), _Tiny(), tmp_path, slim=True)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_export_graph_leaves_the_module_in_its_original_training_mode(tmp_path):
    module = _Tiny().train()

    onnx_export.export_graph(_spec(), module, tmp_path, slim=False)

    assert module.training is True


# --------------------------------------------------------------------------
# export_graph: the refusals
# --------------------------------------------------------------------------

def test_export_graph_rejects_an_opset_below_the_floor(tmp_path):
    spec = _spec(opset=16)

    with pytest.raises(ValueError) as excinfo:
        onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)

    message = str(excinfo.value)
    assert "opset 16" in message
    assert "%d is the floor" % onnx_export.MIN_OPSET in message
    assert "LayerNormalization" in message
    assert not list(tmp_path.glob("*.onnx"))


def test_export_graph_rejects_a_dynamic_dimension_via_spec_validate(tmp_path):
    spec = _spec(inputs={"x": (1, -1)})

    with pytest.raises(ValueError) as excinfo:
        onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)

    message = str(excinfo.value)
    assert "declares no ShapeProfile" in message
    assert "fixed size" in message          # the message names both ways out
    assert not list(tmp_path.glob("*.onnx"))


def test_export_graph_accepts_a_dynamic_dimension_that_declares_a_profile(tmp_path):
    """The same axis is fine once the spec says what range it must cover."""
    spec = _spec(inputs={"x": (-1, 8)})
    spec.profiles = {"x": ShapeProfile(min=(1, 8), opt=(2, 8), max=(8, 8))}

    onnx = pytest.importorskip("onnx")
    path = onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)

    assert path.exists()
    model = onnx.load(str(path))
    batch = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert not batch.HasField("dim_value")   # exported as a free dimension


def test_export_graph_rejects_a_spec_with_no_inputs(tmp_path):
    spec = GraphSpec(key="empty", inputs={}, outputs=["y"])

    with pytest.raises(ValueError, match="has no inputs"):
        onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)


def test_export_graph_rejects_a_spec_with_no_outputs(tmp_path):
    spec = GraphSpec(key="empty", inputs={"x": (1, 8)}, outputs=[])

    with pytest.raises(ValueError, match="has no outputs"):
        onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)


def test_export_graph_validates_the_spec_before_the_opset(tmp_path):
    """A spec that is both dynamic and below the floor fails on the shapes."""
    spec = _spec(inputs={"x": (1, 0)}, opset=16)

    with pytest.raises(ValueError, match="declares no ShapeProfile"):
        onnx_export.export_graph(spec, _Tiny(), tmp_path, slim=False)


def test_export_graph_raises_when_the_op_gate_rejects_the_graph(tmp_path):
    spec = _spec(key="upsampler", inputs={"x": (1, 1, 4, 4)}, outputs=["y"])

    with pytest.raises(RuntimeError) as excinfo:
        onnx_export.export_graph(spec, _Upsampler(), tmp_path,
                                 policy=op_gate.vit_policy(), slim=False)

    message = str(excinfo.value)
    assert "ONNX op gate failed for upsampler" in message
    assert "Resize" in message


def test_the_same_graph_passes_under_the_default_policy(tmp_path):
    spec = _spec(key="upsampler", inputs={"x": (1, 1, 4, 4)}, outputs=["y"])

    path = onnx_export.export_graph(spec, _Upsampler(), tmp_path, slim=False)

    assert path.is_file()


# --------------------------------------------------------------------------
# export_all and the manifest
# --------------------------------------------------------------------------

def _two_specs():
    return [
        GraphSpec(key="encoder", inputs={"rgb": (1, 8)}, outputs=["embed"],
                  component="vision_tower", cadence=Cadence.PER_FRAME,
                  calls_per_decision=1.0, precision_sensitive=True),
        GraphSpec(key="denoiser", inputs={"x": (1, 8)}, outputs=["v"],
                  component="head", cadence=Cadence.PER_STEP,
                  calls_per_decision=20.0, precision_sensitive=False),
    ]


def test_export_all_writes_every_graph_and_the_manifest(tmp_path):
    specs = _two_specs()
    modules = {"encoder": _Tiny(), "denoiser": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    assert (tmp_path / "encoder.onnx").is_file()
    assert (tmp_path / "denoiser.onnx").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert sorted(manifest["graphs"]) == ["denoiser", "encoder"]


def test_the_manifest_records_the_hash_shapes_cadence_and_sensitivity(tmp_path):
    specs = _two_specs()
    modules = {"encoder": _Tiny(), "denoiser": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    encoder = manifest["graphs"]["encoder"]
    assert encoder["onnx"] == "encoder.onnx"
    assert encoder["onnx_sha256"] == onnx_export.sha256(tmp_path / "encoder.onnx")
    assert len(encoder["onnx_sha256"]) == 64
    assert encoder["inputs"] == ["rgb"]
    assert encoder["outputs"] == ["embed"]
    assert encoder["shapes"] == {"rgb": [1, 8]}
    assert encoder["cadence"] == Cadence.PER_FRAME
    assert encoder["calls_per_decision"] == 1.0
    assert encoder["precision_sensitive"] is True

    denoiser = manifest["graphs"]["denoiser"]
    assert denoiser["cadence"] == Cadence.PER_STEP
    assert denoiser["calls_per_decision"] == 20.0
    assert denoiser["precision_sensitive"] is False


def test_the_manifest_hash_is_of_the_shipped_file_not_an_intermediate(tmp_path):
    specs = _two_specs()[:1]
    modules = {"encoder": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    on_disk = hashlib.sha256((tmp_path / "encoder.onnx").read_bytes()).hexdigest()
    assert manifest["graphs"]["encoder"]["onnx_sha256"] == on_disk


def test_the_manifest_written_to_disk_equals_the_returned_dict(tmp_path):
    specs = _two_specs()
    modules = {"encoder": _Tiny(), "denoiser": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_export_all_folds_in_the_extra_keys(tmp_path):
    specs = _two_specs()[:1]
    modules = {"encoder": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False,
                                      extra={"model": "navdp",
                                             "checkpoint_sha256": "abc123"})

    assert manifest["model"] == "navdp"
    assert manifest["checkpoint_sha256"] == "abc123"
    assert "encoder" in manifest["graphs"]


def test_export_all_records_the_opset_each_graph_was_exported_at(tmp_path):
    """A spec above the floor must not be recorded as if it were at the floor."""
    specs = [GraphSpec(key="floor", inputs={"rgb": (1, 8)}, outputs=["embed"]),
             GraphSpec(key="above", inputs={"rgb": (1, 8)}, outputs=["embed"],
                       opset=18)]
    modules = {"floor": _Tiny(), "above": _Tiny()}

    manifest = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    assert manifest["graphs"]["floor"]["opset"] == onnx_export.MIN_OPSET
    assert manifest["graphs"]["above"]["opset"] == 18
    assert manifest["opset_floor"] == onnx_export.MIN_OPSET


def test_export_all_applies_the_per_key_policy(tmp_path):
    specs = [GraphSpec(key="upsampler", inputs={"x": (1, 1, 4, 4)},
                       outputs=["y"])]

    def policy_for(key):
        return op_gate.vit_policy() if key == "upsampler" else None

    with pytest.raises(RuntimeError, match="Resize"):
        onnx_export.export_all(specs, {"upsampler": _Upsampler()}, tmp_path,
                               policy_for=policy_for, slim=False)


def test_export_all_raises_key_error_when_a_wrapper_is_missing(tmp_path):
    specs = _two_specs()

    with pytest.raises(KeyError) as excinfo:
        onnx_export.export_all(specs, {"encoder": _Tiny()}, tmp_path,
                               slim=False)

    assert "denoiser" in str(excinfo.value)
    assert "no export wrapper supplied" in str(excinfo.value)


def test_export_all_writes_no_manifest_when_a_wrapper_is_missing(tmp_path):
    with pytest.raises(KeyError):
        onnx_export.export_all(_two_specs(), {}, tmp_path, slim=False)

    assert not (tmp_path / "manifest.json").exists()


def test_export_all_writes_a_manifest_for_an_empty_spec_list(tmp_path):
    out_dir = tmp_path / "onnx"

    manifest = onnx_export.export_all([], {}, out_dir, slim=False)

    assert manifest["graphs"] == {}
    assert (out_dir / "manifest.json").is_file()


# --------------------------------------------------------------------------
# read_manifest
# --------------------------------------------------------------------------

def test_read_manifest_round_trips_what_export_all_wrote(tmp_path):
    specs = _two_specs()
    modules = {"encoder": _Tiny(), "denoiser": _Tiny()}
    written = onnx_export.export_all(specs, modules, tmp_path, slim=False)

    assert onnx_export.read_manifest(tmp_path) == written


def test_read_manifest_accepts_a_string_path(tmp_path):
    onnx_export.export_all(_two_specs()[:1], {"encoder": _Tiny()}, tmp_path,
                           slim=False)

    assert "encoder" in onnx_export.read_manifest(str(tmp_path))["graphs"]


def test_read_manifest_raises_an_actionable_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        onnx_export.read_manifest(tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "run the ONNX export stage first" in message
    assert "gitignored" in message
