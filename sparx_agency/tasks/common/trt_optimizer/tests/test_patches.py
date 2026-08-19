"""Tests for the generic torch-side export patches.

These need a real torch, so run them with the conda interpreter:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        /home/nadavc/miniconda3/envs/navdp/bin/python -m pytest \\
        sparx_agency/tasks/common/trt_optimizer/tests/test_patches.py -q

Two properties carry most of the weight here and are tested hardest.

**A patch must not leak.** Every one of these changes global torch state or the
module being exported, and a patch still in force after the export would change
the numerics of the very reference the parity gate compares against. So the
restoration is tested on the *raising* path, not just the happy one.

**A replacement must be numerically identical.** ``pool_matrix`` is only a legal
substitute for ``AdaptiveAvgPool1d`` if it agrees with it exactly, including at
the non-divisible lengths where the pooling windows overlap -- which is the case
that a "divide the length into equal blocks" reimplementation gets wrong.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn                                        # noqa: E402
import torch.nn.functional as F                              # noqa: E402

from sparx_agency.tasks.common.trt_optimizer.export import patches  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _MemoryEfficientSwish(nn.Module):
    """EfficientNet's custom-autograd Swish, matched by class NAME."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class _Swish(nn.Module):
    """The other name the same block ships under."""

    def forward(self, x):
        return x * torch.sigmoid(x)


# The module scans ``type(child).__name__``, so the classes must carry the
# upstream names rather than the private ones used above.
_MemoryEfficientSwish.__name__ = "MemoryEfficientSwish"
_Swish.__name__ = "Swish"


class _Block(nn.Module):
    """A small tree with an activation buried one level down."""

    def __init__(self, activation):
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 1)
        self.act = activation


class _FakeViT(nn.Module):
    """A ViT-ish backbone whose pos-embed interpolation depends on the size.

    ``interpolate_pos_encoding`` deliberately answers differently for every
    ``(w, h)``, so a test can tell a baked constant from a live call.
    """

    def __init__(self, dim=8, seq=5):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq, dim))
        self.calls = []

    def interpolate_pos_encoding(self, x, w, h):
        self.calls.append((int(w), int(h)))
        return torch.full((1, x.shape[1], x.shape[2]),
                          float(w) + float(h) / 1000.0)


class _Recorder(object):
    """Counts enter/exit of a context manager under test."""

    def __init__(self):
        self.values = []


# --------------------------------------------------------------------------
# pool_matrix
# --------------------------------------------------------------------------

@pytest.mark.parametrize("in_len,out_len", [
    (8, 4),      # exactly divisible
    (16, 1),     # global average
    (7, 7),      # identity
    (10, 3),     # NOT divisible: windows of 4/3/3
    (9, 4),      # NOT divisible: overlapping window boundaries
    (5, 2),
    (13, 5),
    (6, 4),      # out_len > in_len / 2, windows overlap
    (3, 5),      # upsampling: out_len > in_len
])
def test_pool_matrix_matches_adaptive_avg_pool1d_exactly(in_len, out_len):
    torch.manual_seed(0)
    x = torch.randn(2, 3, in_len)

    expected = F.adaptive_avg_pool1d(x, out_len)
    got = x @ patches.pool_matrix(in_len, out_len)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_pool_matrix_has_the_shape_that_makes_it_a_right_multiplicand():
    matrix = patches.pool_matrix(12, 5)

    assert tuple(matrix.shape) == (12, 5)
    assert matrix.dtype is torch.float32


def test_pool_matrix_columns_are_normalized_averaging_windows():
    matrix = patches.pool_matrix(10, 3)

    # every column sums to 1: it is an average, not a sum
    assert torch.allclose(matrix.sum(dim=0), torch.ones(3), atol=1e-6)
    # and every window is contiguous, with no gaps between columns
    for j in range(3):
        nonzero = torch.nonzero(matrix[:, j]).flatten()
        assert int(nonzero[-1]) - int(nonzero[0]) + 1 == int(nonzero.numel())


def test_pool_matrix_honours_an_explicit_dtype():
    matrix = patches.pool_matrix(8, 2, dtype=torch.float64)

    assert matrix.dtype is torch.float64


# --------------------------------------------------------------------------
# export_context
# --------------------------------------------------------------------------

def test_export_context_puts_the_module_in_eval_and_yields_it():
    module = nn.Linear(4, 4).train()

    with patches.export_context(module) as yielded:
        assert yielded is module
        assert module.training is False

    assert module.training is True


def test_export_context_restores_training_mode_when_the_body_raises():
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).train()
    assert module.training is True

    with pytest.raises(RuntimeError, match="export blew up"):
        with patches.export_context(module):
            assert module.training is False
            raise RuntimeError("export blew up")

    assert module.training is True


def test_export_context_leaves_an_already_eval_module_in_eval():
    module = nn.Linear(4, 4).eval()

    with patches.export_context(module):
        assert module.training is False

    assert module.training is False


def test_export_context_disables_grad_inside_and_restores_it_after():
    module = nn.Linear(4, 4)
    assert torch.is_grad_enabled() is True

    with pytest.raises(ValueError):
        with patches.export_context(module):
            assert torch.is_grad_enabled() is False
            raise ValueError("boom")

    assert torch.is_grad_enabled() is True


def test_export_context_restores_the_mha_fastpath_after_a_raise():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        with pytest.raises(RuntimeError):
            with patches.export_context(nn.Linear(4, 4)):
                assert torch.backends.mha.get_fastpath_enabled() is False
                raise RuntimeError("boom")
        assert torch.backends.mha.get_fastpath_enabled() is True
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


# --------------------------------------------------------------------------
# eval_mode / mha_fastpath_disabled
# --------------------------------------------------------------------------

def test_eval_mode_restores_a_training_module():
    module = nn.Linear(2, 2).train()

    with patches.eval_mode(module) as yielded:
        assert yielded is module
        assert module.training is False

    assert module.training is True


def test_eval_mode_restores_even_when_the_body_raises():
    module = nn.Linear(2, 2).train()

    with pytest.raises(KeyError):
        with patches.eval_mode(module):
            raise KeyError("boom")

    assert module.training is True


def test_mha_fastpath_disabled_restores_the_previous_value_true():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)

        with patches.mha_fastpath_disabled():
            assert torch.backends.mha.get_fastpath_enabled() is False

        assert torch.backends.mha.get_fastpath_enabled() is True
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_mha_fastpath_disabled_restores_a_previously_false_value():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)

        with patches.mha_fastpath_disabled():
            assert torch.backends.mha.get_fastpath_enabled() is False

        # restored to FALSE, not to the module's own default of True
        assert torch.backends.mha.get_fastpath_enabled() is False
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_mha_fastpath_disabled_restores_after_a_raise():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(True)

        with pytest.raises(RuntimeError):
            with patches.mha_fastpath_disabled():
                raise RuntimeError("boom")

        assert torch.backends.mha.get_fastpath_enabled() is True
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_mha_fastpath_disabled_is_a_noop_when_torch_has_no_knob(monkeypatch):
    monkeypatch.delattr(torch.backends, "mha", raising=False)

    with patches.mha_fastpath_disabled():
        pass  # documented no-op rather than an AttributeError


# --------------------------------------------------------------------------
# sdpa_math
# --------------------------------------------------------------------------

def test_sdpa_math_runs_attention_and_leaves_the_backend_restored():
    q = torch.randn(1, 2, 4, 8)

    with patches.sdpa_math():
        inside = F.scaled_dot_product_attention(q, q, q)

    outside = F.scaled_dot_product_attention(q, q, q)
    assert torch.allclose(inside, outside, atol=1e-5)


# --------------------------------------------------------------------------
# gradient_checkpointing_disabled
# --------------------------------------------------------------------------

class _CheckpointedModule(nn.Module):
    """A HuggingFace-shaped module exposing the checkpointing switches."""

    def __init__(self, enabled=True):
        super().__init__()
        self.is_gradient_checkpointing = enabled
        self.log = []

    def disable_gradient_checkpointing(self):
        self.log.append("disable")
        self.is_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.log.append("enable")
        self.is_gradient_checkpointing = True


def test_gradient_checkpointing_disabled_turns_it_off_and_back_on():
    module = _CheckpointedModule(enabled=True)

    with patches.gradient_checkpointing_disabled(module):
        assert module.is_gradient_checkpointing is False

    assert module.is_gradient_checkpointing is True
    assert module.log == ["disable", "enable"]


def test_gradient_checkpointing_disabled_leaves_an_off_module_off():
    module = _CheckpointedModule(enabled=False)

    with patches.gradient_checkpointing_disabled(module):
        pass

    assert module.log == ["disable"]


def test_gradient_checkpointing_disabled_is_a_noop_without_the_switch():
    module = nn.Linear(2, 2)

    with patches.gradient_checkpointing_disabled(module) as yielded:
        assert yielded is module


# --------------------------------------------------------------------------
# reject_adaptive_avg_pool1d
# --------------------------------------------------------------------------

def test_reject_adaptive_avg_pool1d_returns_zero_on_a_clean_tree():
    model = nn.Sequential(
        nn.Conv1d(2, 2, 1),
        _Block(nn.ReLU()),
        nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)),
    )

    assert patches.reject_adaptive_avg_pool1d(model) == 0


def test_reject_adaptive_avg_pool1d_raises_on_a_nested_pool():
    model = nn.Sequential(
        nn.Conv1d(2, 2, 1),
        nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool1d(4)),
    )

    with pytest.raises(ValueError) as excinfo:
        patches.reject_adaptive_avg_pool1d(model)

    message = str(excinfo.value)
    assert "AdaptiveAvgPool1d" in message
    assert "pool_matrix(in_len, out_len)" in message
    assert "runtime input length" in message


def test_reject_adaptive_avg_pool1d_raises_on_a_direct_child():
    model = nn.Sequential(nn.AdaptiveAvgPool1d(2))

    with pytest.raises(ValueError):
        patches.reject_adaptive_avg_pool1d(model)


def test_reject_adaptive_avg_pool1d_ignores_the_2d_and_3d_variants():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool3d(1))

    assert patches.reject_adaptive_avg_pool1d(model) == 0


# --------------------------------------------------------------------------
# replace_memory_efficient_swish
# --------------------------------------------------------------------------

def test_replace_memory_efficient_swish_swaps_and_counts_one():
    model = nn.Sequential(nn.Conv1d(2, 2, 1), _MemoryEfficientSwish())

    replaced = patches.replace_memory_efficient_swish(model)

    assert replaced == 1
    assert isinstance(model[1], nn.SiLU)


def test_replace_memory_efficient_swish_reaches_nested_blocks():
    model = nn.Sequential(
        _Block(_MemoryEfficientSwish()),
        _Block(_Swish()),
        _Block(nn.ReLU()),
    )

    replaced = patches.replace_memory_efficient_swish(model)

    assert replaced == 2
    assert isinstance(model[0].act, nn.SiLU)
    assert isinstance(model[1].act, nn.SiLU)
    assert isinstance(model[2].act, nn.ReLU)


def test_replace_memory_efficient_swish_is_numerically_identical():
    model = nn.Sequential(_MemoryEfficientSwish())
    x = torch.randn(4, 6)
    before = model(x)

    patches.replace_memory_efficient_swish(model)

    assert torch.allclose(model(x), before, atol=1e-6)


def test_replace_memory_efficient_swish_counts_zero_on_a_clean_tree():
    model = nn.Sequential(nn.Conv1d(2, 2, 1), nn.SiLU(), _Block(nn.GELU()))

    assert patches.replace_memory_efficient_swish(model) == 0


# --------------------------------------------------------------------------
# bake_pos_embed
# --------------------------------------------------------------------------

def test_bake_pos_embed_raises_on_a_module_without_the_method():
    with pytest.raises(AttributeError) as excinfo:
        patches.bake_pos_embed(nn.Linear(4, 4), (224, 224), 16)

    message = str(excinfo.value)
    assert "interpolate_pos_encoding" in message
    assert "Linear" in message


def test_bake_pos_embed_replaces_the_method_with_a_constant():
    vit = _FakeViT(dim=8)

    returned = patches.bake_pos_embed(vit, (32, 64), patch_size=16,
                                      num_register_tokens=0, embed_dim=8)

    assert returned is vit
    # one probe call during baking: (w, h) == (64, 32)
    assert vit.calls == [(64, 32)]

    baked = vit.interpolate_pos_encoding(torch.zeros(1, 3, 8), 999, 111)
    # the replacement never calls through, so the recorded calls do not grow
    assert vit.calls == [(64, 32)]
    # 1 + (32 // 16) * (64 // 16) = 9 tokens of width 8
    assert tuple(baked.shape) == (1, 9, 8)
    assert torch.allclose(baked, torch.full((1, 9, 8), 64.0 + 32.0 / 1000.0))


def test_bake_pos_embed_returns_the_same_constant_for_any_size_arguments():
    vit = _FakeViT(dim=8)
    patches.bake_pos_embed(vit, (32, 32), patch_size=16, embed_dim=8)

    first = vit.interpolate_pos_encoding(torch.zeros(1, 1, 1), 7, 7)
    second = vit.interpolate_pos_encoding(torch.zeros(1, 400, 8), 512, 384)

    assert torch.equal(first, second)
    assert tuple(first.shape) == (1, 1 + 2 * 2, 8)


def test_bake_pos_embed_counts_register_tokens_into_the_sequence():
    vit = _FakeViT(dim=8)
    patches.bake_pos_embed(vit, (32, 32), patch_size=16,
                           num_register_tokens=4, embed_dim=8)

    baked = vit.interpolate_pos_encoding(torch.zeros(1, 1, 1), 0, 0)

    assert tuple(baked.shape) == (1, 1 + 4 + 4, 8)


def test_bake_pos_embed_reads_the_width_from_pos_embed_when_not_given():
    vit = _FakeViT(dim=11)
    patches.bake_pos_embed(vit, (32, 32), patch_size=16)

    baked = vit.interpolate_pos_encoding(torch.zeros(1, 1, 1), 0, 0)

    assert baked.shape[-1] == 11


def test_bake_pos_embed_registers_a_non_persistent_buffer():
    vit = _FakeViT(dim=8)
    patches.bake_pos_embed(vit, (32, 32), patch_size=16, embed_dim=8)

    assert "_baked_pos_embed" in dict(vit.named_buffers())
    # non-persistent: it is derived output, not checkpoint state
    assert "_baked_pos_embed" not in vit.state_dict()


def test_bake_pos_embed_keeps_the_original_keyword_signature():
    """The replacement must be a drop-in for the method it overwrites.

    DINOv2-family forwards call ``interpolate_pos_encoding(x, w, h)`` and the
    HuggingFace ports call it with keywords. A replacement that only accepts
    positional arguments raises a TypeError from inside ``torch.onnx.export``,
    which presents as "the positional-embedding pre-bake did not take".
    """
    vit = _FakeViT(dim=8)
    reference = vit.interpolate_pos_encoding(torch.zeros(1, 5, 8), w=32, h=32)
    assert tuple(reference.shape) == (1, 5, 8)

    patches.bake_pos_embed(vit, (32, 32), patch_size=16, embed_dim=8)

    by_keyword = vit.interpolate_pos_encoding(torch.zeros(1, 5, 8), w=32, h=32)
    positionally = vit.interpolate_pos_encoding(torch.zeros(1, 5, 8), 32, 32)
    assert torch.equal(by_keyword, positionally)


def test_bake_pos_embed_does_not_leave_a_grad_on_the_constant():
    vit = _FakeViT(dim=8)
    patches.bake_pos_embed(vit, (32, 32), patch_size=16, embed_dim=8)

    baked = vit.interpolate_pos_encoding(torch.zeros(1, 1, 1), 0, 0)

    assert baked.requires_grad is False
