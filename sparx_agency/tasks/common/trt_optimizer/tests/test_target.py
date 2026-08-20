"""Tests for the build-toolchain target: what this machine can actually build.

Every :class:`Target` here is constructed by hand rather than probed, which is
the whole point of the dataclass being separate from :func:`resolve`: the
TensorRT-10 cases (weak typing, an INT8 entropy calibrator, DLA still alive)
have to be exercised on a workstation whose only TensorRT is 11.1, and the
"TensorRT is not importable" case has to be exercised on a machine where it is.

The one test that probes for real is :func:`test_resolve_on_this_machine`,
guarded with ``importorskip`` so it skips in ``.venv`` and runs under the navdp
conda interpreter, where it pins the single fact the rest of the package is
built around: this TensorRT is strongly typed.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.common.hardware.detect import HardwareProfile
from sparx_agency.tasks.common.trt_optimizer import target as T

MIB = 1 << 20
GIB = 1 << 30


# --------------------------------------------------------------------------
# hand-built hardware and targets
# --------------------------------------------------------------------------

def rtx5070():
    """The dev workstation: RTX 5070 Laptop, 8151 MiB, Blackwell sm_120."""
    return HardwareProfile(
        arch="x86_64", is_jetson=False,
        gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
        compute_capability=(12, 0), total_mem_bytes=8151 * MIB,
        recommended_workspace_bytes=4 * GIB, target_tag="nvidiageforcert_sm120")


def orin():
    """A 32 GB AGX Orin: unified memory, two DLA cores, sm_87."""
    return HardwareProfile(
        arch="aarch64", is_jetson=True, gpu_name="NVIDIA Jetson AGX Orin",
        jetson_model="NVIDIA Jetson AGX Orin", nvpmodel_id=0,
        nvpmodel_name="MODE_50W", power_budget_w=50, compute_capability=(8, 7),
        total_mem_bytes=32 * GIB, dla_cores=2, allow_dla=True,
        recommended_workspace_bytes=2 * GIB, target_tag="orin_sm87")


def trt11(toolchain=(), missing=(), hardware=None):
    """This machine: TensorRT 11.1, strongly typed, no DLA, clocks locked out."""
    return T.Target(hardware=hardware or rtx5070(), trt_version="11.1.0.106",
                    cuda_driver_version="580.65.06", torch_version="2.11.0+cu128",
                    free_mem_bytes=7613 * MIB, strongly_typed=True,
                    dla_usable=False, clocks_lockable=False,
                    toolchain=list(toolchain), missing=list(missing))


def trt10(toolchain=(), missing=(), hardware=None):
    """An Orin-generation toolchain: weak typing, DLA alive, calibrator alive."""
    return T.Target(hardware=hardware or orin(), trt_version="10.7.0.23",
                    cuda_driver_version="540.4.0", torch_version="2.5.0",
                    free_mem_bytes=20 * GIB, strongly_typed=False,
                    dla_usable=True, clocks_lockable=True,
                    toolchain=list(toolchain), missing=list(missing))


ONNXCC = "onnxconverter_common"
MISSING_ONNXCC = ("onnxconverter_common -- would enable FP16 graph conversion, "
                  "which is the ONLY way to get FP16 on TensorRT 11")
MISSING_MODELOPT = ("modelopt -- would enable BF16/INT8/FP8/NVFP4 quantized "
                    "ONNX (Q/DQ)")


# --------------------------------------------------------------------------
# version parsing
# --------------------------------------------------------------------------

def test_trt_major_minor_parses_the_four_part_version_on_this_machine():
    assert trt11().trt_major_minor == (11, 1)


@pytest.mark.parametrize("version,expected", [
    ("11.1.0.106", (11, 1)),
    ("10.7.0.23", (10, 7)),
    ("8.6.1", (8, 6)),
    ("10.13", (10, 13)),
])
def test_trt_major_minor_parses_every_shipped_version_shape(version, expected):
    assert T.Target(hardware=rtx5070(), trt_version=version).trt_major_minor \
        == expected


def test_trt_major_minor_is_none_without_tensorrt():
    """None means 'not importable', and must not be guessed into a number."""
    assert T.Target(hardware=rtx5070()).trt_major_minor is None
    assert T.Target(hardware=rtx5070(), trt_version="").trt_major_minor is None


def test_trt_major_minor_is_none_for_an_unparseable_version():
    assert T.Target(hardware=rtx5070(),
                    trt_version="unknown").trt_major_minor is None


def test_last_dla_trt_is_the_documented_cutoff():
    """10.7 was the last release supporting DLA; 11.x dropped it."""
    assert T.LAST_DLA_TRT == (10, 7)
    assert trt10().trt_major_minor <= T.LAST_DLA_TRT
    assert trt11().trt_major_minor > T.LAST_DLA_TRT


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def test_target_tag_comes_from_the_hardware():
    assert trt11().target_tag == "nvidiageforcert_sm120"
    assert trt10().target_tag == "orin_sm87"


def test_engine_identity_is_tag_version_and_sm():
    """An engine deserializes only under this exact triple -- patch included."""
    assert trt11().engine_identity == ("nvidiageforcert_sm120", "11.1.0.106", 120)


def test_engine_identity_changes_with_a_patch_level_bump():
    """JetPack point releases bump the patch, and that invalidates the plan."""
    before = trt10().engine_identity
    after = T.Target(hardware=orin(), trt_version="10.7.0.24").engine_identity
    assert before != after


def test_engine_identity_reports_a_missing_tensorrt_as_a_string():
    assert T.Target(hardware=rtx5070()).engine_identity[1] == "None"


# --------------------------------------------------------------------------
# supported_precisions -- the four toolchain combinations
# --------------------------------------------------------------------------

def test_strongly_typed_with_no_converters_can_only_build_fp32():
    """TensorRT 11 has no precision flags, so bare it is FP32 and nothing else."""
    assert trt11().supported_precisions() == ["fp32"]


def test_strongly_typed_gains_fp16_only_from_onnxconverter_common():
    assert trt11(toolchain=[ONNXCC]).supported_precisions() == ["fp32", "fp16"]


def test_strongly_typed_without_modelopt_does_not_offer_int8():
    """The IInt8Calibrator is gone on 11, so INT8 needs ModelOpt Q/DQ nodes."""
    assert "int8" not in trt11(toolchain=[ONNXCC]).supported_precisions()


def test_weakly_typed_without_modelopt_still_offers_int8_via_its_calibrator():
    """TensorRT <= 10 ships the entropy calibrator, so INT8 is reachable."""
    precisions = trt10().supported_precisions()
    assert precisions == ["fp32", "int8"]


def test_weakly_typed_with_the_fp16_converter_offers_fp16_and_int8():
    precisions = trt10(toolchain=[ONNXCC]).supported_precisions()
    assert precisions == ["fp32", "fp16", "int8"]


def test_modelopt_unlocks_the_quantized_formats_on_a_strongly_typed_target():
    precisions = trt11(toolchain=[ONNXCC, "modelopt"]).supported_precisions()
    assert precisions == ["fp32", "fp16", "bf16", "int8", "fp8", "nvfp4"]


def test_nvfp4_is_offered_only_on_blackwell():
    """sm_120 has the NVFP4 tensor cores; the Orin's sm_87 does not."""
    assert "nvfp4" in trt11(toolchain=["modelopt"]).supported_precisions()
    assert "nvfp4" not in trt10(toolchain=["modelopt"]).supported_precisions()


def test_nvfp4_is_withheld_when_the_compute_capability_is_unknown():
    """An unknown SM must not be read as 'new enough'."""
    unknown = HardwareProfile(arch="x86_64", is_jetson=False,
                              total_mem_bytes=8 * GIB, target_tag="unknown_smx")
    precisions = trt11(toolchain=["modelopt"],
                       hardware=unknown).supported_precisions()
    assert "nvfp4" not in precisions
    assert "int8" in precisions


def test_modelopt_does_not_duplicate_int8_on_a_weakly_typed_target():
    precisions = trt10(toolchain=["modelopt"]).supported_precisions()
    assert precisions.count("int8") == 1


def test_fp32_is_always_first_and_always_available():
    for target in (trt11(), trt11(toolchain=[ONNXCC, "modelopt"]), trt10()):
        assert target.supported_precisions()[0] == "fp32"


# --------------------------------------------------------------------------
# require_buildable -- the one function that fails loud
# --------------------------------------------------------------------------

def test_require_buildable_raises_without_tensorrt():
    target = T.Target(hardware=rtx5070(), toolchain=[ONNXCC])
    with pytest.raises(RuntimeError) as excinfo:
        T.require_buildable(target, "fp16")
    message = str(excinfo.value)
    assert "TensorRT is not importable" in message
    assert "navdp" in message          # names the interpreter that does have it


def test_require_buildable_raises_naming_the_missing_tool():
    target = trt11(missing=[MISSING_ONNXCC, MISSING_MODELOPT])
    with pytest.raises(RuntimeError) as excinfo:
        T.require_buildable(target, "fp16")
    message = str(excinfo.value)
    assert "Cannot build fp16" in message
    assert "onnxconverter_common" in message
    assert "supported: fp32" in message


def test_require_buildable_raises_for_int8_on_a_strongly_typed_target():
    target = trt11(toolchain=[ONNXCC], missing=[MISSING_MODELOPT])
    with pytest.raises(RuntimeError) as excinfo:
        T.require_buildable(target, "int8")
    assert "modelopt" in str(excinfo.value)


def test_require_buildable_says_none_when_nothing_is_missing():
    target = trt11(toolchain=[ONNXCC])
    with pytest.raises(RuntimeError) as excinfo:
        T.require_buildable(target, "int4")
    assert "Missing: none" in str(excinfo.value)


def test_require_buildable_passes_for_a_precision_the_toolchain_has():
    assert T.require_buildable(trt11(toolchain=[ONNXCC]), "fp16") is None
    assert T.require_buildable(trt11(), "fp32") is None
    assert T.require_buildable(trt10(), "int8") is None


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------

def test_describe_reports_the_gpu_tag_and_precisions():
    text = T.describe(trt11(toolchain=[ONNXCC]))
    assert "NVIDIA GeForce RTX 5070 Laptop GPU" in text
    assert "sm_120" in text
    assert "nvidiageforcert_sm120" in text
    assert "fp32, fp16" in text
    assert "11.1.0.106" in text
    assert "strong typing" in text


def test_describe_reports_memory_in_gibibytes():
    """8151 MiB total, 7613 MiB free -- the measured numbers on this laptop."""
    text = T.describe(trt11())
    assert "8.0 GiB total" in text
    assert "7.4 GiB free" in text


def test_describe_warns_when_clocks_cannot_be_locked():
    """The laptop cannot pin clocks without root, so A/B runs must interleave."""
    assert "NOT lockable" in T.describe(trt11())
    assert "NOT lockable" not in T.describe(trt10())


def test_describe_reports_dla_state_from_the_runtime_not_the_silicon():
    """The Orin has DLA cores at both versions; only TensorRT <= 10 can use them."""
    assert "usable, 2 cores" in T.describe(trt10())
    no_dla = T.Target(hardware=orin(), trt_version="11.1.0.106",
                      strongly_typed=True, dla_usable=False)
    assert "not usable on this runtime" in T.describe(no_dla)


def test_describe_says_not_importable_for_an_absent_toolchain():
    text = T.describe(T.Target(hardware=rtx5070()))
    assert "TensorRT:   not importable" in text
    assert "Torch:      not importable" in text
    assert "weak typing" in text


# --------------------------------------------------------------------------
# the real probe (conda interpreter only)
# --------------------------------------------------------------------------

def test_resolve_on_this_machine():
    """Probe for real: this machine's TensorRT must report strong typing.

    Skipped in ``.venv``, which has no ``tensorrt``. Under the navdp conda
    interpreter this is the fact the whole package is built around -- if it ever
    reports weak typing, the precision-baking path is being skipped and every
    engine is silently FP32.
    """
    pytest.importorskip("tensorrt")
    resolved = T.resolve()
    assert resolved.strongly_typed is True
    assert resolved.target_tag
    assert resolved.trt_version
    assert resolved.trt_major_minor[0] >= 11
    assert resolved.dla_usable is False
    assert T.describe(resolved).splitlines()[0].startswith("GPU:")
