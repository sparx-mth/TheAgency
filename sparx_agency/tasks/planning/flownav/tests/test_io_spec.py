"""The static IO spec must stay consistent with the core runtime's tensor names.

These run torch-free (numpy not even needed), so they catch a name/shape drift
between the exporter and ``FlowNavTRTPolicy`` without a GPU or the FlowNav model.
"""
from sparx_agency.core.planning.flownav.trt import policy as core_policy
from sparx_agency.tasks.planning.flownav.export import io_spec


def test_geometry_constants():
    assert io_spec.OBS_CH == 3 * (io_spec.CTX + 1) == 12
    assert io_spec.IMG == 96
    assert io_spec.HORIZON == 8
    assert io_spec.ACT_DIM == 2
    assert io_spec.COND == 256
    assert io_spec.N >= 1


def test_encoder_io_matches_core_names():
    ins, outs, shapes = io_spec.SPECS[io_spec.ENCODER]
    assert ins == [core_policy.ENC_IN_OBS, core_policy.ENC_IN_GOAL]
    assert outs == [core_policy.ENC_OUT]
    assert shapes[core_policy.ENC_IN_OBS] == (1, io_spec.OBS_CH, io_spec.IMG, io_spec.IMG)
    assert shapes[core_policy.ENC_OUT] == (1, io_spec.COND)


def test_vfield_io_matches_core_names():
    ins, outs, shapes = io_spec.SPECS[io_spec.VFIELD]
    assert ins == [core_policy.VF_IN_SAMPLE, core_policy.VF_IN_TIME, core_policy.VF_IN_COND]
    assert outs == [core_policy.VF_OUT]
    assert shapes[core_policy.VF_IN_SAMPLE] == (io_spec.N, io_spec.HORIZON, io_spec.ACT_DIM)
    assert shapes[core_policy.VF_IN_TIME] == (1,)
    assert shapes[core_policy.VF_IN_COND] == (io_spec.N, io_spec.COND)
    assert shapes[core_policy.VF_OUT] == (io_spec.N, io_spec.HORIZON, io_spec.ACT_DIM)


def test_dist_io_matches_core_names():
    ins, outs, shapes = io_spec.SPECS[io_spec.DIST]
    assert ins == [core_policy.DIST_IN_COND]
    assert outs == [core_policy.DIST_OUT]
    assert shapes[core_policy.DIST_OUT] == (1, 1)


def test_accessors():
    for key in (io_spec.ENCODER, io_spec.VFIELD, io_spec.DIST):
        assert set(io_spec.input_names(key)) | set(io_spec.output_names(key)) \
            == set(io_spec.shapes(key))
