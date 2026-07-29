"""TensorRT-backed NavDP agent: TRT policy under the unchanged preprocessing.

``NavDP_Agent`` (the external repo) owns the host-side preprocessing that must
stay byte-identical: ``process_image`` / ``process_depth`` (keep-aspect resize +
center pad + scale), the 8-frame RGB ``memory_queue`` with left zero-padding, and
the trajectory-mask rendering. ``TRTNavDPAgent`` subclasses it and replaces ONLY
``self.navi_former`` with a :class:`NavDPTRTPolicy`, whose
``predict_pointgoal_action`` has the same signature -- so the inherited
``step_pointgoal`` runs verbatim on top of the engines.

Single-drone only: ``reset`` rejects ``batch_size != 1`` because the
denoise/critic engines are built static at ``N = sample_num``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from sparx_agency.core.planning.vlas.navdp.trt.errors import NavDPError
from sparx_agency.core.planning.vlas.navdp.trt.policy import NavDPTRTPolicy
from sparx_agency.tasks.planning.vlas.navdp.trt.export.build_policy import resolve_navdp_repo


def _import_navdp_agent(navdp_repo):
    """Import ``NavDP_Agent`` from the external repo (for inherited preprocessing)."""
    repo = resolve_navdp_repo(navdp_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from policy_agent import NavDP_Agent
    return NavDP_Agent


def make_trt_agent(image_intrinsic, engine_dir, head_params_npz, navdp_repo=None,
                   image_size=224, memory_size=8, predict_size=24,
                   render_cam_height=0.2, sample_num=16, policy=None):
    """Construct a ``TRTNavDPAgent`` (factory keeps the external import lazy).

    Args:
        image_intrinsic: 3x3 camera matrix (for the trajectory-mask render only).
        engine_dir: directory with ``selected.json`` + the ``.engine`` files.
        head_params_npz: exported numpy head params.
        navdp_repo: external NavDP repo (for ``NavDP_Agent``; else ``NAVDP_REPO``).
        policy: optional pre-built :class:`NavDPTRTPolicy` (the server builds it at
            startup so engine load / version-lock fails loud before serving); a
            fresh one is constructed when ``None``.

    Returns:
        A ready ``TRTNavDPAgent`` instance.
    """
    NavDP_Agent = _import_navdp_agent(navdp_repo)

    class TRTNavDPAgent(NavDP_Agent):
        """``NavDP_Agent`` with a TensorRT policy instead of the torch model."""

        def __init__(self):
            self.image_intrinsic = image_intrinsic
            self.device = "cuda:0"
            self.predict_size = predict_size
            self.image_size = image_size
            self.memory_size = memory_size
            self.render_cam_height = render_cam_height
            self.navi_former = policy if policy is not None else NavDPTRTPolicy(
                engine_dir, head_params_npz, sample_num=sample_num,
                predict_size=predict_size)

        def reset(self, batch_size, threshold):
            if int(batch_size) != 1:
                raise NavDPError("TRTNavDPAgent is single-drone; batch_size=%s "
                                 "not supported by the static engines" % batch_size)
            super().reset(batch_size, threshold)

    return TRTNavDPAgent()
