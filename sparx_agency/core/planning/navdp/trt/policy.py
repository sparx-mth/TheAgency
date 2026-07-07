"""TensorRT drop-in for NavDP's point-goal policy.

``NavDPTRTPolicy`` exposes the SAME ``predict_pointgoal_action`` signature and
4-tuple return as ``NavDP_Policy`` (the external PyTorch model), so the FALCON
server's ``NavDP_Agent.step_pointgoal`` -- including its 8-frame memory queue,
left zero-padding, and ``process_image``/``process_depth`` preprocessing -- runs
byte-unchanged on top of it. Only the three heavy transformer forward passes are
replaced by TensorRT engines:

    encoder  : (images (1,8,3,224,224), depth (1,1,224,224)) -> rgbd_embed (1,128,384)
    denoise  : (last_actions (16,24,3), time_token (16,1,384), goal_embed (16,1,384),
                rgbd_embed (16,128,384)) -> noise_pred (16,24,3)        [run 10x]
    critic   : (predict_trajectory (16,24,3), rgbd_embed (16,128,384)) -> critic (16,1)

Everything stochastic or data-dependent stays in numpy here (the DDPM scheduler,
the sample fan-out, the cumsum/zeroing/ranking), so the result matches the
PyTorch reference up to the blessed FP16/INT8 precision of the engines. The class
is numpy-only at import; the engine runner lazy-imports TensorRT/pycuda.

This is single-drone only: ``batch_size`` (the leading dim of ``input_images``)
must be 1, because the denoise/critic engines are built static at N = sample_num.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from sparx_agency.core.planning.navdp.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.navdp.trt.errors import NavDPError
from sparx_agency.core.planning.navdp.trt.point_encoder import NavDPPointEncoder
from sparx_agency.core.planning.navdp.trt.postprocess import finalize_trajectories
from sparx_agency.core.planning.navdp.trt.scheduler import NumpyDDPMScheduler

# Engine IO tensor names -- the ONNX export MUST use exactly these (the runner
# binds tensors by name). Centralized here so the contract has one source.
ENC_IN_IMAGES = "images"
ENC_IN_DEPTH = "depth"
ENC_OUT = "rgbd_embed"
DEN_IN_ACTIONS = "last_actions"
DEN_IN_TIME = "time_token"
DEN_IN_GOAL = "goal_embed"
DEN_IN_RGBD = "rgbd_embed"
DEN_OUT = "noise_pred"
CRI_IN_TRAJ = "predict_trajectory"
CRI_IN_RGBD = "rgbd_embed"
CRI_OUT = "critic_values"


class NavDPTRTPolicy:
    """Numpy + TensorRT re-implementation of NavDP point-goal inference.

    Args:
        engine_dir: directory containing ``selected.json`` (precision choice +
            engine filenames) and the three ``.engine`` files.
        head_params_npz: path to the exported head params (``point_encoder``
            weight/bias, the 10-row sinusoidal time table, ``alphas_cumprod``).
        sample_num: diffusion samples per goal (engines are built for this N).
        predict_size: trajectory horizon (24).
        device_id: CUDA device index.

    Raises:
        NavDPError: any engine, manifest, or head-param file missing/inconsistent.
    """

    def __init__(self, engine_dir, head_params_npz, sample_num=16,
                 predict_size=24, device_id=0):
        self.engine_dir = Path(engine_dir)
        self.sample_num = int(sample_num)
        self.predict_size = int(predict_size)
        self.device_id = int(device_id)

        sel_path = self.engine_dir / "selected.json"
        if not sel_path.exists():
            raise NavDPError("selected.json not found in %s (run the benchmark/"
                             "accuracy gate to choose a precision)" % self.engine_dir)
        sel = json.loads(sel_path.read_text())
        engines = sel.get("engines", {})
        self.precision = sel.get("precision", "?")
        self._enc = self._runner(engines, "encoder")
        self._den = self._runner(engines, "denoise")
        self._cri = self._runner(engines, "critic")

        self._load_head_params(head_params_npz)

    def _runner(self, engines, key):
        """Build a :class:`TRTEngineRunner` for the named engine in selected.json."""
        if key not in engines:
            raise NavDPError("selected.json missing engine %r" % key)
        return TRTEngineRunner(self.engine_dir / engines[key], device_id=self.device_id)

    @staticmethod
    def _infer_out(runner, feeds, out_name):
        """Run ``runner`` and return its ``out_name`` output, raising a clear
        :class:`NavDPError` (not a bare ``KeyError``) if the engine was built with
        a differently-named output than the runtime expects."""
        out = runner.infer(feeds)
        if out_name not in out:
            raise NavDPError("engine %s produced no output %r; got %r"
                             % (runner.engine_path.name, out_name, list(out)))
        return out[out_name]

    def _load_head_params(self, npz_path):
        """Load point-encoder weights, the time table, and alphas_cumprod."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise NavDPError("head params npz not found: %s" % npz_path)
        p = np.load(npz_path)
        for key in ("point_encoder_weight", "point_encoder_bias",
                    "time_table", "alphas_cumprod"):
            if key not in p:
                raise NavDPError("head params npz %s missing %r" % (npz_path, key))
        self.point_encoder = NavDPPointEncoder(p["point_encoder_weight"],
                                               p["point_encoder_bias"])
        self.time_table = np.asarray(p["time_table"], dtype=np.float32)  # (T, 384)
        self.scheduler = NumpyDDPMScheduler(p["alphas_cumprod"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_pointgoal_action(self, goal_point, input_images, input_depths,
                                 sample_num=16, init_noise=None, variance_noises=None):
        """Point-goal inference; drop-in for ``NavDP_Policy.predict_pointgoal_action``.

        Args:
            goal_point: ``(1, 3)`` body-frame goal (forward, left, 0), host-clipped.
            input_images: ``(1, 8, 224, 224, 3)`` RGB memory stack in [0, 1].
            input_depths: ``(1, 224, 224, 1)`` current-frame metric depth.
            sample_num: diffusion samples (must equal the built engine N).
            init_noise: optional ``(N, 24, 3)`` initial noise (gate injection).
            variance_noises: optional ``(steps, N, 24, 3)`` per-step variance
                noise (gate injection); ``None`` -> fresh draws (production).

        Returns:
            ``(all_trajectory, critic_values, positive_trajectory,
            negative_trajectory)`` numpy arrays, identical layout to the
            reference.
        """
        if sample_num != self.sample_num:
            raise NavDPError("sample_num=%d but engines built for N=%d"
                             % (sample_num, self.sample_num))
        if input_images.shape[0] != 1:
            raise NavDPError("NavDPTRTPolicy is single-drone; batch=%d not supported"
                             % input_images.shape[0])

        rgbd_embed = self._encode(input_images, input_depths)              # (1,128,384)
        goal_embed = self.point_encoder(np.asarray(goal_point, np.float32))  # (1,384)
        rgbd_n = np.repeat(rgbd_embed, self.sample_num, axis=0)            # (N,128,384)
        goal_n = np.repeat(goal_embed[:, None, :], self.sample_num, axis=0)  # (N,1,384)

        naction = self._denoise_loop(rgbd_n, goal_n, init_noise, variance_noises)
        critic = self._infer_out(self._cri, {CRI_IN_TRAJ: naction, CRI_IN_RGBD: rgbd_n}, CRI_OUT)
        return finalize_trajectories(naction, critic.reshape(-1), 1, self.sample_num)

    def _encode(self, input_images, input_depths):
        """Host-permute to NCHW and run the encoder engine -> ``(1,128,384)``."""
        images = np.ascontiguousarray(
            np.transpose(input_images, (0, 1, 4, 2, 3)), dtype=np.float32)  # (1,8,3,224,224)
        depth = np.ascontiguousarray(
            np.transpose(input_depths, (0, 3, 1, 2)), dtype=np.float32)     # (1,1,224,224)
        return self._infer_out(self._enc, {ENC_IN_IMAGES: images, ENC_IN_DEPTH: depth}, ENC_OUT)

    def _denoise_loop(self, rgbd_n, goal_n, init_noise, variance_noises):
        """Run the 10-step DDPM loop; conditioning is uploaded once and resident."""
        shape = (self.sample_num, self.predict_size, 3)
        naction = (np.random.randn(*shape).astype(np.float32)
                   if init_noise is None else np.asarray(init_noise, np.float32))
        # Conditioning is constant across all 10 steps: upload once, leave resident.
        self._den.upload({DEN_IN_RGBD: rgbd_n, DEN_IN_GOAL: goal_n})
        for step, k in enumerate(self.scheduler.timesteps):
            k = int(k)
            noise_pred = self._infer_out(
                self._den, {DEN_IN_ACTIONS: naction, DEN_IN_TIME: self._time_token(k)}, DEN_OUT)
            vn = None if variance_noises is None else variance_noises[step]
            naction = self.scheduler.step(noise_pred, k, naction, variance_noise=vn)
        return naction

    def _time_token(self, timestep):
        """Tiled sinusoidal time embedding ``(N, 1, 384)`` for one timestep."""
        row = self.time_table[timestep][None, None, :]
        return np.repeat(row, self.sample_num, axis=0).astype(np.float32)
