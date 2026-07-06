"""TensorRT runtime for FlowNav image-goal inference (numpy control loop).

``FlowNavTRTPolicy`` runs FlowNav with three TensorRT engines and a pure-numpy
flow-matching loop, so the result matches the PyTorch reference up to the
engines' FP16 precision. The three engines (names bound positionally by the
exporter; see the IO-name constants below):

    encoder : (obs_img (1,Cobs,96,96), goal_img (1,3,96,96)) -> obsgoal_cond (1,256)
    vfield  : (sample (N,8,2), timestep (1,), global_cond (N,256)) -> vfield (N,8,2)   [run K-1x]
    dist    : (obsgoal_cond (1,256)) -> distance (1,1)

``Cobs = 3 * (context_size + 1)`` (12 for context_size=3). The encoder is exported
in **navigation mode** (goal used; ``input_goal_mask = 0`` baked in), so it takes
no mask input.

Everything stochastic / data-dependent stays in numpy here: the initial noise
``x0 ~ N(0, I)``, the deterministic Euler integration, and the action
de-normalization. The class is numpy-only at import; the engine runner
lazy-imports TensorRT/pycuda. Single-drone only (encoder batch = 1); the velocity
field runs at the static built sample count ``N``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from sparx_agency.core.planning.flownav.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.flownav.trt.errors import FlowNavError
from sparx_agency.core.planning.flownav.trt.postprocess import get_action
from sparx_agency.core.planning.flownav.trt.scheduler import FlowMatchEulerScheduler

# Engine IO tensor names -- the ONNX export MUST use exactly these (the runner
# binds tensors by name). Centralized here so the contract has one source.
ENC_IN_OBS = "obs_img"
ENC_IN_GOAL = "goal_img"
ENC_OUT = "obsgoal_cond"
VF_IN_SAMPLE = "sample"
VF_IN_TIME = "timestep"
VF_IN_COND = "global_cond"
VF_OUT = "vfield"
DIST_IN_COND = "obsgoal_cond"
DIST_OUT = "distance"


class FlowNavTRTPolicy:
    """Numpy + TensorRT re-implementation of FlowNav image-goal inference.

    Args:
        engine_dir: directory with ``selected.json`` (precision, engine
            filenames, built ``num_samples`` and gate-chosen ``num_steps``) and
            the three ``.engine`` files.
        head_params_npz: path to the exported head params (``action_min`` /
            ``action_max`` used to de-normalize the action deltas).
        num_steps: override the flow-matching step count K (default: the value
            chosen by the accuracy gate in ``selected.json``).
        num_samples: override the trajectory sample count N (default: the static
            N the ``vfield`` engine was built for, from ``selected.json``).
        device_id: CUDA device index.

    Raises:
        FlowNavError: any engine, manifest, head-param, or ``selected.json`` file
            missing/inconsistent, or a feed shape that does not match an engine.
    """

    def __init__(self, engine_dir, head_params_npz, num_steps=None,
                 num_samples=None, device_id=0):
        self.engine_dir = Path(engine_dir)
        self.device_id = int(device_id)

        sel_path = self.engine_dir / "selected.json"
        if not sel_path.exists():
            raise FlowNavError("selected.json not found in %s (run the benchmark/"
                               "accuracy gate to choose a precision + K)" % self.engine_dir)
        sel = json.loads(sel_path.read_text())
        engines = sel.get("engines", {})
        self.precision = sel.get("precision", "?")
        self.num_samples = int(num_samples if num_samples is not None
                               else sel.get("num_samples", 8))
        k = num_steps if num_steps is not None else sel.get("num_steps")
        if k is None:
            raise FlowNavError("num_steps (K) not given and absent from selected.json")
        self.scheduler = FlowMatchEulerScheduler(int(k))
        self.horizon = int(sel.get("horizon", 8))
        self.action_dim = int(sel.get("action_dim", 2))

        self._enc = self._runner(engines, "encoder")
        self._vf = self._runner(engines, "vfield")
        self._dist = self._runner(engines, "dist")
        self._load_head_params(head_params_npz)

    def _runner(self, engines, key):
        """Build a :class:`TRTEngineRunner` for the named engine in selected.json."""
        if key not in engines:
            raise FlowNavError("selected.json missing engine %r" % key)
        return TRTEngineRunner(self.engine_dir / engines[key], device_id=self.device_id)

    def _load_head_params(self, npz_path):
        """Load the action normalization stats used to de-normalize deltas."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FlowNavError("head params npz not found: %s" % npz_path)
        p = np.load(npz_path)
        for key in ("action_min", "action_max"):
            if key not in p:
                raise FlowNavError("head params npz %s missing %r" % (npz_path, key))
        self.action_min = np.asarray(p["action_min"], dtype=np.float32).reshape(-1)
        self.action_max = np.asarray(p["action_max"], dtype=np.float32).reshape(-1)

    @property
    def num_steps(self):
        """The flow-matching step count K (grid size of the Euler integrator)."""
        return self.scheduler.num_steps

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def encode(self, obs_img, goal_img):
        """Run the encoder engine -> conditioning embedding ``(1, 256)``.

        Args:
            obs_img: ``(1, Cobs, 96, 96)`` float32, the ImageNet-normalized
                context stack (``context_size + 1`` RGB frames on the channel
                axis), exactly as ``transform_images`` produces it.
            goal_img: ``(1, 3, 96, 96)`` float32, the normalized goal frame.

        Returns:
            ``(1, 256)`` float32 conditioning embedding.
        """
        obs = np.ascontiguousarray(obs_img, dtype=np.float32)
        goal = np.ascontiguousarray(goal_img, dtype=np.float32)
        if obs.shape[0] != 1:
            raise FlowNavError("FlowNavTRTPolicy.encode is single-drone; batch=%d"
                               % obs.shape[0])
        return self._enc.infer({ENC_IN_OBS: obs, ENC_IN_GOAL: goal})[ENC_OUT]

    def distance(self, obsgoal_cond):
        """Run the distance head -> scalar temporal distance to the goal.

        Args:
            obsgoal_cond: ``(1, 256)`` conditioning embedding from :meth:`encode`.

        Returns:
            float temporal distance (the reference uses it for topomap-node
            localization, not for ranking action samples).
        """
        out = self._dist.infer({DIST_IN_COND: np.ascontiguousarray(obsgoal_cond, np.float32)})
        return float(out[DIST_OUT].reshape(-1)[0])

    def sample_actions(self, obsgoal_cond, init_noise=None):
        """Integrate the velocity field K-1 Euler steps -> waypoint samples.

        Args:
            obsgoal_cond: ``(1, 256)`` conditioning embedding from :meth:`encode`.
            init_noise: optional ``(N, horizon, action_dim)`` initial state (gate
                injection); ``None`` -> a fresh ``N(0, I)`` draw (production).

        Returns:
            ``(N, horizon, action_dim)`` absolute waypoints (de-normalized,
            cumulatively summed).
        """
        cond = np.ascontiguousarray(obsgoal_cond, dtype=np.float32)
        cond_n = np.repeat(cond, self.num_samples, axis=0)              # (N, 256)
        shape = (self.num_samples, self.horizon, self.action_dim)
        x = (np.random.randn(*shape).astype(np.float32)
             if init_noise is None else np.asarray(init_noise, np.float32))
        # global_cond is constant across all Euler steps: upload once, resident.
        self._vf.upload({VF_IN_COND: cond_n})
        for i in range(self.scheduler.num_field_evals):
            t = np.array([self.scheduler.timesteps[i]], dtype=np.float32)
            vfield = self._vf.infer({VF_IN_SAMPLE: x, VF_IN_TIME: t})[VF_OUT]
            x = self.scheduler.step(vfield, i, x)
        return get_action(x, self.action_min, self.action_max)

    def predict(self, obs_img, goal_img, init_noise=None):
        """Full inference: encode, score distance, and sample action waypoints.

        Args:
            obs_img: ``(1, Cobs, 96, 96)`` normalized context stack.
            goal_img: ``(1, 3, 96, 96)`` normalized goal frame.
            init_noise: optional ``(N, horizon, action_dim)`` initial state.

        Returns:
            ``(actions, distance)`` where ``actions`` is
            ``(N, horizon, action_dim)`` waypoints and ``distance`` is a float.
        """
        cond = self.encode(obs_img, goal_img)
        dist = self.distance(cond)
        actions = self.sample_actions(cond, init_noise=init_noise)
        return actions, dist
