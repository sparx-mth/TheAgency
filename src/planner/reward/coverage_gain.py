"""
Reward function that gives **step-wise coverage-gain feedback** for multi-agent
SLAM coverage tasks.

    Mathematical definition
    -----------------------
    Let

    * ``c_max``   total number of discoverable cells.
    * ``c_t``     number of *new* cells revealed at step *t*.
    * ``r_t``     fraction of observations at step *t* that were redundant.
    * ``d_t``     average pair-wise drone spacing **beyond** a threshold
                   (zero if spacing ≤ threshold).
    * ``coll_pen_t``  –1 if any collision occurred at this step, else 0.
    * ``done``    indicator that full coverage is achieved.

    The per-step reward is::

        reward_t =  alpha * (c_t / c_max)     # coverage gain
                    - beta                    # time penalty
                    - lambda_ * r_t           # redundancy penalty
                    + delta * d_t             # dispersion bonus
                    + gamma * coll_pen_t      # safety penalty
                    + r_finish * 1[done]      # terminal bonus

    Hyper-parameters (``alpha``, ``beta``, ``lambda_``, ``delta``,
    ``gamma``, ``r_finish``) are supplied via **kwargs**.

    Typical starting values
    -----------------------
    ==================  ============
    Parameter            Default
    ==================  ============
    ``alpha``            1 / c_max
    ``beta``             0.02
    ``lambda_``          0.3
    ``delta``            0.02
    ``gamma``            1.0
    ``r_finish``         10.0
    ==================  ============

Implementation notes
--------------------
* ``reset`` caches ``C_max`` at episode start.
* ``step`` receives a diagnostics dictionary produced by
  :pymeth:`planner.simulation.master_controller.MasterController.step`.
  The reward *never* mutates environment state—it is a pure function of
  the post-step world plus diagnostics.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np
from .base import RewardFunction
if TYPE_CHECKING:
    from src.planner.simulation.grid_map_env import GridMapEnv


class CoverageGainReward(RewardFunction):
    """
    Dense coverage-gain reward for multi-drone SLAM.

    Parameters
    ----------
    alpha : float, optional
        Weight for fractional coverage gain per step.
    beta : float, optional
        Constant time penalty each tick.
    lambda_ : float, optional
        Weight for redundancy penalty (observing already-known cells).
    delta : float, optional
        Weight for dispersion bonus (spreading drones apart).
    gamma : float, optional
        Weight for collision penalty.
    r_finish : float, optional
        Terminal bonus added once when mapping is complete.

    Notes
    -----
    * All parameters are passed via **kwargs; missing ones fall back to sensible
      defaults so that the reward works “out of the box”.
    * The class is entirely self-contained: it does no logging, keeps no
      references between episodes except cached map size, and touches only the
      public environment attributes (`discoverable_mask`).
    """
    def __init__(self, **params):
        super().__init__(**params)
        # Placeholder; real value is set in reset().
        self.c_max: int = 1

    # ------------------------------------------------------------------ #
    # Episode-level bookkeeping
    # ------------------------------------------------------------------ #
    def reset(self, env:"GridMapEnv"):
        """Cache the number of discoverable cells for normalisation."""
        self.c_max = np.count_nonzero(env.discoverable_mask)
        self.c_max = max(1, self.c_max)               # guard /0

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def step(self, env:"GridMapEnv", info: Dict[str, Any]) -> float:
        """
        Compute reward for the current tick.

        Parameters
        ----------
        env : GridMapEnv
            Reference to the **post-step** environment (unused here but kept
            for interface consistency).
        info : dict
            Diagnostics dict with keys ``new_cells``, ``redundant``,
            ``collisions``, ``avg_distance``, ``done``.

        Returns
        -------
        float
            Scalar reward to feed into the RL algorithm.
        """
        alpha     = self.params.get("alpha",   1.0 / self.c_max)
        beta      = self.params.get("beta",    0.02)
        lambda_   = self.params.get("lambda_", 0.3)
        delta     = self.params.get("delta",   0.02)
        gamma     = self.params.get("gamma",   1.0)
        r_finish  = self.params.get("r_finish", 10.0)

        # ---------------- components ---------------- #
        delta_c_t = info["new_cells"] / self.c_max

        total_obs = info["new_cells"] + info["redundant"]
        r_t       = (info["redundant"] / total_obs) if total_obs else 0.0

        d_t           = info["avg_distance"]          # already thresholded
        coll_penalty  = -1.0 if info["collisions"] else 0.0
        done_bonus    = r_finish if info["done"] else 0.0

        # ---------------- final reward -------------- #
        return float(
            alpha * delta_c_t
            - beta
            - lambda_ * r_t
            + delta * d_t
            + gamma * coll_penalty
            + done_bonus
        )
