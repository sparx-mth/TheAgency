"""The flow-matching Euler schedule InternVLA-N1 System 1 denoises with, in numpy.

``generate_traj`` builds a ``diffusers.FlowMatchEulerDiscreteScheduler`` with
``sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)`` and
``set_timesteps``, then calls ``scheduler.step`` once per iteration. That whole
object reduces to two lines of arithmetic:

* the timestep handed to the network is ``sigma * num_train_timesteps``, with
  ``num_train_timesteps = 1000``;
* the update is ``sample + (sigma_next - sigma) * velocity``, computed in float32
  and cast back -- there is no noise term, so the loop is deterministic given the
  initial sample.

Reproducing it here rather than importing diffusers keeps the deployment runtime
numpy-only and means the engine path and the torch reference cannot drift apart
through a library upgrade.

Numpy-only, and Python-3.8 clean.
"""
from __future__ import annotations

#: ``FlowMatchEulerDiscreteScheduler``'s ``num_train_timesteps`` default. The
#: network is conditioned on ``sigma * this``, not on the step index.
NUM_TRAIN_TIMESTEPS = 1000.0


def sigmas(steps):
    """The sigma ladder for ``steps`` Euler steps, terminating at 0.0.

    Args:
        steps: number of inference steps. Must be at least 2 -- upstream's
            ``linspace(1.0, 1/steps, steps)`` is undefined as a ladder below
            that, and a one-step flow match is a different algorithm.

    Returns:
        List[float] of length ``steps + 1``: ``linspace(1.0, 1/steps, steps)``
        followed by the terminal ``0.0``.

    Raises:
        ValueError: when ``steps`` is below 2.
    """
    steps = int(steps)
    if steps < 2:
        raise ValueError(
            "flow-matching needs at least 2 steps, got %d; upstream builds its "
            "ladder with linspace(1.0, 1/steps, steps)" % steps)
    span = (1.0 - 1.0 / steps) / (steps - 1)
    ladder = [1.0 - i * span for i in range(steps)]
    ladder.append(0.0)
    return ladder


def schedule(steps):
    """Yield ``(sigma, sigma_next, timestep)`` for each Euler step.

    Args:
        steps: number of inference steps.

    Yields:
        Tuple[float, float, float]: the current sigma, the next one, and the
        timestep to condition the network on.
    """
    ladder = sigmas(steps)
    for i in range(int(steps)):
        yield ladder[i], ladder[i + 1], ladder[i] * NUM_TRAIN_TIMESTEPS


def euler_step(sample, velocity, sigma, sigma_next):
    """One Euler update: ``sample + (sigma_next - sigma) * velocity``.

    Computed in float32 regardless of the inputs' dtype, matching diffusers,
    which upcasts before the update "to avoid precision issues".

    Args:
        sample: current trajectory estimate, ``(B, T, 3)``.
        velocity: the network's prediction for this step, same shape.
        sigma: current sigma.
        sigma_next: the next one.

    Returns:
        numpy.ndarray: the updated sample, float32.

    Raises:
        ValueError: on a shape mismatch, which would otherwise broadcast into a
            silently wrong trajectory.
    """
    import numpy as np

    sample = np.asarray(sample, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if sample.shape != velocity.shape:
        raise ValueError(
            "euler_step got sample %s and velocity %s; they must match"
            % (sample.shape, velocity.shape))
    return sample + np.float32(sigma_next - sigma) * velocity
