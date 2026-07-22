"""Interactive verification of the NavDP -> instantaneous-PF/ESDF correction loop.

Before spending GPU-hours fine-tuning, this subpackage lets you *see* the training
signal on a real flight frame: click a pixel goal on the depth/colour image, run
NavDP, build the single-frame potential field / signed ESDF from the same depth,
lightly push the NavDP trajectory off the walls, and compare the original vs the
corrected trajectory on a bird's-eye map -- all in the shared body FLU frame.

Everything runs in the ``navdp`` conda env (TensorRT + numpy, no torch model load).
See ``README.md`` for the run command and the knobs.
"""
