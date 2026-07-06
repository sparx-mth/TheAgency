"""Pose-free, pixel-goal fine-tuning: label generation, dataset, short trainer, eval.

This is the *pixel-goal* training path (distinct from the pose-based
``datasets.esdf_label_gen`` / ``datasets.flight_dataset``): it needs no
``poses.npy``. For each frame we sample many pixel goals, run NavDP, and push its
own trajectory off the walls -- that corrected+smoothed path is the behaviour-
cloning label. Runs in the ``navdp`` conda env (labels use TensorRT NavDP; the
trainer uses the torch NavDP model).
"""
