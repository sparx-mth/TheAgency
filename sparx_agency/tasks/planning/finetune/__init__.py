"""Fine-tuning infrastructure for the NavDP and FlowNav navigation policies.

Adapts the pretrained ground-robot policies to the drone's ~1.0 m viewpoint using
a small set of flight recordings, with a potential-field / ESDF signal that shapes
the output trajectory away from walls. Torch-heavy and dev/host-only -- lives under
``tasks/`` and must never be imported by ``core`` (which stays ROS-free + Py3.8).

See ``README.md`` for the architecture, loss, and training-method write-up.
"""
