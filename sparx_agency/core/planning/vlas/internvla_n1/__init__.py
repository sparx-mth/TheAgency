"""InternVLA-N1 dual-system VLN policy: the ROS-free wire contract.

InternVLA-N1 (`InternRobotics/InternNav <https://github.com/InternRobotics/InternNav>`_)
is a vision-language navigation policy: an RGB frame plus a natural-language
instruction produce a discrete VLN-CE action (``STOP`` / ``MOVE_FORWARD`` /
``TURN_LEFT`` / ``TURN_RIGHT``) and, when the System-2 planner fires, a pixel
sub-goal in the input image.

The model itself runs in an external InternNav checkout/conda env behind an HTTP
server; this package owns only the two ROS-free pieces:

* :mod:`~sparx_agency.core.planning.vlas.internvla_n1.client` -- the HTTP wire
  contract (``/agent/init``, ``/agent/{name}/step``, ``/agent/{name}/reset``).
* :mod:`~sparx_agency.core.planning.vlas.internvla_n1.types` -- the action
  vocabulary and the parsed step response.

The ROS2 node that drives them lives in
``tasks/planning/vlas/internvla_n1/ros2/``; the Rooster R1 / Sphera topic and
actuation binding lives in ``robots/ROBOTICAN/``.

Python 3.8 compatible; ``requests`` is imported at module scope here only
because nothing in the FALCON Noetic path imports this package -- keep it that
way, or move the import inside the methods as NavDP does.
"""
