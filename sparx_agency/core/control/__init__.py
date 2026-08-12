"""The airframe control chain: a planned trajectory in, an actuator command out.

This is the layer below planning and above the autopilot. It is separate from
``core.planning.trackers`` on purpose, and the boundary is what each one's output
*means*:

* ``core.planning.trackers`` produce **navigation** commands -- a twist, a
  velocity, a heading -- for a vehicle whose autopilot will work out how to
  achieve them. Those are the FALCON ``controller:=`` modes.
* ``core.control`` produces **airframe** commands -- an acceleration, an
  attitude, a thrust -- by reasoning about the vehicle as a body with mass.

Three packages, one per stage of the cascade:

``trajectory_tracking``
    A trajectory and a measured state in, a **desired acceleration and heading**
    out. Feedforward from the plan plus feedback from the error.

``flatness``
    A desired acceleration and heading in, an **attitude and a specific thrust**
    out. Pure geometry: a multirotor accelerates only by tilting, so the wanted
    acceleration *is* the wanted attitude.

``thrust_model``
    A specific thrust in, a **normalized throttle** out, with the scale learned
    in flight because it moves with battery voltage.

Everything here is pure numpy, ROS-free and Python 3.8-clean.

.. code-block:: text

    trajectory + state  ->  trajectory_tracking  ->  acceleration + yaw
                                                          |
                                                       flatness
                                                          v
                                              attitude + specific thrust
                                                          |
                                                     thrust_model
                                                          v
                                              attitude + normalized thrust
"""
