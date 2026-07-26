"""Readers that recover a command's full parameter set from its own source.

One module per declaration style, because they share nothing but their answer:

* :mod:`ros2_node` -- a ROS2 node's ``declare_parameter`` calls
* :mod:`argparse_cli` -- a plain script's ``add_argument`` calls
* :mod:`roslaunch_xml` -- a launch file's ``<arg>`` declarations
* :mod:`yaml_config` -- a hand-commented YAML config

:mod:`pysource` is the shared Python-reading helper behind the first two.
"""
