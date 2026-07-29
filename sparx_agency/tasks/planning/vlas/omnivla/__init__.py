"""OmniVLA task package: in-process torch model + ROS2 bridge.

OmniVLA is loaded IN-PROCESS (unlike NavDP/FlowNav/InternVLA-N1, which are HTTP
servers), so its model wrapper lives in ``serve/`` and needs the external
OmniVLA/prismatic checkout on PYTHONPATH.
"""
