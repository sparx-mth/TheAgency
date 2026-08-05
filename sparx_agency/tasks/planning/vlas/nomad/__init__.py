"""NoMaD task package: ROS2 bridge + the topic-name override for upstream.

The NoMaD model is NOT vendored here -- upstream `visualnav-transformer` runs as
a sibling process and publishes waypoints on a ROS2 topic. The bridge turns
those waypoints into platform commands.
"""
