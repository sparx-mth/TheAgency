#!/usr/bin/env python3
import math
import sys
import rclpy
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

# Where the drone appears when nobody says otherwise. Unchanged, so every
# existing launch path spawns exactly where it always did.
DEFAULT_SPAWN = (1.0, 1.0, 2.0, 0.0)


def parse_spawn(text):
    """Parse an optional "x,y,z" or "x,y,z,yaw" argument into a 4-tuple.

    Returns None on anything unparseable rather than raising: this runs inside
    a launch file, and a typo in a spawn pose should put the drone in the usual
    place with a warning, not fail the whole bring-up with a traceback into a
    log nobody reads.
    """
    if not text:
        return DEFAULT_SPAWN
    parts = [p for p in str(text).replace(' ', '').split(',') if p != '']
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 3:
        return (values[0], values[1], values[2], 0.0)
    if len(values) == 4:
        return (values[0], values[1], values[2], values[3])
    return None


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('spawn_drone')
    cli = node.create_client(SpawnEntity, '/spawn_entity')

    content = sys.argv[1]
    namespace = sys.argv[2]
    # OPTIONAL third argument: "x,y,z" or "x,y,z,yaw" in the world frame.
    # Added so a search campaign can start the aircraft in different parts of
    # the building; without it every trial begins in the same corner and the
    # thing being measured is mostly the route out of that corner.
    raw = sys.argv[3] if len(sys.argv) > 3 else ''
    spawn = parse_spawn(raw)
    if spawn is None:
        node.get_logger().warn(
            'unparseable spawn %r; expected x,y,z or x,y,z,yaw. '
            'Spawning at the default %r' % (raw, DEFAULT_SPAWN))
        spawn = DEFAULT_SPAWN
    x, y, z, yaw = spawn

    initial_pose = Pose()
    initial_pose.position.x = x
    initial_pose.position.y = y
    initial_pose.position.z = z

    # Yaw only: the drone spawns level, and a roll or pitch here would start
    # the run already inside the follower's capsize guard.
    initial_pose.orientation.x = 0.0
    initial_pose.orientation.y = 0.0
    initial_pose.orientation.z = math.sin(yaw / 2.0)
    initial_pose.orientation.w = math.cos(yaw / 2.0)

    req = SpawnEntity.Request()
    req.name = namespace
    req.xml = content
    req.robot_namespace = namespace
    req.reference_frame = "world"
    req.initial_pose = initial_pose

    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None:
        node.get_logger().info(
            f'Spawned at ({initial_pose.position.x}, {initial_pose.position.y}, {initial_pose.position.z}) - ' +
            f'Result: {future.result().success} - {future.result().status_message}')
    else:
        node.get_logger().info('Service call failed %r' % (future.exception(),))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()