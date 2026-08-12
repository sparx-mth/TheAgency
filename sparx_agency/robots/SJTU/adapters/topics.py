"""Every name the SJTU sim drone answers to, in one place.

Constants only -- no imports, no logic, nothing that can fail at import time.
The point is that a node never spells a topic itself. The plugin builds most of
these by concatenating its own namespace with a hardcoded leaf, so a rename in
``sjtu_drone_bringup/config/drone.yaml`` moves *all* of them at once; a stack
that spelled them by hand would drift one file at a time and fail as "no data"
rather than as an error.

``tests/test_topics.py`` pins the composition against the literals the plugin
publishes, so this file cannot quietly disagree with the simulator it names.

**The frames are not namespaced the way the topics are.** Three different
conventions coexist on this robot, and mixing them is the usual first bug:

* the URDF links and the odom frame carry the namespace *without* a leading
  slash once normalised -- ``simple_drone/base_link``;
* the plugin builds ``odom.header.frame_id`` as ``get_namespace() + "/odom"``,
  and ``get_namespace()`` already starts with a slash, so what arrives on the
  wire is ``/simple_drone/odom``. Strip the leading slash before using it as a
  TF frame;
* both front cameras set ``<frame_name>camera</frame_name>``, which is *not*
  namespaced at all. Image headers say ``camera``, full stop, and no such TF
  frame is published. A consumer must supply the camera extrinsics itself --
  see :data:`FRONT_CAMERA_OFFSET_FLU`.
"""

NAMESPACE = "/simple_drone"
"""ROS namespace every topic sits under, from ``drone.yaml``'s ``namespace``."""

FRAME_PREFIX = "simple_drone"
"""The same name as a TF prefix: the namespace with its leading slash removed."""

# ---------------------------------------------------------------------------
# Actuation. This is the entire control surface: a body twist plus five latches.
# There is no attitude, rate, thrust or motor input while flying -- the plugin
# owns all four inner loops and exposes only the outermost one.
# ---------------------------------------------------------------------------

CMD_VEL = NAMESPACE + "/cmd_vel"
"""``geometry_msgs/Twist``. Body-frame ``linear.x/y/z`` + ``angular.z``.

Interpreted in the *yaw-aligned* body frame (FLU), not the fully-rotated one:
the plugin projects the measured velocity through a yaw-only quaternion before
comparing. While the aircraft is level -- the only attitude it commands -- that
is the same thing. ``angular.z`` is a yaw *rate*, in rad/s, positive
counter-clockwise.

Under ``POSCTRL`` this same message means something else entirely; see
:data:`POSCTRL`.
"""

TAKEOFF = NAMESPACE + "/takeoff"
"""``std_msgs/Empty``. Ignored unless the aircraft is LANDED."""

LAND = NAMESPACE + "/land"
"""``std_msgs/Empty``. Ignored unless the aircraft is FLYING."""

RESET = NAMESPACE + "/reset"
"""``std_msgs/Empty``. Teleports the model home and clears the PID integrators."""

POSCTRL = NAMESPACE + "/posctrl"
"""``std_msgs/Bool``. Latches ``cmd_vel`` from a velocity into a **world position**.

True re-reads ``cmd_vel.linear.x/y/z`` as an absolute setpoint in the odom
frame, fed through the plugin's own position PIDs. It is a mode switch, not a
scaling: a stack that leaves this latched and then publishes velocities flies to
the coordinate whose numbers happen to equal its speeds.
"""

DRONEVEL_MODE = NAMESPACE + "/dronevel_mode"
"""``std_msgs/Bool``. Selects how ``cmd_vel.angular.x/y`` are read while *not* flying.

True treats them as horizontal velocity targets, False as direct roll/pitch
angles. It has no effect in the FLYING state, which is the only state this
platform is normally driven in.
"""

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

ODOM = NAMESPACE + "/odom"
"""``nav_msgs/Odometry`` at 30 Hz. **The feedback source.**

Its twist is correctly rotated into the child frame
(``pose.Rot().Inverse().RotateVector(world_velocity)``), so it is the one
velocity on this robot that means what it says. Pose is ground truth, since the
plugin reads it straight from Gazebo.
"""

GT_POSE = NAMESPACE + "/gt_pose"
"""``geometry_msgs/Pose``. Ground-truth world pose, unstamped.

Correct, but carries no header -- no timestamp and no frame -- so it cannot be
associated with an image. :data:`ODOM` carries the same pose with a header.
"""

GT_VEL = NAMESPACE + "/gt_vel"
"""``geometry_msgs/Twist``. **Mis-rotated. Do not use as feedback.**

The plugin computes it as ``pose.Rot().RotateVector(world_velocity)``, which is
the body-to-world rotation applied to a vector that is already in world. The
correct call is ``RotateVectorReverse``. The result is a velocity in no frame at
all: right only while the aircraft's heading is zero, and off by twice the yaw
angle otherwise -- so it looks perfectly plausible in a straight test flight and
silently inverts after a 180-degree turn. Use :data:`ODOM`'s twist.
"""

GT_ACC = NAMESPACE + "/gt_acc"
"""``geometry_msgs/Twist``. Ground-truth acceleration, with the same bug as
:data:`GT_VEL` -- it is rotated by ``R`` instead of ``R^T``. Do not use it."""

IMU = NAMESPACE + "/imu/out"
"""``sensor_msgs/Imu`` at 100 Hz, in ``simple_drone/base_link``.

Note the plugin *subscribes* to this as well: its attitude state is taken from
the IMU's orientation rather than from Gazebo directly.
"""

SONAR = NAMESPACE + "/sonar/out"
"""``sensor_msgs/Range`` at 30 Hz, 0.02-10 m, looking down from ``sonar_link``."""

BUMPER_STATES = NAMESPACE + "/bumper_states"
"""``gazebo_msgs/ContactsState``. The collision truth a sim episode is scored on."""

STATE = NAMESPACE + "/state"
"""``std_msgs/Int8``. LANDED / FLYING / TAKINGOFF / LANDING.

The gate on :data:`TAKEOFF` and :data:`LAND`, both of which are silently ignored
from the wrong state -- so a mission that fires takeoff once and assumes it
worked has no way to notice that it did not.
"""

CMD_MODE = NAMESPACE + "/cmd_mode"
"""``std_msgs/String``. Which interpretation of ``cmd_vel`` is currently latched."""

# ---------------------------------------------------------------------------
# Cameras. Both front sensors are mounted on the same link and see the same
# scene; they are separate Gazebo sensors rendering at different rates.
# ---------------------------------------------------------------------------

FRONT_IMAGE = NAMESPACE + "/front/image_raw"
"""``sensor_msgs/Image``, rgb8, 600x600 at 60 Hz. See ``config/camera_front_600x600.yaml``."""

FRONT_CAMERA_INFO = NAMESPACE + "/front/camera_info"
"""``sensor_msgs/CameraInfo`` for :data:`FRONT_IMAGE`. The runtime authority on intrinsics."""

FRONT_DEPTH_IMAGE = NAMESPACE + "/front_depth/depth/image_raw"
"""``sensor_msgs/Image``, **32FC1 metres**, 600x600 at 15 Hz, valid over 0.1-10 m.

Float metres, not the 16UC1 millimetres a real RGBD sensor publishes -- code
ported from an XTEND bag will be off by a factor of 1000.
"""

FRONT_DEPTH_CAMERA_INFO = NAMESPACE + "/front_depth/depth/camera_info"
"""``sensor_msgs/CameraInfo`` for :data:`FRONT_DEPTH_IMAGE`."""

FRONT_DEPTH_POINTS = NAMESPACE + "/front_depth/points"
"""``sensor_msgs/PointCloud2`` -- the depth image already back-projected by Gazebo."""

BOTTOM_IMAGE = NAMESPACE + "/bottom/image_raw"
"""``sensor_msgs/Image``, rgb8, 640x360 at 15 Hz, looking straight down."""

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

FRAME_WORLD = "world"
"""The fixed frame. A static transform publisher ties it to :data:`FRAME_ODOM`
with an identity transform, so world and odom coincide exactly."""

FRAME_ODOM = FRAME_PREFIX + "/odom"
"""Odometry parent frame. Arrives on the wire as ``/simple_drone/odom`` -- with a
leading slash -- because the plugin concatenates ``get_namespace()``. Strip it."""

FRAME_BASE_FOOTPRINT = FRAME_PREFIX + "/base_footprint"
"""Odometry child frame, and what the plugin's TF broadcast moves."""

FRAME_BASE_LINK = FRAME_PREFIX + "/base_link"
"""Body frame, FLU. The IMU and the bumper report in it."""

FRAME_SONAR = FRAME_PREFIX + "/sonar_link"
FRAME_FRONT_CAM = FRAME_PREFIX + "/front_cam_link"
FRAME_BOTTOM_CAM = FRAME_PREFIX + "/bottom_cam_link"

FRAME_CAMERA_SENSOR = "camera"
"""What both front cameras actually put in their image headers.

Unnamespaced, shared by the RGB and depth sensors, and **not** a frame anything
publishes a transform for -- ``gazebo_ros_camera``'s ``<frame_name>`` was left at
the literal ``camera``. Reading the camera pose out of TF by this name fails; use
:data:`FRAME_FRONT_CAM` plus :data:`FRONT_CAMERA_OFFSET_FLU` instead.
"""

FRONT_CAMERA_OFFSET_FLU = (0.2, 0.0, 0.0)
"""Front camera position in ``base_link``, metres, body FLU.

From the URDF's ``front_cam_joint`` (``xyz="0.2 0 0"``, zero rotation). It is
here because the sensor's own header frame is a dead end, and because 20 cm of
forward offset is not negligible for a mapper: the camera carves free space
outward from *itself*, so the body origin sits in the one place it can never
observe.
"""
