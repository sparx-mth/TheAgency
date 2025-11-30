std_msgs/Header header

bool valid_position

# origin in global coordinates
ros_common_interfaces/LLA origin

# UAV position in global coordinates
ros_common_interfaces/LLA lla

# UAV position in relative to origin [m] - ENU
geometry_msgs/Vector3 position

# UAV velocity in m/s  - ENU
geometry_msgs/Vector3 velocity

# Altitude above FCU home
float32 altitude_relative

# Altitude above ground
#TODO: float32 altitude_agl

# UAV azimuth in radians, 0=North, Positive CW (East=PI/2)
float32 azimuth

# Body angles
float32 roll
float32 pitch

string fcu_mode

bool armed

bool connected

bool airborne

float32 temperature

bool armable
