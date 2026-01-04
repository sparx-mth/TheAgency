std_msgs/Header header

uint8 fix_type
uint8 FIX_TYPE_NO_FIX=0
uint8 FIX_TYPE_DEAD_RECKONING=1
uint8 FIX_TYPE_2D=2
uint8 FIX_TYPE_3D=3
uint8 FIX_TYPE_GNSS_PLUS_DEAD_RECKONING=4
uint8 FIX_TYPE_TIME_ONLY=6

# Position in global coordinates
ros_common_interfaces/LLA lla

float32 horizontal_accuracy # GPS HDOP horizontal dilution of position in [m]. If unknown, set to: 65535
float32 vertical_accuracy # GPS VDOP vertical dilution of position in [m]. If unknown, set to: 65535

float32 ground_speed # GPS ground speed. If unknown, set to: 65535

# Velocity in m/s  - ENU
geometry_msgs/Vector3 velocity

float32 course_over_ground # Course over ground (NOT heading, but direction of movement), 0.0..359.99 degrees. If unknown, set to: 65535

uint8 num_satellites # Number of satellites visible. If unknown, set to 255

uint8 id # GPS ID (zero indexed). Used for multiple GPS inputs

float32 azimuth # Yaw of vehicle relative to Earth's North, zero means not available, use 360 for north

float32 time_since_startup

float32 time_to_first_fix

uint8 spoofing_state
uint8 SPOOFING_STATE_UNKNOWN_OR_DEACTIVATED=0
uint8 SPOOFING_STATE_NO_SPOOFING=1
uint8 SPOOFING_STATE_SPOOFING_INDICATED=2
uint8 SPOOFING_STATE_MULTIPLE_SPOOFING_INDICATED=3

uint8[] jamming_state
uint8 JAMMING_STATE_UNKNOWN=0
uint8 JAMMING_STATE_NO_JAMMING=1
uint8 JAMMING_STATE_JAMMING_WARNING=2
uint8 JAMMING_STATE_JAMMING_CRITICAL=3

# CW jamming indicator, scaled (0 = no CW jamming, 255 = strong CW jamming)
uint8[] jamming_indicator

uint16[] noise_per_measurement

float32 UNKNOWN=65535