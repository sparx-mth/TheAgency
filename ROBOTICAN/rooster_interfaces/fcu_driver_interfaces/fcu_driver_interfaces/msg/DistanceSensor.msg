# https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR

std_msgs/Header header

float32 min_distance
float32 max_distance
float32 current_distance

# uint8 type # MAV_DISTANCE_SENSOR

# uint8 id

uint8 orientation # MAV_SENSOR_ORIENTATION

uint8 covariance

float32 horizontal_fov

float32 vertical_fov

geometry_msgs/Quaternion quaternion

# uint8 signal_quality

