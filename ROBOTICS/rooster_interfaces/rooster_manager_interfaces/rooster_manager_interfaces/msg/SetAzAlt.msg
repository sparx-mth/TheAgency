# Set rooster azimuth and altitude
# The message published by GUI periodically
# when  objects tracking and/or to close altitude functions activated.

std_msgs/Header header

bool azimuth_valid # if true set azimuth
float32 azimuth # radians

bool altitude_valid # if true set altitude
float32 altitude # AGL in meters
