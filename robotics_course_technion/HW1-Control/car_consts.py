#!/usr/bin/env python3

import numpy as np
'''
update the values manually in:
bicycle_steering_controller.yaml
xacro files
low level (arduino interface)
'''
max_steering_angle_deg = 27 # [deg]
max_steering_angle_rad = np.deg2rad(max_steering_angle_deg)
max_dt_steering_angle = np.deg2rad(150.0)
wheelbase = 0.335 #[meter]
car_width = 0.296
min_linear_velocity = 0.7 # m/s
max_linear_velocity = 1.5 # m/s
wheel_radius = 0.056 # m
