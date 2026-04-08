import numpy as np

# Local frame convention:
#   X = forward, Y = left, Z = up
# Local origin is at object center (matches Gazebo box geometry default).

LANDMARK_OBJECTS = [
    {
        "id": "da3_red_box",
        "color": "red",
        "pose_world": {
            "position": np.array([3.0, 1.0, 0.30], dtype=np.float32),
            "rpy": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        },
        "size_xyz": np.array([0.40, 0.40, 0.60], dtype=np.float32),
        "points_local": {
            "front_top_left":  np.array([0.20,  0.20,  0.30], dtype=np.float32),
            "front_top_right": np.array([0.20, -0.20,  0.30], dtype=np.float32),
            "front_bot_left":  np.array([0.20,  0.20, -0.30], dtype=np.float32),
            "front_bot_right": np.array([0.20, -0.20, -0.30], dtype=np.float32),
            "front_center":    np.array([0.20,  0.00,  0.00], dtype=np.float32),
        },
    },
    {
        "id": "da3_green_box",
        "color": "green",
        "pose_world": {
            "position": np.array([5.0, -1.2, 0.25], dtype=np.float32),
            "rpy": np.array([0.0, 0.0, 0.4], dtype=np.float32),
        },
        "size_xyz": np.array([0.30, 0.70, 0.50], dtype=np.float32),
        "points_local": {
            "front_top_left":  np.array([0.15,  0.35,  0.25], dtype=np.float32),
            "front_top_right": np.array([0.15, -0.35,  0.25], dtype=np.float32),
            "front_bot_left":  np.array([0.15,  0.35, -0.25], dtype=np.float32),
            "front_bot_right": np.array([0.15, -0.35, -0.25], dtype=np.float32),
            "top_center":      np.array([0.00,  0.00,  0.25], dtype=np.float32),
        },
    },
    {
        "id": "da3_purple_board",
        "color": "purple",
        "pose_world": {
            "position": np.array([4.2, 2.0, 0.60], dtype=np.float32),
            "rpy": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        "size_xyz": np.array([0.80, 0.02, 1.20], dtype=np.float32),
        "points_local": {
            "top_left":     np.array([0.40,  0.01,  0.60], dtype=np.float32),
            "top_right":    np.array([0.40, -0.01,  0.60], dtype=np.float32),
            "bottom_left":  np.array([0.40,  0.01, -0.60], dtype=np.float32),
            "bottom_right": np.array([0.40, -0.01, -0.60], dtype=np.float32),
            "center":       np.array([0.40,  0.00,  0.00], dtype=np.float32),
        },
    },
]
