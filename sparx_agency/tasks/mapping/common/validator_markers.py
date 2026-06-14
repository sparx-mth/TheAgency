import numpy as np

MARKER_OBJECTS = [

    # ===== ORIGINAL WALL =====
    {
        "id": "da3_marker_red",
        "color": "red",
        "center_world": np.array([6.35, 2.22, 1.50], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 0, "h_high": 10, "s_min": 110, "v_min": 70},
    },
    {
        "id": "da3_marker_purple",
        "color": "purple",
        "center_world": np.array([6.35, 2.10, 1.00], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 135, "h_high": 165, "s_min": 70, "v_min": 50},
    },
    {
        "id": "da3_marker_green",
        "color": "green",
        "center_world": np.array([6.35, 2.00, 1.28], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 45, "h_high": 85, "s_min": 80, "v_min": 60},
    },
    {
        "id": "da3_marker_cyan",
        "color": "cyan",
        "center_world": np.array([6.35, 3.00, 1.80], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 80, "h_high": 105, "s_min": 70, "v_min": 70},
    },
    {
        "id": "da3_marker_yellow",
        "color": "yellow",
        "center_world": np.array([6.35, 1.70, 1.30], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 18, "h_high": 40, "s_min": 80, "v_min": 80},
    },
    {
        "id": "da3_marker_orange",
        "color": "orange",
        "center_world": np.array([6.35, 2.54, 1.08], dtype=np.float32),
        "radius_m": 0.07,
        "hsv": {"h_low": 8, "h_high": 22, "s_min": 100, "v_min": 80},
    },

    # ===== ELEVATOR WALL =====
    {
        "id": "da3_elev_marker_red",
        "color": "red",
        "center_world": np.array([-2.70, 19.35, 1.50], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 0, "h_high": 10, "s_min": 110, "v_min": 70},
    },
    {
        "id": "da3_elev_marker_purple",
        "color": "purple",
        "center_world": np.array([-2.55, 19.35, 0.75], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 135, "h_high": 165, "s_min": 70, "v_min": 50},
    },
    {
        "id": "da3_elev_marker_green",
        "color": "green",
        "center_world": np.array([2.50, 19.35, 1.28], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 45, "h_high": 85, "s_min": 80, "v_min": 60},
    },
    {
        "id": "da3_elev_marker_cyan",
        "color": "cyan",
        "center_world": np.array([2.65, 19.35, 1.80], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 80, "h_high": 105, "s_min": 70, "v_min": 70},
    },
    {
        "id": "da3_elev_marker_yellow",
        "color": "yellow",
        "center_world": np.array([0.05, 19.35, 2.70], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 18, "h_high": 40, "s_min": 80, "v_min": 80},
    },
    {
        "id": "da3_elev_marker_orange",
        "color": "orange",
        "center_world": np.array([0.35, 19.35, 0.90], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 8, "h_high": 22, "s_min": 100, "v_min": 80},
    },

    # ===== PORTRAIT WALL =====
    {
        "id": "da3_portrait_marker_red",
        "color": "red",
        "center_world": np.array([-4.56141, -34.8, 1.38136], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 0, "h_high": 10, "s_min": 110, "v_min": 70},
    },
    {
        "id": "da3_portrait_marker_purple",
        "color": "purple",
        "center_world": np.array([-5.56141, -34.8, 1.10], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 135, "h_high": 165, "s_min": 70, "v_min": 50},
    },
    {
        "id": "da3_portrait_marker_green",
        "color": "green",
        "center_world": np.array([-5.16141, -34.8, 2.10], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 45, "h_high": 85, "s_min": 80, "v_min": 60},
    },
    {
        "id": "da3_portrait_marker_cyan",
        "color": "cyan",
        "center_world": np.array([-5.06141, -34.8, 1.55], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 80, "h_high": 105, "s_min": 70, "v_min": 70},
    },
    {
        "id": "da3_portrait_marker_yellow",
        "color": "yellow",
        "center_world": np.array([-5.76141, -34.8, 1.75], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 18, "h_high": 40, "s_min": 80, "v_min": 80},
    },
    {
        "id": "da3_portrait_marker_orange",
        "color": "orange",
        "center_world": np.array([-6.26141, -34.8, 1.30], dtype=np.float32),
        "radius_m": 0.12,
        "hsv": {"h_low": 8, "h_high": 22, "s_min": 100, "v_min": 80},
    },
]