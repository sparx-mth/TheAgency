import numpy as np
import cv2
import yaml
from sensor_msgs.msg import CameraInfo

def depth_to_vis_u8(depth_m: np.ndarray, clip_min=0.3, clip_max=50.0) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, clip_min, clip_max)
    # normalize to 0..255
    u8 = ((d - clip_min) / max(1e-6, (clip_max - clip_min)) * 255.0).astype(np.uint8)
    return u8

def make_depth_grid_vis(depth_m: np.ndarray, grid_w: int, grid_h: int,
                        clip_min=0.3, clip_max=50.0) -> np.ndarray:
    """
    heatmap of depth values based on grid_w, grid_h
    """
    H, W = depth_m.shape[:2]
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = 0.0

    # downsample with INTER_AREA ~= average
    small = cv2.resize(d, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # clip+normalize
    small = np.clip(small, clip_min, clip_max)
    small_u8 = ((small - clip_min) / max(1e-6, (clip_max - clip_min)) * 255.0).astype(np.uint8)

    big = cv2.resize(small_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    # colormap
    colored = cv2.applyColorMap(big, cv2.COLORMAP_TURBO)

    for x in np.linspace(0, W, grid_w + 1).astype(int):
        cv2.line(colored, (x, 0), (x, H - 1), (50, 50, 50), 1)
    for y in np.linspace(0, H, grid_h + 1).astype(int):
        cv2.line(colored, (0, y), (W - 1, y), (50, 50, 50), 1)

    return colored

def load_camera_info_from_yaml(yaml_path: str, frame_id: str) -> CameraInfo:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    msg = CameraInfo()
    msg.header.frame_id = frame_id

    msg.width = int(data["image_width"])
    msg.height = int(data["image_height"])

    msg.distortion_model = data.get("distortion_model", "plumb_bob")
    msg.d = list(data["distortion_coefficients"]["data"])
    msg.k = list(data["camera_matrix"]["data"])
    msg.r = list(data["rectification_matrix"]["data"])
    msg.p = list(data["projection_matrix"]["data"])

    return msg

def copy_camera_info(msg: CameraInfo) -> CameraInfo:
    out = CameraInfo()
    out.header = msg.header
    out.width = msg.width
    out.height = msg.height
    out.distortion_model = msg.distortion_model
    out.d = list(msg.d)
    out.k = list(msg.k)
    out.r = list(msg.r)
    out.p = list(msg.p)
    return out


def padded_camera_info(
    base: CameraInfo,
    pad_left: int,
    pad_top: int,
    new_width: int,
    new_height: int,
) -> CameraInfo:
    out = copy_camera_info(base)

    out.width = int(new_width)
    out.height = int(new_height)

    # K:
    # [fx  0 cx]
    # [ 0 fy cy]
    # [ 0  0  1]
    out.k[2] = float(base.k[2]) + float(pad_left)
    out.k[5] = float(base.k[5]) + float(pad_top)

    # P:
    # [fx  0 cx Tx]
    # [ 0 fy cy Ty]
    # [ 0  0  1  0]
    out.p[2] = float(base.p[2]) + float(pad_left)
    out.p[6] = float(base.p[6]) + float(pad_top)

    return out

def crop_resize_camera_info(
    base: CameraInfo,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    new_width: int,
    new_height: int,
) -> CameraInfo:
    out = copy_camera_info(base)

    sx = float(new_width) / float(crop_width)
    sy = float(new_height) / float(crop_height)

    out.width = int(new_width)
    out.height = int(new_height)

    # K
    out.k[0] = float(base.k[0]) * sx
    out.k[2] = (float(base.k[2]) - float(crop_left)) * sx
    out.k[4] = float(base.k[4]) * sy
    out.k[5] = (float(base.k[5]) - float(crop_top)) * sy

    # P
    out.p[0] = float(base.p[0]) * sx
    out.p[2] = (float(base.p[2]) - float(crop_left)) * sx
    out.p[3] = float(base.p[3]) * sx

    out.p[5] = float(base.p[5]) * sy
    out.p[6] = (float(base.p[6]) - float(crop_top)) * sy
    out.p[7] = float(base.p[7]) * sy

    return out


def resized_camera_info(
    base: CameraInfo,
    new_width: int,
    new_height: int,
) -> CameraInfo:
    out = copy_camera_info(base)

    old_width = float(base.width)
    old_height = float(base.height)

    sx = float(new_width) / old_width
    sy = float(new_height) / old_height

    out.width = int(new_width)
    out.height = int(new_height)

    # Scale K
    out.k[0] = float(base.k[0]) * sx  # fx
    out.k[2] = float(base.k[2]) * sx  # cx
    out.k[4] = float(base.k[4]) * sy  # fy
    out.k[5] = float(base.k[5]) * sy  # cy

    # Scale P
    out.p[0] = float(base.p[0]) * sx  # fx
    out.p[2] = float(base.p[2]) * sx  # cx
    out.p[3] = float(base.p[3]) * sx  # Tx, usually 0

    out.p[5] = float(base.p[5]) * sy  # fy
    out.p[6] = float(base.p[6]) * sy  # cy
    out.p[7] = float(base.p[7]) * sy  # Ty, usually 0

    return out
