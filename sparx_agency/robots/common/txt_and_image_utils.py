import os, json, time, cv2, glob, datetime, re
from typing import Optional, Tuple

_POSE_NAME_RES = [
    re.compile(
        r".*?/x(?P<x>-?\d{1,6})y(?P<y>-?\d{1,6})z(?P<z>-?\d{1,6})yaw(?P<yaw>-?\d{1,9})(?:__[^/]+)?\.[A-Za-z0-9]+$"
    ),
    re.compile(
        r".*?_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)_z(?P<z>-?\d+(?:\.\d+)?)_yaw(?P<yaw>-?\d+(?:\.\d+)?)(?:\.[A-Za-z0-9]+)$"
    ),
]
def _update_sidecar_json(json_path: str, pose: dict, image_basename: str, vlm_text: Optional[str]):
    obj = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                obj = json.load(f)
        except Exception:
            obj = {}
    obj.setdefault("pose", pose)
    obj.setdefault("image", image_basename)
    if vlm_text:
        obj["vlm_caption"] = vlm_text
        entries = obj.setdefault("entries", [])
        entries.append({
            "timestamp": int(time.time()),
            "prompt": "Describe the image",
            "response": vlm_text
        })
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, json_path)

def list_frames(frames_dir: str):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    return sorted(files)

def get_pose_for_frame(path: str, *, angles_map: dict, from_name: bool):
    if from_name:
        p = parse_pose_from_name(path)
        if p:
            return p
    p = angles_map.get(os.path.basename(path))
    if isinstance(p, dict): return p
    return {"x":0.0,"y":0.0,"z":1.5,"yaw":0.0}

def pose_to_name(pose: dict) -> str:
    x = _fmt_signed(pose["x"],   scale=1000,      width=4, eps=5e-4)     # mm
    y = _fmt_signed(pose["y"],   scale=1000,      width=4, eps=5e-4)     # mm
    z = _fmt_signed(pose["z"],   scale=1000,      width=4, eps=5e-4)     # mm
    yaw = _fmt_signed(pose["yaw"], scale=1_000_000, width=7, eps=5e-7)   # microrad
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

    return f"x{x}y{y}z{z}yaw{yaw}__{timestamp}"

def center_crop_frac(img, frac: float):
    """
    Center-crop by a fraction of the original size.
    frac=1.0 → no crop, 0.5 → crop to middle 50% (both width & height).
    """
    frac = max(0.05, min(1.0, float(frac)))  # clamp
    H, W = img.shape[:2]
    cw, ch = int(W * frac), int(H * frac)
    x0 = (W - cw) // 2
    y0 = (H - ch) // 2
    return img[y0:y0+ch, x0:x0+cw], (x0, y0, x0+cw, y0+ch)

def _apply_crop_and_flip(img, crop_frac: float, flip180: bool):
    """Center-crop then optionally rotate 180°."""
    work = img
    crop_box = None
    if crop_frac < 1.0:
        # assumes you already have center_crop_frac(img, frac) -> (cropped, (x1,y1,x2,y2))
        work, crop_box = center_crop_frac(img, crop_frac)
    if flip180:
        work = cv2.rotate(work, cv2.ROTATE_180)
    return work, crop_box

import cv2
import numpy as np

def correct_histogram(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Apply histogram correction to enhance image contrast.

    Args:
        image (np.ndarray): Input image, can be grayscale or color (BGR).
        method (str): 'clahe' for adaptive (default), 'global' for global histogram equalization.

    Returns:
        np.ndarray: Contrast-enhanced image.
    """
    if len(image.shape) == 2:  # Grayscale
        if method == "global":
            return cv2.equalizeHist(image)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
    elif len(image.shape) == 3:  # Color (BGR)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        if method == "global":
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        raise ValueError("Unsupported image format")

def _save_jpg(path: str, bgr):
    if not cv2.imwrite(path, bgr):
        raise RuntimeError(f"failed to write {path}")


def _fmt_signed(value: float, scale: int, width: int, eps: float) -> str:
    if abs(value) < eps:
        value = 0.0
    n = int(round(value * scale))
    if n < 0:
        return f"-{abs(n):0{width}d}"
    else:
        return f"{n:0{width}d}"

def parse_pose_from_name(fname: str):
    base = os.path.basename(fname)
    for rx in _POSE_NAME_RES:
        m = rx.match(fname) or rx.match(base)
        if not m:
            continue
        gd = m.groupdict()
        # If the captures are integers (mm / microrad), convert to meters / radians
        try:
            # compact-int format → ints
            x_mm   = int(gd["x"])
            y_mm   = int(gd["y"])
            z_mm   = int(gd["z"])
            yaw_ur = int(gd["yaw"])   # microradians
            return {
                "x": x_mm / 1000.0,
                "y": y_mm / 1000.0,
                "z": z_mm / 1000.0,
                "yaw": yaw_ur / 1_000_000.0,
            }
        except ValueError:
            # legacy underscore/float format → floats already in meters/radians
            return {
                "x": float(gd["x"]),
                "y": float(gd["y"]),
                "z": float(gd["z"]),
                "yaw": float(gd["yaw"]),
            }
    return None

def load_angles_map(path: str):
    try:
        with open(path, "r") as f: j = json.load(f)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}

def _unique_name(base_path: str, enable_suffix: bool, counter: int) -> Tuple[str, str]:
    """
    Return (jpg_path, json_path). If enable_suffix, append _0001, _0002...
    """
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
    base_path = f"{base_path}___{timestamp}"
    jpg_path = base_path + ".jpg"
    json_path = base_path + ".json"
    if not enable_suffix:
        return jpg_path, json_path
    n = counter
    while True:
        suffix = f"_{n:04d}"
        jp = base_path + suffix + ".jpg"
        jj = base_path + suffix + ".json"
        if not (os.path.exists(jp) or os.path.exists(jj)):
            return jp, jj
        n += 1
