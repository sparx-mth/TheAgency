import os, json, time, glob, datetime, re
from typing import Optional, Tuple

import cv2
import numpy as np

_POSE_NAME_RES = [
    re.compile(
        r".*?/x(?P<x>-?\d{1,6})y(?P<y>-?\d{1,6})z(?P<z>-?\d{1,6})yaw(?P<yaw>-?\d{1,9})(?:__[^/]+)?\.[A-Za-z0-9]+$"
    ),
    re.compile(
        r".*?_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)_z(?P<z>-?\d+(?:\.\d+)?)_yaw(?P<yaw>-?\d+(?:\.\d+)?)(?:\.[A-Za-z0-9]+)$"
    ),
]


def update_sidecar_json(json_path: str, pose: dict, image_basename: str, vlm_text: Optional[str]):
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

def strip_leading_slash(s: str) -> str:
    """Normalizes ROS frame names."""
    if not s:
        return s
    return s[1:] if s.startswith("/") else s

def stamp_to_sec(stamp) -> float:
    """Converts ROS builtin_interfaces/Time to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

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