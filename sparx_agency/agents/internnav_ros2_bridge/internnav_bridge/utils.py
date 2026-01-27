#!/usr/bin/env python3
"""Utility functions for InternNav Bridge."""

import math
import base64
from typing import Tuple

import cv2
import numpy as np


def quaternion_to_yaw(q: Tuple[float, float, float, float]) -> float:
    """Convert quaternion (x, y, z, w) to yaw angle in radians."""
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Convert yaw angle to quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def relative_goal(
        pos: Tuple[float, float],
        yaw: float,
        goal: Tuple[float, float]
) -> Tuple[float, float]:
    """Calculate distance and angle to goal from current pose."""
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    dist = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx) - yaw
    # Normalize to [-pi, pi]
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return dist, angle


def resize_image(
        image: np.ndarray,
        size: Tuple[int, int],
        keep_aspect: bool = False
) -> np.ndarray:
    """Resize image to target size (width, height)."""
    if not keep_aspect:
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    h, w = image.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    result = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result


def encode_image_base64(image: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def decode_image_base64(b64_str: str) -> np.ndarray:
    """Decode base64 string to BGR image."""
    data = base64.b64decode(b64_str)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)