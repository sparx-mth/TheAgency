#!/usr/bin/env python3
"""
InternNav Bridge Utilities

Helper functions and classes for the bridge node.
"""

import numpy as np
import cv2
import base64
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class ActionType(Enum):
    """Navigation action types."""
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class NavigationAction:
    """Represents a navigation action with parameters."""
    action_type: ActionType
    distance: float = 0.0  # meters for forward
    angle: float = 0.0  # degrees for turn
    duration: float = 0.0  # seconds
    velocity: Tuple[float, float] = (0.0, 0.0)  # (linear, angular)
    
    @classmethod
    def from_discrete(cls, action: ActionType, config: Dict) -> "NavigationAction":
        """Create NavigationAction from discrete action and config."""
        action_params = config.get('action_to_velocity', {}).get(action.value, {})
        
        distance = config.get('forward_distance', 0.25)
        angle = config.get('turn_angle', 30)
        
        linear = action_params.get('linear_x', 0.0)
        angular = action_params.get('angular_z', 0.0)
        
        if action == ActionType.MOVE_FORWARD:
            duration = distance / max(abs(linear), 0.01)
        elif action in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT]:
            duration = math.radians(angle) / max(abs(angular), 0.01)
        else:
            duration = 0.0
            
        return cls(
            action_type=action,
            distance=distance if action == ActionType.MOVE_FORWARD else 0.0,
            angle=angle if action in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT] else 0.0,
            duration=duration,
            velocity=(linear, angular)
        )


class ActionParser:
    """Parse action strings from model output."""
    
    # Common action patterns and their mappings
    ACTION_PATTERNS = {
        # Forward actions
        ('forward', 'ahead', 'straight', 'move_forward', 'go_forward'): ActionType.MOVE_FORWARD,
        # Left turns
        ('left', 'turn_left', 'rotate_left'): ActionType.TURN_LEFT,
        # Right turns
        ('right', 'turn_right', 'rotate_right'): ActionType.TURN_RIGHT,
        # Stop actions
        ('stop', 'done', 'finish', 'complete', 'arrived', 'halt'): ActionType.STOP,
        # Look actions
        ('look_up', 'up'): ActionType.LOOK_UP,
        ('look_down', 'down'): ActionType.LOOK_DOWN,
    }
    
    @classmethod
    def parse(cls, action_str: str) -> ActionType:
        """
        Parse an action string to ActionType.
        
        Args:
            action_str: Raw action string from model
            
        Returns:
            Parsed ActionType
        """
        action_lower = action_str.lower().strip()
        
        # Direct enum match
        try:
            return ActionType[action_str.upper().replace(' ', '_')]
        except KeyError:
            pass
            
        # Pattern matching
        for patterns, action_type in cls.ACTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in action_lower:
                    return action_type
                    
        return ActionType.UNKNOWN
        
    @classmethod
    def extract_from_text(cls, text: str) -> List[ActionType]:
        """
        Extract all actions mentioned in a text.
        
        Args:
            text: Text that may contain action mentions
            
        Returns:
            List of ActionTypes found
        """
        actions = []
        text_lower = text.lower()
        
        for patterns, action_type in cls.ACTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    actions.append(action_type)
                    break  # Only add each action type once
                    
        return actions


class ImageUtils:
    """Image processing utilities."""
    
    @staticmethod
    def resize_with_aspect_ratio(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        pad_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Resize image maintaining aspect ratio with padding.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            interpolation: OpenCV interpolation method
            pad_color: Color for padding
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Create padded image
        result = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        
        # Calculate padding
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
        
    @staticmethod
    def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Center crop an image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        start_x = (w - target_w) // 2
        start_y = (h - target_h) // 2
        
        return image[start_y:start_y + target_h, start_x:start_x + target_w]
        
    @staticmethod
    def encode_base64(
        image: np.ndarray,
        encoding: str = 'jpeg',
        quality: int = 95
    ) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Input image (BGR or RGB)
            encoding: 'jpeg' or 'png'
            quality: JPEG quality (1-100)
            
        Returns:
            Base64 encoded string
        """
        if encoding == 'jpeg':
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            _, buffer = cv2.imencode('.png', image)
            
        return base64.b64encode(buffer).decode('utf-8')
        
    @staticmethod
    def decode_base64(base64_str: str) -> np.ndarray:
        """
        Decode base64 string to image.
        
        Args:
            base64_str: Base64 encoded image
            
        Returns:
            Decoded image as numpy array
        """
        image_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


class CoordinateTransforms:
    """Coordinate transformation utilities."""
    
    @staticmethod
    def quaternion_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion (x, y, z, w)
            
        Returns:
            Euler angles (roll, pitch, yaw) in radians
        """
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """
        Convert Euler angles to quaternion.
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            Quaternion (x, y, z, w)
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return x, y, z, w
        
    @staticmethod
    def relative_pose(
        current_pos: Tuple[float, float, float],
        current_yaw: float,
        goal_pos: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        """
        Calculate relative position of goal in robot's frame.
        
        Args:
            current_pos: Current position (x, y, z)
            current_yaw: Current yaw angle in radians
            goal_pos: Goal position (x, y, z)
            
        Returns:
            (distance, angle) to goal
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        
        distance = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.atan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw
        
        # Normalize angle to [-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
            
        return distance, relative_angle


class ConfigValidator:
    """Validate configuration files."""
    
    REQUIRED_SECTIONS = ['bridge', 'inputs', 'outputs', 'model']
    
    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required sections
        for section in cls.REQUIRED_SECTIONS:
            if section not in config:
                errors.append(f"Missing required section: {section}")
                
        if errors:
            return False, errors
            
        # Validate server config
        server = config.get('bridge', {}).get('server', {})
        if 'host' not in server:
            errors.append("Missing server.host")
        if 'port' not in server:
            errors.append("Missing server.port")
            
        # Validate inputs
        inputs = config.get('inputs', {})
        if inputs.get('rgb', {}).get('enabled', True):
            if 'topic' not in inputs.get('rgb', {}):
                errors.append("Missing inputs.rgb.topic")
                
        # Validate outputs
        outputs = config.get('outputs', {})
        if outputs.get('discrete', {}).get('enabled', True):
            if 'topic' not in outputs.get('discrete', {}):
                errors.append("Missing outputs.discrete.topic")
                
        return len(errors) == 0, errors
        
    @classmethod
    def merge_configs(cls, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls.merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result


class MetricsCollector:
    """Collect and report metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = []
        self.action_counts = {action: 0 for action in ActionType}
        self.total_inferences = 0
        self.errors = 0
        
    def record_inference(self, inference_time_ms: float, action: ActionType, error: bool = False):
        """Record an inference result."""
        self.inference_times.append(inference_time_ms)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
            
        self.action_counts[action] += 1
        self.total_inferences += 1
        
        if error:
            self.errors += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        if not self.inference_times:
            avg_time = 0.0
            min_time = 0.0
            max_time = 0.0
        else:
            avg_time = sum(self.inference_times) / len(self.inference_times)
            min_time = min(self.inference_times)
            max_time = max(self.inference_times)
            
        return {
            'total_inferences': self.total_inferences,
            'errors': self.errors,
            'error_rate': self.errors / max(self.total_inferences, 1),
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'action_distribution': {
                action.value: count
                for action, count in self.action_counts.items()
            }
        }
        
    def reset(self):
        """Reset all metrics."""
        self.inference_times = []
        self.action_counts = {action: 0 for action in ActionType}
        self.total_inferences = 0
        self.errors = 0
