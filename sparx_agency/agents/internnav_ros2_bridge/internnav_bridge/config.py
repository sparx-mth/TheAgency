#!/usr/bin/env python3
"""Configuration management for InternNav Bridge."""

import yaml
from pathlib import Path
from typing import Dict, Optional


def load_config(config_path: Optional[str] = None, logger=None) -> Dict:
    """Load configuration from YAML file, merging with defaults."""
    config = get_default_config()

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        config = _deep_merge(config, user_config)
        if logger:
            logger.info(f"Loaded config from: {config_path}")
    elif logger:
        logger.warn("Using default configuration")

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config() -> Dict:
    return {
        'bridge': {
            'server': {'host': 'localhost', 'port': 8000, 'protocol': 'http', 'timeout_sec': 30.0},
            'control': {'inference_rate': 4.0, 'continuous_inference': False, 'min_inference_interval': 0.1, 'handheld': False}
        },
        'inputs': {
            'rgb': {'enabled': True, 'topic': '/camera/rgb/image_raw', 'msg_type': 'sensor_msgs/Image'},
            'depth': {'enabled': False, 'topic': '/camera/depth/image_raw'},
            'instruction': {'enabled': True, 'topic': '/navigation/instruction', 'default': 'Navigate to the goal'},
            'odometry': {'enabled': False, 'topic': '/odom'},
        },
        'outputs': {
            'discrete': {'enabled': True, 'topic': '/navigation/action',
                        'action_mapping': {'MOVE_FORWARD': 'forward', 'TURN_LEFT': 'left',
                                          'TURN_RIGHT': 'right', 'STOP': 'stop'}},
            'continuous': {'enabled': False, 'topic': '/cmd_vel'},
            'feedback': {'enabled': True, 'topic': '/navigation/feedback'},
            'status': {'enabled': True, 'topic': '/navigation/status'}
        },
        'model': {'variant': 'InternVLA-N1', 'ckpt_path': '',
                 'input_format': {'target_width': 640, 'target_height': 480}},
    }