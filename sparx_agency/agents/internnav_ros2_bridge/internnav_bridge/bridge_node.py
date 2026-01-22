#!/usr/bin/env python3
"""
InternNav Bridge Node

A configurable ROS2 bridge that connects any simulation to the InternNav model server.
Handles topic subscription, data transformation, model inference, and action publishing.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import cv2
import base64
import pickle
import json
import yaml
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum

# ROS2 message imports
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# HTTP client
import requests
from urllib.parse import urljoin


class ActionType(Enum):
    """Supported action types from the model."""
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class BridgeState:
    """Maintains the current state of the bridge."""
    current_rgb: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    current_instruction: str = ""
    current_odometry: Optional[Dict] = None
    current_goal: Optional[Dict] = None
    rgb_timestamp: float = 0.0
    depth_timestamp: float = 0.0
    last_inference_time: float = 0.0
    is_navigating: bool = False
    image_history: deque = field(default_factory=lambda: deque(maxlen=10))


class ModelClient:
    """HTTP client for InternNav Agent Server API.
    
    Handles the stateful agent lifecycle:
    - init: Initialize agent with model config
    - step: Get action from observation (uses pickle+base64 encoding)
    - reset: Reset for new episode
    """
    
    # Standard VLN-CE action space
    ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    
    # Default model settings for InternVLA-N1
    DEFAULT_MODEL_SETTINGS = {
        "policy_name": "InternVLAN1_Policy",
        "state_encoder": None,
        "env_num": 1,
        "sim_num": 1,
        "model_path": "checkpoints/InternVLA-N1-DualVLN",
        "camera_intrinsic": [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        "width": 640,
        "height": 480,
        "hfov": 79,
        "resize_w": 384,
        "resize_h": 384,
        "max_new_tokens": 1024,
        "num_frames": 32,
        "num_history": 8,
        "num_future_steps": 4,
        "device": "cuda:0",
        "predict_step_nums": 32,
        "continuous_traj": True,
        "infer_mode": "partial_async",
        "vis_debug": False,
        "vis_debug_path": "./logs/vis_debug"
    }
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.base_url = f"{config['protocol']}://{config['host']}:{config['port']}"
        self.timeout = config.get('timeout_sec', 30.0)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.session = requests.Session()
        
        # Agent state
        self.agent_name = config.get('agent_name', 'internvla_n1')
        self.model_name = config.get('model_name', 'internvla_n1')
        self.ckpt_path = config.get('ckpt_path', '')
        self.initialized = False
        
    def check_health(self) -> bool:
        """Check if the model server is reachable."""
        try:
            # InternNav server exposes /openapi.json
            url = f"{self.base_url}/openapi.json"
            response = self.session.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warn(f"Health check failed: {e}")
            return False
    
    def init_agent(self, model_name: str = None, ckpt_path: str = None, model_settings: Dict = None) -> bool:
        """Initialize the agent on the server."""
        url = f"{self.base_url}/agent/init"
        
        # Merge default settings with provided settings
        final_settings = self.DEFAULT_MODEL_SETTINGS.copy()
        if model_settings:
            final_settings.update(model_settings)
        
        payload = {
            "agent_config": {
                "model_name": model_name or self.model_name,
                "ckpt_path": ckpt_path or self.ckpt_path,
                "model_settings": final_settings,
                "server_host": self.config['host'],
                "server_port": self.config['port']
            }
        }
        
        try:
            self.logger.info(f"Initializing agent '{self.agent_name}' with model '{payload['agent_config']['model_name']}'...")
            response = self.session.post(url, json=payload, timeout=self.timeout * 3)
            
            if response.status_code == 201:
                self.initialized = True
                # Try to extract agent name from response
                try:
                    data = response.json()
                    if "agent_name" in data:
                        self.agent_name = data["agent_name"]
                except:
                    pass
                self.logger.info(f"Agent initialized successfully: {self.agent_name}")
                return True
            else:
                self.logger.error(f"Agent init failed: HTTP {response.status_code} - {response.text[:200]}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.warn("Init timeout (model loading may take time)")
            self.initialized = True  # Might still succeed
            return True
        except Exception as e:
            self.logger.error(f"Agent init error: {e}")
            return False
    
    def reset_agent(self, reset_index: list = None) -> bool:
        """Reset the agent for a new episode."""
        url = f"{self.base_url}/agent/{self.agent_name}/reset"
        
        try:
            response = self.session.post(
                url,
                json={"reset_index": reset_index},
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.logger.info("Agent reset successful")
                return True
            else:
                self.logger.warn(f"Agent reset failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.logger.warn(f"Agent reset error: {e}")
            return False
    
    def infer(self, payload: Dict) -> Optional[Dict]:
        """Send step request to the agent (inference).
        
        The payload should contain:
        - 'rgb': numpy array (H, W, 3) uint8
        - 'depth': numpy array (H, W, 1) float32
        - 'instruction': string
        
        Observations are encoded as pickle+base64 as required by InternNav API.
        """
        if not self.initialized:
            self.logger.warn("Agent not initialized, attempting init...")
            if not self.init_agent():
                return None
        
        url = f"{self.base_url}/agent/{self.agent_name}/step"
        
        # Build observation in InternNav format (list of dicts)
        obs = [{
            'rgb': payload.get('rgb'),
            'depth': payload.get('depth'),
            'instruction': payload.get('instruction', '')
        }]
        
        # Encode as pickle + base64 (required by InternNav)
        encoded_obs = base64.b64encode(pickle.dumps(obs)).decode('utf-8')
        
        request_payload = {"observation": encoded_obs}
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    url,
                    json=request_payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json())
                else:
                    self.logger.warn(f"Step failed with status {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                self.logger.warn(f"Step timeout (attempt {attempt + 1}/{self.retry_attempts})")
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error: {e}")
                break
            except Exception as e:
                self.logger.error(f"Step error: {e}")
                break
                
        return None
    
    def _parse_response(self, data: Dict) -> Dict:
        """Parse step response and extract action.
        
        InternNav response format: {"action":[{"action":[2],"ideal_flag":true}]}
        """
        result = {"raw": data}
        
        action = "STOP"
        action_index = 0
        
        try:
            # InternNav nested format: {"action":[{"action":[2],"ideal_flag":true}]}
            if "action" in data:
                action_data = data["action"]
                
                # Handle nested list format
                if isinstance(action_data, list) and len(action_data) > 0:
                    first_action = action_data[0]
                    
                    # Format: [{"action": [2], "ideal_flag": true}]
                    if isinstance(first_action, dict) and "action" in first_action:
                        inner_action = first_action["action"]
                        if isinstance(inner_action, list) and len(inner_action) > 0:
                            action_index = int(inner_action[0])
                        elif isinstance(inner_action, (int, float)):
                            action_index = int(inner_action)
                    # Format: [2]
                    elif isinstance(first_action, (int, float)):
                        action_index = int(first_action)
                        
                # Handle simple int format
                elif isinstance(action_data, (int, float)):
                    action_index = int(action_data)
                    
            # Map action index to name
            if 0 <= action_index < len(self.ACTION_NAMES):
                action = self.ACTION_NAMES[action_index]
            elif action_index == -1:
                # Special case: -1 means continue/no action
                action = "CONTINUE"
                
        except Exception as e:
            self.logger.warn(f"Error parsing action response: {e}")
        
        result["action"] = action
        result["action_index"] = action_index
        return result


class ImageProcessor:
    """Handles image preprocessing for the model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_size = tuple(config.get('target_size', [224, 224]))
        self.interpolation_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
        }
        self.interpolation = self.interpolation_map.get(
            config.get('interpolation', 'bilinear'), cv2.INTER_LINEAR
        )
        
        normalize_cfg = config.get('normalize', {})
        self.normalize = normalize_cfg.get('enabled', False)
        self.mean = np.array(normalize_cfg.get('mean', [0.485, 0.456, 0.406]))
        self.std = np.array(normalize_cfg.get('std', [0.229, 0.224, 0.225]))
        
        self.color_convert = config.get('color_convert', 'none')
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to an image."""
        # Color conversion
        if self.color_convert == 'bgr_to_rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_convert == 'rgb_to_bgr':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize
        if image.shape[:2] != self.target_size[::-1]:  # OpenCV uses (height, width)
            image = cv2.resize(image, self.target_size, interpolation=self.interpolation)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            
        return image
    
    def to_base64(self, image: np.ndarray, encoding: str = 'jpeg', quality: int = 95) -> str:
        """Convert image to base64 string."""
        # Denormalize if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = ((image * self.std + self.mean) * 255).clip(0, 255).astype(np.uint8)
        
        if encoding == 'jpeg':
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            _, buffer = cv2.imencode('.png', image)
            
        return base64.b64encode(buffer).decode('utf-8')


class DepthProcessor:
    """Handles depth image preprocessing."""
    
    def __init__(self, config: Dict):
        self.config = config
        normalize_cfg = config.get('normalize', {})
        self.normalize = normalize_cfg.get('enabled', False)
        self.min_depth = normalize_cfg.get('min_depth', 0.0)
        self.max_depth = normalize_cfg.get('max_depth', 10.0)
        self.output_range = normalize_cfg.get('output_range', [0.0, 1.0])
        self.target_size = tuple(config.get('target_size', [224, 224]))
        
    def process(self, depth: np.ndarray) -> np.ndarray:
        """Process depth image."""
        # Resize if needed
        if depth.shape[:2] != self.target_size[::-1]:
            depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        if self.normalize:
            depth = np.clip(depth, self.min_depth, self.max_depth)
            depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
            depth = depth * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            
        return depth


class InternNavBridge(Node):
    """
    Main ROS2 bridge node that connects simulations to InternNav.
    
    This node:
    1. Subscribes to configurable input topics (RGB, depth, instruction, etc.)
    2. Preprocesses data according to configuration
    3. Sends inference requests to the InternNav model server
    4. Transforms and publishes actions to the simulation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__('internnav_bridge')
        
        # Declare parameters
        self.declare_parameter('config_path', '')
        self.declare_parameter('use_preset', '')
        
        # Load configuration
        config_path = config_path or self.get_parameter('config_path').value
        self.config = self._load_config(config_path)
        
        # Apply preset if specified
        preset = self.get_parameter('use_preset').value or self.config.get('use_preset')
        if preset and preset in self.config.get('presets', {}):
            self._apply_preset(preset)
        
        # Initialize state
        self.state = BridgeState()
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()
        
        # Initialize processors
        self._init_processors()
        
        # Initialize model client
        self.model_client = ModelClient(
            self.config['bridge']['server'],
            self.get_logger()
        )
        
        # Check model server health
        if not self.model_client.check_health():
            self.get_logger().warn("Model server is not responding. Will retry on inference.")
        else:
            self.get_logger().info("Model server is healthy.")
            # Initialize the agent
            model_config = self.config.get('model', {})
            if self.model_client.init_agent(
                model_name=model_config.get('variant', 'InternVLA-N1'),
                ckpt_path=model_config.get('ckpt_path', ''),
                model_settings=model_config.get('model_settings', {})
            ):
                self.get_logger().info("Agent initialized and ready for inference.")
            else:
                self.get_logger().warn("Agent initialization failed. Will retry on first inference.")
        
        # Setup callback groups for parallel execution
        self.input_callback_group = ReentrantCallbackGroup()
        self.inference_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Setup subscribers and publishers
        self._setup_subscribers()
        self._setup_publishers()
        
        # Setup inference timer
        control_config = self.config['bridge']['control']
        inference_rate = control_config.get('inference_rate', 4.0)
        self.inference_timer = self.create_timer(
            1.0 / inference_rate,
            self._inference_callback,
            callback_group=self.inference_callback_group
        )
        
        self.get_logger().info("InternNav Bridge initialized successfully!")
        self._log_configuration()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not config_path:
            # Try default locations
            default_paths = [
                Path(__file__).parent.parent / 'config' / 'bridge_config.yaml',
                Path.home() / '.internnav' / 'bridge_config.yaml',
                Path('/etc/internnav/bridge_config.yaml'),
            ]
            for path in default_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if not config_path or not Path(config_path).exists():
            self.get_logger().warn(f"Config not found, using defaults")
            return self._get_default_config()
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.get_logger().info(f"Loaded configuration from: {config_path}")
        return config
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'bridge': {
                'server': {
                    'host': 'localhost',
                    'port': 8000,
                    'protocol': 'http',
                    'timeout_sec': 10.0,
                    'retry_attempts': 3,
                    'inference_endpoint': '/v1/inference',
                    'health_endpoint': '/health',
                },
                'control': {
                    'inference_rate': 4.0,
                    'continuous_inference': False,
                    'min_inference_interval': 0.1,
                    'sync_inputs': True,
                    'sync_timeout': 1.0,
                }
            },
            'inputs': {
                'rgb': {
                    'enabled': True,
                    'topic': '/camera/rgb/image_raw',
                    'msg_type': 'sensor_msgs/Image',
                    'queue_size': 1,
                    'preprocessing': {
                        'target_size': [224, 224],
                        'interpolation': 'bilinear',
                        'normalize': {'enabled': True, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                        'color_convert': 'bgr_to_rgb',
                    }
                },
                'depth': {'enabled': False},
                'instruction': {
                    'enabled': True,
                    'topic': '/navigation/instruction',
                    'msg_type': 'std_msgs/String',
                    'queue_size': 1,
                    'default': 'Navigate to the goal',
                    'persistent': True,
                },
                'odometry': {'enabled': False},
                'goal': {'enabled': False},
            },
            'outputs': {
                'mode': 'discrete',
                'discrete': {
                    'enabled': True,
                    'topic': '/navigation/action',
                    'msg_type': 'std_msgs/String',
                    'queue_size': 1,
                    'action_mapping': {
                        'MOVE_FORWARD': 'forward',
                        'TURN_LEFT': 'left',
                        'TURN_RIGHT': 'right',
                        'STOP': 'stop',
                    }
                },
                'continuous': {'enabled': False},
                'feedback': {
                    'enabled': True,
                    'topic': '/navigation/feedback',
                    'msg_type': 'std_msgs/String',
                },
                'status': {
                    'enabled': True,
                    'topic': '/navigation/status',
                    'msg_type': 'std_msgs/String',
                }
            },
            'model': {
                'variant': 'InternVLA-N1',
                'input_format': {
                    'image_format': 'base64',
                    'image_encoding': 'jpeg',
                    'jpeg_quality': 95,
                    'use_history': False,
                    'history_length': 4,
                },
                'action_space': {
                    'type': 'discrete',
                    'discrete_actions': ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP'],
                    'forward_distance': 0.25,
                    'turn_angle': 30,
                },
                'prompt_template': 'Instruction: {instruction}\nOutput the next action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.',
            },
            'transforms': {
                'use_tf': False,
            },
            'presets': {},
            'use_preset': None,
        }
    
    def _apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        preset = self.config['presets'].get(preset_name, {})
        self._deep_update(self.config, preset)
        self.get_logger().info(f"Applied preset: {preset_name}")
        
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update a dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
                
    def _init_processors(self):
        """Initialize image and depth processors."""
        rgb_config = self.config['inputs']['rgb'].get('preprocessing', {})
        self.rgb_processor = ImageProcessor(rgb_config)
        
        depth_config = self.config['inputs'].get('depth', {}).get('preprocessing', {})
        self.depth_processor = DepthProcessor(depth_config)
        
    def _setup_subscribers(self):
        """Setup all configured subscribers."""
        inputs = self.config['inputs']
        
        # RGB subscriber
        if inputs['rgb'].get('enabled', True):
            rgb_cfg = inputs['rgb']
            msg_type = Image if 'Image' in rgb_cfg.get('msg_type', 'Image') and 'Compressed' not in rgb_cfg.get('msg_type', '') else CompressedImage
            
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=rgb_cfg.get('queue_size', 1)
            )
            
            self.rgb_sub = self.create_subscription(
                msg_type,
                rgb_cfg['topic'],
                self._rgb_callback,
                qos,
                callback_group=self.input_callback_group
            )
            self.get_logger().info(f"Subscribed to RGB: {rgb_cfg['topic']}")
            
        # Depth subscriber
        if inputs.get('depth', {}).get('enabled', False):
            depth_cfg = inputs['depth']
            
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=depth_cfg.get('queue_size', 1)
            )
            
            self.depth_sub = self.create_subscription(
                Image,
                depth_cfg['topic'],
                self._depth_callback,
                qos,
                callback_group=self.input_callback_group
            )
            self.get_logger().info(f"Subscribed to Depth: {depth_cfg['topic']}")
            
        # Instruction subscriber
        if inputs.get('instruction', {}).get('enabled', True):
            inst_cfg = inputs['instruction']
            
            self.instruction_sub = self.create_subscription(
                String,
                inst_cfg['topic'],
                self._instruction_callback,
                inst_cfg.get('queue_size', 1),
                callback_group=self.input_callback_group
            )
            # Set default instruction
            self.state.current_instruction = inst_cfg.get('default', '')
            self.get_logger().info(f"Subscribed to Instruction: {inst_cfg['topic']}")
            
        # Odometry subscriber
        if inputs.get('odometry', {}).get('enabled', False):
            odom_cfg = inputs['odometry']
            
            self.odom_sub = self.create_subscription(
                Odometry,
                odom_cfg['topic'],
                self._odometry_callback,
                odom_cfg.get('queue_size', 1),
                callback_group=self.input_callback_group
            )
            self.get_logger().info(f"Subscribed to Odometry: {odom_cfg['topic']}")
            
        # Goal subscriber
        if inputs.get('goal', {}).get('enabled', False):
            goal_cfg = inputs['goal']
            
            self.goal_sub = self.create_subscription(
                PoseStamped,
                goal_cfg['topic'],
                self._goal_callback,
                goal_cfg.get('queue_size', 1),
                callback_group=self.input_callback_group
            )
            self.get_logger().info(f"Subscribed to Goal: {goal_cfg['topic']}")
            
    def _setup_publishers(self):
        """Setup all configured publishers."""
        outputs = self.config['outputs']
        
        # Discrete action publisher
        if outputs.get('discrete', {}).get('enabled', True):
            disc_cfg = outputs['discrete']
            self.action_pub = self.create_publisher(
                String,
                disc_cfg['topic'],
                disc_cfg.get('queue_size', 1)
            )
            self.get_logger().info(f"Publishing actions to: {disc_cfg['topic']}")
            
        # Continuous velocity publisher
        if outputs.get('continuous', {}).get('enabled', False):
            cont_cfg = outputs['continuous']
            self.cmd_vel_pub = self.create_publisher(
                Twist,
                cont_cfg['topic'],
                cont_cfg.get('queue_size', 1)
            )
            self.get_logger().info(f"Publishing velocities to: {cont_cfg['topic']}")
            
        # Feedback publisher
        if outputs.get('feedback', {}).get('enabled', True):
            fb_cfg = outputs['feedback']
            self.feedback_pub = self.create_publisher(
                String,
                fb_cfg['topic'],
                fb_cfg.get('queue_size', 1)
            )
            
        # Status publisher
        if outputs.get('status', {}).get('enabled', True):
            status_cfg = outputs['status']
            self.status_pub = self.create_publisher(
                String,
                status_cfg['topic'],
                status_cfg.get('queue_size', 1)
            )
            
    def _rgb_callback(self, msg):
        """Handle incoming RGB image."""
        try:
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.lock:
                self.state.current_rgb = image
                self.state.rgb_timestamp = time.time()
                
                # Add to history if needed
                model_cfg = self.config['model']['input_format']
                if model_cfg.get('use_history', False):
                    self.state.image_history.append(image.copy())
                    
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")
            
    def _depth_callback(self, msg):
        """Handle incoming depth image."""
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg)
            
            with self.lock:
                self.state.current_depth = depth
                self.state.depth_timestamp = time.time()
                
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
            
    def _instruction_callback(self, msg: String):
        """Handle incoming navigation instruction."""
        with self.lock:
            self.state.current_instruction = msg.data
            self.state.is_navigating = True
        self.get_logger().info(f"Received instruction: {msg.data}")
        
    def _odometry_callback(self, msg: Odometry):
        """Handle incoming odometry."""
        odom_cfg = self.config['inputs']['odometry']
        
        odom_data = {}
        if odom_cfg.get('use_position', True):
            odom_data['position'] = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z,
            }
        if odom_cfg.get('use_orientation', True):
            odom_data['orientation'] = {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w,
            }
        if odom_cfg.get('use_velocity', False):
            odom_data['linear_velocity'] = {
                'x': msg.twist.twist.linear.x,
                'y': msg.twist.twist.linear.y,
                'z': msg.twist.twist.linear.z,
            }
            
        with self.lock:
            self.state.current_odometry = odom_data
            
    def _goal_callback(self, msg: PoseStamped):
        """Handle incoming goal pose."""
        with self.lock:
            self.state.current_goal = {
                'position': {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z,
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w,
                }
            }
            
    def _inference_callback(self):
        """Periodic callback to run model inference."""
        control_cfg = self.config['bridge']['control']
        
        # Check minimum interval
        current_time = time.time()
        if current_time - self.state.last_inference_time < control_cfg.get('min_inference_interval', 0.1):
            return
            
        # Check if we have required inputs
        with self.lock:
            if self.state.current_rgb is None:
                return
                
            if not control_cfg.get('continuous_inference', False) and not self.state.is_navigating:
                return
                
            # Prepare inference payload
            payload = self._prepare_payload()
            
        if payload is None:
            return
            
        # Run inference
        result = self.model_client.infer(payload)
        
        if result is None:
            self._publish_status("inference_failed")
            return
            
        # Process and publish result
        self._process_result(result)
        self.state.last_inference_time = current_time
        
    def _prepare_payload(self) -> Optional[Dict]:
        """Prepare the inference payload for the model server.
        
        For InternNav, we pass raw numpy arrays:
        - 'rgb': numpy array (H, W, 3) uint8
        - 'depth': numpy array (H, W, 1) float32
        - 'instruction': string
        """
        model_cfg = self.config['model']
        input_format = model_cfg.get('input_format', {})
        
        # Get target size from config (InternNav expects specific sizes)
        target_w = input_format.get('target_width', 640)
        target_h = input_format.get('target_height', 480)
        
        # Process RGB image - resize to expected dimensions
        rgb = self.state.current_rgb
        if rgb.shape[1] != target_w or rgb.shape[0] != target_h:
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure RGB is uint8
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        
        # Build payload with raw numpy arrays
        payload = {
            'rgb': rgb,
            'instruction': self.state.current_instruction,
        }
        
        # Add depth if enabled
        if self.config['inputs'].get('depth', {}).get('enabled', False) and self.state.current_depth is not None:
            depth = self.state.current_depth
            if depth.shape[1] != target_w or depth.shape[0] != target_h:
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            # Ensure depth has shape (H, W, 1)
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
            # Ensure depth is float32
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            payload['depth'] = depth
        else:
            # InternNav requires depth, create a placeholder if not available
            payload['depth'] = np.zeros((target_h, target_w, 1), dtype=np.float32)
            
        # Add odometry if available (optional)
        if self.state.current_odometry is not None:
            payload['odometry'] = self.state.current_odometry
            
        # Add goal if available (optional)
        if self.state.current_goal is not None:
            payload['goal'] = self.state.current_goal
            
        return payload
        
    def _process_result(self, result: Dict):
        """Process model inference result and publish actions."""
        # Extract action from result
        action_str = result.get('action', result.get('output', 'STOP'))
        
        # Parse action
        action = self._parse_action(action_str)
        
        # Publish discrete action
        outputs = self.config['outputs']
        if outputs.get('discrete', {}).get('enabled', True):
            self._publish_discrete_action(action, action_str)
            
        # Publish continuous velocity
        if outputs.get('continuous', {}).get('enabled', False):
            self._publish_velocity(action)
            
        # Publish feedback
        if outputs.get('feedback', {}).get('enabled', True):
            self._publish_feedback(result, action)
            
        # Check for STOP action
        if action == ActionType.STOP:
            self.state.is_navigating = False
            self._publish_status("completed")
        else:
            self._publish_status("running")
            
    def _parse_action(self, action_str: str) -> ActionType:
        """Parse action string to ActionType enum."""
        action_upper = action_str.upper().strip()
        
        for action_type in ActionType:
            if action_type.value in action_upper:
                return action_type
                
        # Try common variations
        if 'FORWARD' in action_upper or 'AHEAD' in action_upper:
            return ActionType.MOVE_FORWARD
        elif 'LEFT' in action_upper:
            return ActionType.TURN_LEFT
        elif 'RIGHT' in action_upper:
            return ActionType.TURN_RIGHT
        elif 'STOP' in action_upper or 'DONE' in action_upper or 'FINISH' in action_upper:
            return ActionType.STOP
            
        return ActionType.UNKNOWN
        
    def _publish_discrete_action(self, action: ActionType, raw_action: str):
        """Publish discrete action to the configured topic."""
        disc_cfg = self.config['outputs']['discrete']
        action_mapping = disc_cfg.get('action_mapping', {})
        
        # Map action to simulation's expected format
        mapped_action = action_mapping.get(action.value, raw_action)
        
        msg = String()
        msg.data = mapped_action
        self.action_pub.publish(msg)
        
        self.get_logger().debug(f"Published action: {mapped_action}")
        
    def _publish_velocity(self, action: ActionType):
        """Publish velocity command based on action."""
        cont_cfg = self.config['outputs']['continuous']
        action_to_vel = cont_cfg.get('action_to_velocity', {})
        
        vel_cfg = action_to_vel.get(action.value, {'linear_x': 0.0, 'angular_z': 0.0})
        
        msg = Twist()
        msg.linear.x = vel_cfg.get('linear_x', 0.0)
        msg.angular.z = vel_cfg.get('angular_z', 0.0)
        
        # Apply limits
        limits = cont_cfg.get('limits', {})
        max_linear = limits.get('max_linear', 1.0)
        max_angular = limits.get('max_angular', 1.0)
        
        msg.linear.x = max(-max_linear, min(max_linear, msg.linear.x))
        msg.angular.z = max(-max_angular, min(max_angular, msg.angular.z))
        
        self.cmd_vel_pub.publish(msg)
        
    def _publish_feedback(self, result: Dict, action: ActionType):
        """Publish inference feedback."""
        fb_cfg = self.config['outputs'].get('feedback', {})
        
        feedback = {
            'action': action.value,
            'timestamp': time.time(),
        }
        
        if fb_cfg.get('include_confidence', True) and 'confidence' in result:
            feedback['confidence'] = result['confidence']
            
        if fb_cfg.get('include_reasoning', True) and 'reasoning' in result:
            feedback['reasoning'] = result['reasoning']
            
        if fb_cfg.get('include_raw_output', False):
            feedback['raw_output'] = result
            
        msg = String()
        msg.data = json.dumps(feedback)
        self.feedback_pub.publish(msg)
        
    def _publish_status(self, status_key: str):
        """Publish navigation status."""
        status_cfg = self.config['outputs'].get('status', {})
        messages = status_cfg.get('messages', {})
        
        status_msg = messages.get(status_key, status_key)
        
        msg = String()
        msg.data = status_msg
        self.status_pub.publish(msg)
        
    def _log_configuration(self):
        """Log the current configuration summary."""
        self.get_logger().info("=" * 50)
        self.get_logger().info("InternNav Bridge Configuration:")
        self.get_logger().info(f"  Model: {self.config['model'].get('variant', 'InternVLA-N1')}")
        self.get_logger().info(f"  Server: {self.config['bridge']['server']['host']}:{self.config['bridge']['server']['port']}")
        self.get_logger().info(f"  RGB Topic: {self.config['inputs']['rgb']['topic']}")
        self.get_logger().info(f"  Instruction Topic: {self.config['inputs']['instruction']['topic']}")
        self.get_logger().info(f"  Action Topic: {self.config['outputs']['discrete']['topic']}")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    
    # Get config path from command line or environment
    import sys
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--config' and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
            
    node = InternNavBridge(config_path)
    
    # Use multi-threaded executor for parallel callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
