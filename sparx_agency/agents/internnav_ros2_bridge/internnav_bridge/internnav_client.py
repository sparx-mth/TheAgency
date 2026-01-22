#!/usr/bin/env python3
"""
InternNav Agent Client

Client for the InternNav Agent Server API.
This handles the stateful agent lifecycle: init -> step -> reset

API Endpoints:
- POST /agent/init - Initialize agent with model config
- POST /agent/{agent_name}/step - Get action from observation
- POST /agent/{agent_name}/reset - Reset for new episode
"""

import base64
import io
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Run: pip install requests")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for initializing an InternNav agent."""
    model_name: str = "InternVLA-N1"
    ckpt_path: str = ""
    model_settings: Dict[str, Any] = field(default_factory=dict)
    server_host: str = "localhost"
    server_port: int = 8087


@dataclass 
class StepResponse:
    """Response from agent step."""
    action: str = "STOP"
    action_index: int = 0
    raw_response: Dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    success: bool = True
    error: str = ""


class InternNavAgentClient:
    """
    Client for InternNav Agent Server.
    
    Usage:
        client = InternNavAgentClient(host="127.0.0.1", port=8087)
        
        # Initialize agent (do this once)
        client.init_agent(model_name="InternVLA-N1", ckpt_path="/path/to/model")
        
        # Reset for new episode
        client.reset()
        
        # Step loop
        while navigating:
            response = client.step(rgb_image, instruction)
            action = response.action  # "MOVE_FORWARD", "TURN_LEFT", etc.
    """
    
    # Standard VLN-CE action space
    ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8087,
        timeout: float = 30.0,
        agent_name: str = "nav_agent"
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.agent_name = agent_name
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.initialized = False
        
    def check_connection(self) -> bool:
        """Check if server is reachable."""
        if not self.session:
            return False
        try:
            response = self.session.get(
                f"{self.base_url}/openapi.json",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
            
    def init_agent(
        self,
        model_name: str = "InternVLA-N1",
        ckpt_path: str = "",
        model_settings: Optional[Dict] = None
    ) -> bool:
        """
        Initialize the agent on the server.
        
        This loads the model and prepares it for inference.
        Call this once before starting navigation.
        
        Args:
            model_name: Name of the model to use
            ckpt_path: Path to model checkpoint
            model_settings: Additional model configuration
            
        Returns:
            True if initialization successful
        """
        if not self.session:
            print("Error: requests library not available")
            return False
            
        url = f"{self.base_url}/agent/init"
        
        payload = {
            "agent_config": {
                "model_name": model_name,
                "ckpt_path": ckpt_path,
                "model_settings": model_settings or {},
                "server_host": self.host,
                "server_port": self.port
            }
        }
        
        try:
            print(f"[InternNavClient] Initializing agent '{self.agent_name}'...")
            print(f"[InternNavClient] Model: {model_name}")
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout * 3  # Init takes longer
            )
            
            if response.status_code == 201:
                print(f"[InternNavClient] Agent initialized successfully")
                self.initialized = True
                
                # Try to extract agent name from response
                try:
                    data = response.json()
                    if "agent_name" in data:
                        self.agent_name = data["agent_name"]
                except:
                    pass
                    
                return True
            else:
                print(f"[InternNavClient] Init failed: HTTP {response.status_code}")
                print(f"[InternNavClient] Response: {response.text[:500]}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[InternNavClient] Init timeout (this is normal for model loading)")
            # Might still have succeeded - check with a step
            self.initialized = True
            return True
        except Exception as e:
            print(f"[InternNavClient] Init error: {e}")
            return False
            
    def reset(self, reset_index: Optional[List] = None) -> bool:
        """
        Reset the agent for a new episode.
        
        Call this when starting a new navigation task.
        
        Args:
            reset_index: Optional reset indices
            
        Returns:
            True if reset successful
        """
        if not self.session:
            return False
            
        url = f"{self.base_url}/agent/{self.agent_name}/reset"
        
        payload = {
            "reset_index": reset_index
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                print(f"[InternNavClient] Agent reset successful")
                return True
            else:
                print(f"[InternNavClient] Reset failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[InternNavClient] Reset error: {e}")
            return False
            
    def step(
        self,
        rgb: np.ndarray,
        instruction: str,
        depth: Optional[np.ndarray] = None,
        **extra_obs
    ) -> StepResponse:
        """
        Execute one navigation step.
        
        Args:
            rgb: RGB image as numpy array (H, W, 3), uint8
            instruction: Navigation instruction text
            depth: Optional depth image
            **extra_obs: Additional observation data
            
        Returns:
            StepResponse with action and metadata
        """
        if not self.session:
            return StepResponse(success=False, error="requests not available")
            
        url = f"{self.base_url}/agent/{self.agent_name}/step"
        
        # Build observation payload
        observation = self._build_observation(rgb, instruction, depth, **extra_obs)
        
        payload = {
            "observation": observation
        }
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            inference_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return self._parse_step_response(response.json(), inference_time)
            else:
                return StepResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    inference_time_ms=inference_time
                )
                
        except requests.exceptions.Timeout:
            return StepResponse(
                success=False,
                error=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            return StepResponse(
                success=False,
                error=str(e)
            )
            
    def _build_observation(
        self,
        rgb: np.ndarray,
        instruction: str,
        depth: Optional[np.ndarray] = None,
        **extra
    ) -> Dict:
        """Build the observation dict for step request."""
        obs = {
            "instruction": instruction,
        }
        
        # Encode RGB image
        rgb_b64 = self._encode_image(rgb)
        obs["rgb"] = rgb_b64
        
        # Also add as common alternative field names
        obs["image"] = rgb_b64
        obs["rgb_image"] = rgb_b64
        
        # Add depth if provided
        if depth is not None:
            depth_normalized = depth
            if depth.max() > 1:
                depth_normalized = depth / depth.max()
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            obs["depth"] = self._encode_image(depth_uint8)
            
        # Add any extra observation data
        obs.update(extra)
        
        return obs
        
    def _encode_image(self, image: np.ndarray, quality: int = 90) -> str:
        """Encode image to base64 JPEG string."""
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif CV2_AVAILABLE:
            # Ensure BGR for cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buffer).decode("utf-8")
        else:
            raise RuntimeError("Neither PIL nor OpenCV available for image encoding")
            
    def _parse_step_response(self, data: Dict, inference_time: float) -> StepResponse:
        """Parse the step response from server."""
        # Try to extract action from various possible field names
        action = "STOP"
        action_index = 0
        
        # Check for action index (numeric)
        if "action" in data:
            action_val = data["action"]
            if isinstance(action_val, (int, float)):
                action_index = int(action_val)
                if 0 <= action_index < len(self.ACTION_NAMES):
                    action = self.ACTION_NAMES[action_index]
            elif isinstance(action_val, str):
                action = action_val.upper()
                if action in self.ACTION_NAMES:
                    action_index = self.ACTION_NAMES.index(action)
                    
        # Check alternative field names
        for field in ["action_index", "action_id", "pred_action"]:
            if field in data and isinstance(data[field], (int, float)):
                action_index = int(data[field])
                if 0 <= action_index < len(self.ACTION_NAMES):
                    action = self.ACTION_NAMES[action_index]
                break
                
        return StepResponse(
            action=action,
            action_index=action_index,
            raw_response=data,
            inference_time_ms=inference_time,
            success=True
        )


def test_client():
    """Test the InternNav agent client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test InternNav agent client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8087, help="Server port")
    parser.add_argument("--model", default="InternVLA-N1", help="Model name")
    parser.add_argument("--ckpt", default="", help="Checkpoint path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("InternNav Agent Client Test")
    print("=" * 60)
    
    client = InternNavAgentClient(host=args.host, port=args.port)
    
    # Test connection
    print(f"\n[1] Testing connection to {args.host}:{args.port}...")
    if client.check_connection():
        print("    ✓ Server is reachable")
    else:
        print("    ✗ Cannot reach server")
        return 1
        
    # Test init
    print(f"\n[2] Initializing agent with model '{args.model}'...")
    if client.init_agent(model_name=args.model, ckpt_path=args.ckpt):
        print("    ✓ Agent initialized")
    else:
        print("    ✗ Agent initialization failed")
        print("    Note: You may need to provide --ckpt path")
        return 1
        
    # Test reset
    print(f"\n[3] Resetting agent...")
    if client.reset():
        print("    ✓ Agent reset")
    else:
        print("    ✗ Agent reset failed")
        
    # Test step with dummy image
    print(f"\n[4] Testing step with dummy observation...")
    dummy_rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    instruction = "Move forward to the door"
    
    response = client.step(dummy_rgb, instruction)
    
    if response.success:
        print(f"    ✓ Step successful")
        print(f"    Action: {response.action} (index: {response.action_index})")
        print(f"    Inference time: {response.inference_time_ms:.1f}ms")
        print(f"    Raw response: {json.dumps(response.raw_response, indent=2)[:500]}")
    else:
        print(f"    ✗ Step failed: {response.error}")
        return 1
        
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(test_client())
