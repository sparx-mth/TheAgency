#!/usr/bin/env python3
"""
InternNav Agent Client

Client for the InternNav AgentServer API.
Handles the stateful agent lifecycle: init -> step -> reset

API Endpoints:
- POST /agent/init - Initialize agent with config
- POST /agent/{agent_name}/step - Get action from observation  
- POST /agent/{agent_name}/reset - Reset agent state
"""

import base64
import io
import json
import time
from typing import Dict, List, Optional, Any, Tuple
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
    server_host: str = "localhost"
    server_port: int = 8087
    model_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "ckpt_path": self.ckpt_path,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "model_settings": self.model_settings
        }


@dataclass 
class StepResponse:
    """Response from a step request."""
    action: str = "STOP"
    action_index: int = 0
    raw_response: Dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    success: bool = True
    error: str = ""


class InternNavAgentClient:
    """
    Client for InternNav AgentServer.
    
    Usage:
        client = InternNavAgentClient("127.0.0.1", 8087)
        
        # Initialize agent (once)
        client.init_agent(AgentConfig(model_name="InternVLA-N1"))
        
        # Step loop
        while navigating:
            response = client.step(rgb_image, instruction)
            execute_action(response.action)
            
        # Reset for new episode
        client.reset()
    """
    
    # Standard VLN-CE action mapping
    ACTION_NAMES = {
        0: "STOP",
        1: "MOVE_FORWARD",
        2: "TURN_LEFT",
        3: "TURN_RIGHT",
        4: "LOOK_UP",
        5: "LOOK_DOWN",
    }
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8087,
        timeout: float = 60.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        
        # Agent state
        self.agent_name: Optional[str] = None
        self.is_initialized: bool = False
        
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif CV2_AVAILABLE:
            # Ensure RGB->BGR for cv2
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buffer).decode("utf-8")
        else:
            raise RuntimeError("Neither PIL nor cv2 available for image encoding")
    
    def _encode_image_bytes(self, image: np.ndarray) -> bytes:
        """Encode numpy image to bytes."""
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()
        elif CV2_AVAILABLE:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buffer.tobytes()
        else:
            raise RuntimeError("Neither PIL nor cv2 available for image encoding")
            
    def check_connection(self) -> bool:
        """Check if server is reachable."""
        if not self.session:
            return False
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def init_agent(self, config: AgentConfig) -> Tuple[bool, str]:
        """
        Initialize an agent on the server.
        
        Args:
            config: Agent configuration
            
        Returns:
            Tuple of (success, agent_name or error message)
        """
        if not self.session:
            return False, "requests library not available"
            
        url = f"{self.base_url}/agent/init"
        
        payload = {
            "agent_config": config.to_dict()
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 201:
                result = response.json()
                # Extract agent name from response
                self.agent_name = result.get("agent_name", config.model_name)
                self.is_initialized = True
                print(f"[InternNavClient] Agent initialized: {self.agent_name}")
                return True, self.agent_name
            else:
                error = f"Init failed: HTTP {response.status_code} - {response.text}"
                print(f"[InternNavClient] {error}")
                return False, error
                
        except Exception as e:
            error = f"Init error: {e}"
            print(f"[InternNavClient] {error}")
            return False, error
    
    def step(
        self,
        rgb: np.ndarray,
        instruction: str,
        depth: Optional[np.ndarray] = None,
        pose: Optional[Dict] = None,
        **kwargs
    ) -> StepResponse:
        """
        Execute one navigation step.
        
        Args:
            rgb: RGB image as numpy array (H, W, 3), values 0-255
            instruction: Navigation instruction text
            depth: Optional depth image
            pose: Optional pose/odometry data
            
        Returns:
            StepResponse with action and metadata
        """
        if not self.session:
            return StepResponse(success=False, error="requests library not available")
            
        if not self.is_initialized or not self.agent_name:
            return StepResponse(success=False, error="Agent not initialized. Call init_agent() first.")
        
        url = f"{self.base_url}/agent/{self.agent_name}/step"
        
        # Build observation dict
        # The exact format depends on the model - try common formats
        observation = {
            "rgb": self._encode_image_base64(rgb),
            "instruction": instruction,
        }
        
        # Alternative field names that models might expect
        observation["text"] = instruction
        observation["image"] = observation["rgb"]
        
        # Add depth if provided
        if depth is not None:
            # Normalize depth to 0-255 if needed
            if depth.max() <= 1.0:
                depth_uint8 = (depth * 255).astype(np.uint8)
            else:
                depth_uint8 = depth.astype(np.uint8)
            observation["depth"] = self._encode_image_base64(
                np.stack([depth_uint8] * 3, axis=-1) if len(depth_uint8.shape) == 2 else depth_uint8
            )
            
        # Add pose if provided
        if pose is not None:
            observation["pose"] = pose
            observation["odometry"] = pose
            
        # Add any extra kwargs
        observation.update(kwargs)
        
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
                result = response.json()
                
                # Parse action from response
                action, action_idx = self._parse_action(result)
                
                return StepResponse(
                    action=action,
                    action_index=action_idx,
                    raw_response=result,
                    inference_time_ms=inference_time,
                    success=True
                )
            else:
                return StepResponse(
                    success=False,
                    error=f"Step failed: HTTP {response.status_code} - {response.text[:200]}",
                    inference_time_ms=inference_time
                )
                
        except requests.exceptions.Timeout:
            return StepResponse(success=False, error=f"Timeout after {self.timeout}s")
        except Exception as e:
            return StepResponse(success=False, error=f"Step error: {e}")
    
    def _parse_action(self, result: Dict) -> Tuple[str, int]:
        """Parse action from server response."""
        # Try different response formats
        
        # Format 1: action as index
        if "action" in result:
            action = result["action"]
            if isinstance(action, int):
                return self.ACTION_NAMES.get(action, "STOP"), action
            elif isinstance(action, str):
                # Find index for string action
                action_upper = action.upper()
                for idx, name in self.ACTION_NAMES.items():
                    if name in action_upper or action_upper in name:
                        return name, idx
                return action_upper, 0
            elif isinstance(action, list) and len(action) > 0:
                # Action might be a list, take first element
                return self._parse_action({"action": action[0]})
                
        # Format 2: action_index field
        if "action_index" in result:
            idx = result["action_index"]
            return self.ACTION_NAMES.get(idx, "STOP"), idx
            
        # Format 3: output field
        if "output" in result:
            return self._parse_action({"action": result["output"]})
            
        # Format 4: prediction field
        if "prediction" in result:
            return self._parse_action({"action": result["prediction"]})
            
        # Default
        return "STOP", 0
    
    def reset(self, reset_index: Optional[List] = None) -> bool:
        """
        Reset the agent for a new episode.
        
        Args:
            reset_index: Optional reset indices
            
        Returns:
            True if successful
        """
        if not self.session or not self.agent_name:
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
                print(f"[InternNavClient] Reset failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[InternNavClient] Reset error: {e}")
            return False
            
    def close(self):
        """Close the client session."""
        if self.session:
            self.session.close()


def test_client():
    """Interactive test of the InternNav client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test InternNav Agent Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8087, help="Server port")
    parser.add_argument("--model", default="InternVLA-N1", help="Model name")
    parser.add_argument("--ckpt", default="", help="Checkpoint path")
    args = parser.parse_args()
    
    print(f"Testing InternNav Agent Client")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 50)
    
    client = InternNavAgentClient(args.host, args.port)
    
    # Test connection
    print("\n[1] Testing connection...")
    if client.check_connection():
        print("    ✓ Server reachable")
    else:
        print("    ✗ Cannot reach server")
        return
    
    # Initialize agent
    print("\n[2] Initializing agent...")
    config = AgentConfig(
        model_name=args.model,
        ckpt_path=args.ckpt,
        model_settings={}
    )
    
    success, result = client.init_agent(config)
    if success:
        print(f"    ✓ Agent initialized: {result}")
    else:
        print(f"    ✗ Init failed: {result}")
        print("\n    Note: You may need to provide a valid model config.")
        print("    Check the InternNav documentation for required settings.")
        return
    
    # Test step with dummy image
    print("\n[3] Testing step with dummy observation...")
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    instruction = "Walk forward to the door"
    
    response = client.step(dummy_rgb, instruction)
    
    if response.success:
        print(f"    ✓ Step successful")
        print(f"    Action: {response.action} (index: {response.action_index})")
        print(f"    Inference time: {response.inference_time_ms:.1f}ms")
    else:
        print(f"    ✗ Step failed: {response.error}")
    
    # Reset
    print("\n[4] Testing reset...")
    if client.reset():
        print("    ✓ Reset successful")
    else:
        print("    ✗ Reset failed")
    
    client.close()
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_client()
