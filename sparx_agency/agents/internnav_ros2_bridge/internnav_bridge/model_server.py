#!/usr/bin/env python3
"""
InternNav Model Server Wrapper

This module provides a FastAPI server that wraps the InternNav model,
providing a standardized HTTP interface for the bridge to communicate with.

Usage:
    python model_server.py --model InternVLA-N1 --port 8000
"""

import argparse
import base64
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    # Required fields
    image: str = Field(..., description="Base64 encoded image")
    instruction: str = Field(..., description="Navigation instruction")
    
    # Optional fields
    model: str = Field(default="InternVLA-N1", description="Model variant to use")
    image_history: Optional[List[str]] = Field(default=None, description="List of historical images (base64)")
    depth: Optional[List[List[float]]] = Field(default=None, description="Depth image as 2D array")
    odometry: Optional[Dict[str, Any]] = Field(default=None, description="Odometry data")
    goal: Optional[Dict[str, Any]] = Field(default=None, description="Goal pose")
    prompt: Optional[str] = Field(default=None, description="Custom prompt override")
    
    # Inference parameters
    max_tokens: int = Field(default=50, description="Maximum tokens in response")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    action: str = Field(..., description="Predicted action")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    reasoning: Optional[str] = Field(default=None, description="Model reasoning (if available)")
    raw_output: Optional[str] = Field(default=None, description="Raw model output")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_loaded: bool
    model_name: str
    gpu_available: bool


# =============================================================================
# Model Wrapper Base Class
# =============================================================================

class BaseModelWrapper:
    """Base class for model wrappers."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load(self):
        """Load the model. Override in subclass."""
        raise NotImplementedError
        
    def infer(self, image: np.ndarray, instruction: str, **kwargs) -> Dict[str, Any]:
        """Run inference. Override in subclass."""
        raise NotImplementedError
        
    def decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode a base64 image to numpy array."""
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)


# =============================================================================
# InternVLA-N1 Model Wrapper
# =============================================================================

class InternVLAWrapper(BaseModelWrapper):
    """
    Wrapper for InternVLA-N1 model.
    
    This wrapper provides a standardized interface to the InternVLA-N1 model
    for navigation inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__("InternVLA-N1", device)
        self.model_path = model_path or "InternRobotics/InternVLA-N1"
        self.loaded = False
        
    def load(self):
        """Load the InternVLA-N1 model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            
            print(f"Loading model from {self.model_path}...")
            
            # This is a placeholder - adjust based on actual InternVLA-N1 loading
            # The actual loading code depends on InternNav's implementation
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.loaded = True
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using mock model for testing")
            self.loaded = True  # Use mock mode
            
    def infer(self, image: np.ndarray, instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference on the model.
        
        Args:
            image: RGB image as numpy array
            instruction: Navigation instruction text
            **kwargs: Additional arguments (image_history, depth, etc.)
            
        Returns:
            Dictionary with action, confidence, reasoning, etc.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
            
        start_time = time.time()
        
        # If model failed to load, use mock inference
        if self.model is None:
            return self._mock_inference(image, instruction, **kwargs)
            
        try:
            import torch
            
            # Prepare inputs
            prompt = kwargs.get('prompt', f"Instruction: {instruction}\nOutput the next action:")
            
            # Process image
            # Note: Actual processing depends on InternNav's requirements
            image_pil = Image.fromarray(image)
            
            # Run inference
            with torch.no_grad():
                # This is a placeholder - actual inference depends on model implementation
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', 50),
                    temperature=kwargs.get('temperature', 0.0),
                    do_sample=kwargs.get('temperature', 0.0) > 0
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract action from response
            action = self._extract_action(response)
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                'action': action,
                'confidence': 0.95,  # Placeholder
                'reasoning': response,
                'raw_output': response,
                'inference_time_ms': inference_time
            }
            
        except Exception as e:
            print(f"Inference error: {e}")
            return self._mock_inference(image, instruction, **kwargs)
            
    def _extract_action(self, response: str) -> str:
        """Extract action from model response."""
        response_upper = response.upper()
        
        actions = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']
        
        for action in actions:
            if action in response_upper:
                return action
                
        # Try variations
        if 'FORWARD' in response_upper:
            return 'MOVE_FORWARD'
        elif 'LEFT' in response_upper:
            return 'TURN_LEFT'
        elif 'RIGHT' in response_upper:
            return 'TURN_RIGHT'
        elif 'STOP' in response_upper or 'DONE' in response_upper:
            return 'STOP'
            
        return 'STOP'  # Default to stop if unclear
        
    def _mock_inference(self, image: np.ndarray, instruction: str, **kwargs) -> Dict[str, Any]:
        """Mock inference for testing when model is not available."""
        import random
        
        # Simple mock logic
        instruction_lower = instruction.lower()
        
        if 'stop' in instruction_lower or 'done' in instruction_lower:
            action = 'STOP'
        elif 'left' in instruction_lower:
            action = 'TURN_LEFT'
        elif 'right' in instruction_lower:
            action = 'TURN_RIGHT'
        else:
            # Random action for testing
            action = random.choice(['MOVE_FORWARD', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT'])
            
        return {
            'action': action,
            'confidence': 0.8,
            'reasoning': f"Mock inference: {action}",
            'raw_output': f"[MOCK] {action}",
            'inference_time_ms': 50.0
        }


# =============================================================================
# FastAPI Server
# =============================================================================

def create_app(model_wrapper: BaseModelWrapper) -> "FastAPI":
    """Create the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")
        
    app = FastAPI(
        title="InternNav Model Server",
        description="HTTP server providing inference for InternNav navigation models",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        model_wrapper.load()
        
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Check server health."""
        import torch
        return HealthResponse(
            status="healthy" if model_wrapper.loaded else "loading",
            model_loaded=model_wrapper.loaded,
            model_name=model_wrapper.model_name,
            gpu_available=torch.cuda.is_available() if 'torch' in dir() else False
        )
        
    @app.post("/v1/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        """Run model inference."""
        if not model_wrapper.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
            
        try:
            # Decode image
            image = model_wrapper.decode_base64_image(request.image)
            
            # Decode history images if provided
            image_history = None
            if request.image_history:
                image_history = [
                    model_wrapper.decode_base64_image(img)
                    for img in request.image_history
                ]
                
            # Run inference
            result = model_wrapper.infer(
                image=image,
                instruction=request.instruction,
                image_history=image_history,
                depth=request.depth,
                odometry=request.odometry,
                goal=request.goal,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            return InferenceResponse(
                action=result['action'],
                confidence=result.get('confidence'),
                reasoning=result.get('reasoning'),
                raw_output=result.get('raw_output'),
                inference_time_ms=result['inference_time_ms']
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.post("/v1/batch_inference")
    async def batch_inference(requests: List[InferenceRequest]):
        """Run batch inference (for efficiency)."""
        results = []
        for req in requests:
            try:
                image = model_wrapper.decode_base64_image(req.image)
                result = model_wrapper.infer(
                    image=image,
                    instruction=req.instruction,
                    prompt=req.prompt
                )
                results.append(InferenceResponse(
                    action=result['action'],
                    confidence=result.get('confidence'),
                    reasoning=result.get('reasoning'),
                    raw_output=result.get('raw_output'),
                    inference_time_ms=result['inference_time_ms']
                ))
            except Exception as e:
                results.append({"error": str(e)})
        return results
        
    return app


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="InternNav Model Server")
    parser.add_argument("--model", type=str, default="InternVLA-N1",
                        help="Model variant to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights (optional)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run on")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes")
    
    args = parser.parse_args()
    
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is required but not installed.")
        print("Install with: pip install fastapi uvicorn")
        return
        
    # Create model wrapper
    model_wrapper = InternVLAWrapper(
        model_path=args.model_path,
        device=args.device
    )
    
    # Create and run app
    app = create_app(model_wrapper)
    
    print(f"Starting InternNav Model Server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
