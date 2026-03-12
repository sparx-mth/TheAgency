import sys
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.robots.common.spatial_math import intrinsics_from_fov, load_intrinsics_from_yaml


class DA3TensorRTModel(DepthModel):
    def __init__(self, engine_path: str, yaml_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(e)

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.intrinsics = load_intrinsics_from_yaml(yaml_path)
        print(f"Loaded Intrinsics: {self.intrinsics.fx}x{self.intrinsics.fy}")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        # In TRT 10+, iterating over the engine yields tensor NAMES (strings)
        for tensor_name in self.engine:
            # Get shape and dtype using the string name
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Add to bindings list (still needs the memory address)
            bindings.append(int(device_mem))

            # Check if it's an input or output using get_tensor_mode
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})

        return inputs, outputs, bindings, stream


    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Implements abstract method: Returns Metric Depth Map"""
        depth, _ = self.infer_all(rgb)
        return depth

    def infer_all(self, frame: np.ndarray):
        """Returns (Depth Map, Point Cloud)"""
        # --- 1. Pre-process ---
        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)[2:]  # e.g. (518, 518)

        img = cv2.resize(frame, (input_shape[1], input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)).ravel()

        # --- 2. Execute TensorRT ---
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            print(i, name, self.engine.get_tensor_shape(name), self.engine.get_tensor_mode(name))
            self.context.set_tensor_address(name, self.bindings[i])

        self.inputs[0]['host'] = np.ascontiguousarray(img)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute and sync
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        # --- 3. Post-process Depth ---
        # Find the depth output by name (safer than outputs[0])
        depth_out = None
        for out in self.outputs:
            if "depth" in out["name"].lower():
                depth_out = out
                break
        if depth_out is None:
            depth_out = self.outputs[0]  # fallback

        out_shape = tuple(self.engine.get_tensor_shape(depth_out["name"]))

        # Common cases: (1,1,H,W) or (1,H,W) or (H,W)
        if len(out_shape) == 4:
            _, _, h, w = out_shape
        elif len(out_shape) == 3:
            _, h, w = out_shape
        elif len(out_shape) == 2:
            h, w = out_shape
        else:
            raise ValueError(f"Unexpected depth output shape: {out_shape} for tensor {depth_out['name']}")

        depth_map = depth_out["host"].reshape(h, w).astype(np.float32)
        print("Depth output tensor:", depth_out["name"], "shape:", out_shape, "depth_map:", depth_map.shape)
        # --- 4. Vectorized Point Cloud Projection with Scaled Intrinsics ---
        h, w = depth_map.shape
        # Scale intrinsics from original frame (640x480) to depth map size (504x280)
        # Note: Use frame.shape[1] for width, [0] for height
        scale_x = w / frame.shape[1]
        scale_y = h / frame.shape[0]

        fx = self.intrinsics.fx * scale_x
        fy = self.intrinsics.fy * scale_y
        cx = self.intrinsics.cx * scale_x
        cy = self.intrinsics.cy * scale_y

        i, j = np.indices((h, w))
        z = depth_map
        # Using the same coordinate logic as your C++ snippet
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy

        # optical -> mapper convention: [Left, Up, Forward]
        x_left = -x  # left = -right
        y_up = -y  # up   = -down
        z_fwd = z

        point_cloud = np.stack((x_left, y_up, z_fwd), axis=-1).astype(np.float32)

        return depth_map, point_cloud