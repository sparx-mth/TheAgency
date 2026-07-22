import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401 — initializes CUDA context before any pycuda.driver calls
import pycuda.autoinit
import pycuda.driver as cuda

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.common.spatial_math import load_intrinsics_from_yaml


class DA3TensorRTModel(DepthModel):
    def __init__(self, engine_path: str, yaml_path: str, log_fn=None):
        self.log_fn = log_fn
        self.enable_timing = True
        self.timing_log_period_sec = 1.0
        self._last_timing_log_t = 0.0
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}: {e}") from e



        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.intrinsics = load_intrinsics_from_yaml(yaml_path)

        self.input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        self._log_info(f"Loaded Intrinsics: {self.intrinsics.fx}x{self.intrinsics.fy}")

        # --- 2. Execute TensorRT ---
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # print(i, name, self.engine.get_tensor_shape(name), self.engine.get_tensor_mode(name))
            self.context.set_tensor_address(name, self.bindings[i])

        self.depth_output = next(
            out for out in self.outputs
            if "depth" in out["name"].lower()
        )

    def _log_info(self, text: str):
        if self.log_fn is not None:
            self.log_fn(text)
        else:
            print(text)

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
        return self.infer_all(rgb)

    def infer_all(self, frame: np.ndarray) -> np.ndarray:
        """Returns depth map (H, W) float32. BGR input."""
        t0 = time.perf_counter()

        img = cv2.resize(
            frame,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) * (1.0 / 255.0)
        img = np.transpose(img, (2, 0, 1)).ravel()
        t1 = time.perf_counter()

        np.copyto(self.inputs[0]["host"], img)
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream,
        )
        t2 = time.perf_counter()

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        t3 = time.perf_counter()

        cuda.memcpy_dtoh_async(
            self.depth_output["host"],
            self.depth_output["device"],
            self.stream,
        )
        self.stream.synchronize()
        t4 = time.perf_counter()

        out_shape = tuple(self.engine.get_tensor_shape(self.depth_output["name"]))

        if len(out_shape) == 4:
            _, _, h, w = out_shape
        elif len(out_shape) == 3:
            _, h, w = out_shape
        elif len(out_shape) == 2:
            h, w = out_shape
        else:
            raise ValueError(
                f"Unexpected depth output shape: {out_shape} "
                f"for tensor {self.depth_output['name']}"
            )

        depth = self.depth_output["host"].reshape(h, w).astype(np.float32, copy=False)
        t5 = time.perf_counter()

        now = time.perf_counter()
        if self.enable_timing and (now - self._last_timing_log_t) >= self.timing_log_period_sec:
            self._last_timing_log_t = now
            self._log_info(
                "da3_trt timing ms: "
                f"preprocess={(t1 - t0) * 1000.0:.1f}, "
                f"h2d={(t2 - t1) * 1000.0:.1f}, "
                f"execute_submit={(t3 - t2) * 1000.0:.1f}, "
                f"d2h_sync={(t4 - t3) * 1000.0:.1f}, "
                f"post={(t5 - t4) * 1000.0:.1f}, "
                f"total={(t5 - t0) * 1000.0:.1f}"
            )

        return depth

    def infer_pointcloud(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (depth_map, point_cloud) in [Left, Up, Forward] convention."""
        depth_map = self.infer_all(frame)
        h, w = depth_map.shape

        scale_x = w / frame.shape[1]
        scale_y = h / frame.shape[0]
        fx = self.intrinsics.fx * scale_x
        fy = self.intrinsics.fy * scale_y
        cx = self.intrinsics.cx * scale_x
        cy = self.intrinsics.cy * scale_y

        i, j = np.indices((h, w))
        z = depth_map
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy

        point_cloud = np.stack((-x, -y, z), axis=-1).astype(np.float32)
        return depth_map, point_cloud
