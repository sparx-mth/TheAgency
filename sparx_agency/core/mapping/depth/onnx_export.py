import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
H = W = 504  # recommend multiple-of-14 for clean TensorRT

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval().cuda()

# Dummy input
dummy = torch.zeros(1, 3, H, W, device="cuda", dtype=torch.float16)

class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        out = self.m(pixel_values=pixel_values)
        # predicted_depth: (B, 1, H, W) float
        return out.predicted_depth

wrapped = Wrapper(model).eval().half()

torch.onnx.export(
    wrapped,
    (dummy,),
    "depth_anything_v2_metric_indoor_small_504.onnx",
    input_names=["pixel_values"],
    output_names=["predicted_depth"],
    opset_version=17,
    do_constant_folding=True,
)
