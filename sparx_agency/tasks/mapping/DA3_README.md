## Generating DepthAnything V3 TensorRT Engines
This guide explains how to build and deploy the DepthAnything V3 (DA3) metric depth model using TensorRT for real-time robotics mapping.

### 1. Clone the Repository
Clone the specialized DA3 TensorRT integration repository into your workspace:
```bash
mkdir -p ~/depth_anything_ws/src
cd ~/depth_anything_ws/src
git clone https://github.com/daphnaa/ros2-depth-anything-v3-trt.git
cd ros2-depth-anything-v3-trt
```
### 2. Download DA3 Weights
Get the ONNX model (Two Options): 
- A. Download the ONNX file from Huggingface: 
  - https://huggingface.co/depth-anything/DA3METRIC-LARGE
  - https://huggingface.co/depth-anything/DA3-SMALL (Note: not metric version)
  
- B. Generate ONNX following the instruction [here](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt/blob/main/onnx/README.md)

Place model file: Put the ONNX/engine file in the models/ directory

Ensure you are using the Metric version for accurate point cloud projection.

### 3. Build and Install Dependencies
Follow the internal build instructions to compile the required TensorRT plugins and dependencies:
```bash
cd ~/depth_anything_ws
# From your ROS 2 workspace
colcon build --packages-select depth_anything_v3 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```
### 4. Generate the TensorRT Engine
Use the provided generation script to convert the ONNX model into a serialized .engine file optimized for your NVIDIA Jetson (AGX Orin/Nano).
```bash
cd ~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx
# Generate the engine
python3 generate_engine.py \
    --onnx ../onnx/DA3METRIC-LARGE.onnx \
    --output ../onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine \
    --fp16 
```
Note: Using --fp16 is highly recommended on Jetson hardware for a significant performance boost with minimal accuracy loss.

### 5. Switch git branch
Switch to the `potential_field_daphna` branch of the repository to use the latest version of the code.
```bash 
git checkout potential_field_daphna
```

### 6. Run the Mapping Pipeline
Once the engine is generated, use the specialized DA3 mapping script to process a video stream or camera input.
```bash
source /opt/ros/humble/setup.bash
python3 sparx_agency/tasks/mapping/create_map_from_video_with_da3.py \
    --engine /home/$USER/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine \
    --yaml /home/$USER/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_exmple.yaml \
```