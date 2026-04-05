---
trigger: always_on
---

# Rule: Spatial Computation Standards

- **Depth Projection**: When converting DepthAnything V3 output to Point Cloud, the agent MUST use the camera intrinsic matrix $K$.
- **TensorRT Optimization**: Since we are using the TRT version, the agent must verify `trt_engine` existence before inference.
- **Occupancy Logic**: 
    - Temporary Map: $M_{temp}$ (Current Frame).
    - Accumulated Map: $M_{acc} = (1 - \alpha)M_{acc} + \alpha M_{temp}$.
- **Potential Field**: Calculate the gradient $\nabla U_{rep}$ using a workspace-clearing cost function to ensure the gradient flows away from probability peaks.

- **DepthAnything Integrity**: Use @depth_handler.py. If modifying inference logic, ensure the output tensor shape is validated before saving.
- **Testing Requirement**: Any change to potential_field.py must be followed by a simulation run. Use Model Decision mode to trigger this rule whenever physics logic is touched.
- **Documentation**: Every new function must include Google-style docstrings and a complexity note ($O(n)$).