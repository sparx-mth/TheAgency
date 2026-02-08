# Depth-Based Object Masking Pipeline

- [x] Design the pipeline
    - [x] Create an `implementation_plan.md`
- [x] Implement the solution
    - [x] Create `clean_mask.py` with `DepthMasker` (RANSAC & Gradient)
    - [x] Create `clean_mask_test.py`
    - [x] Implement floor filtering (vertical gradient or plane fitting)
    - [x] Implement wall/background filtering (thresholding)
    - [x] Combine to create a mask for objects
- [x] Verify the solution
    - [x] Run the script on test images
    - [x] Analyze failure cases (floor remaining, objects lost, walls remaining)
    - [x] Refine `clean_mask.py` with normal constraints and better wall filtering
    - [x] Re-verify on new images

# Exhibition Hall Occupancy Map

- [x] Design the occupancy pipeline
    - [x] Update `implementation_plan.md`
- [x] Implement Point Cloud alignment
    - [x] Reuse/Refine RANSAC floor fit to get rotation matrix (align floor normal to Y-axis)
    - [x] Implement `align_points_to_floor` function
- [x] Implement Occupancy Grid generation
    - [x] Create `create_costmap_from_image.py` (originally `create_occupancy_grid.py`)
    - [x] Project aligned points to 2D grid (XZ plane)
    - [x] Compute occupancy (obstacles) vs free space (floor)
    - [x] Verify using the uploaded image and user feedback (auto-scaling, schematic style, thresholds)
- [ ] Improve occupancy map quality (User feedback: "not good")
    - [ ] Investigate depth distortion issues
    - [ ] Explore alternative alignment methods
