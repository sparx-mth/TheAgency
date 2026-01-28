# Depth Masking Pipeline Walkthrough

I have implemented a robust depth-based object masking pipeline to prepare clean images for downstream tasks like OWL-ViT.

## Changes
- **Refined [`DepthMasker`](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/depth/clean_mask.py)**:
    - **RANSAC Floor Fitting**: Robustly identifies the floor plane with **orientation constraints** (must be horizontal).
    - **Wall Removal**: Explicitly detects and removes large vertical planes (walls) to isolate objects even in confined spaces.
    - **Refined Thresholds**: Improved object preservation by adjusting area and distance thresholds.
- **Test Script [`clean_mask_test.py`](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/depth/clean_mask_test.py)**:
    - Runs on your `Pictures` folder.
    - Generates a 2x3 grid visualization (now includes **Wall Mask**).
    - Saves results to `<appDataDir>/brain/<conversation-id>/test_results`.

## how to Run

To run the masking test on your images:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 sparx_agency/core/mapping/depth/clean_mask_test.py
```

## Results
The pipeline successfully processes the images. The output for each image includes:
1.  **Row 1**: Original RGB | Depth Map | Floor Mask (Red)
2.  **Row 2**: Wall Mask (Blue) | Object Mask (Green) | Clean RGB

I have saved the test results for the images in:
`file:///home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/test_results`

You should now see that:
- **Red** regions (Floor) are correctly identified even if the floor is tilted or partial.
- **Blue** regions (Walls) are masked out.
- **Green** regions (Objects) remain and are cleaner.

# Exhibition Occupancy Map (Costmap)

I have created a pipeline to generate a schematic occupancy map from a single high-angle image of the exhibition hall.

## Implementation
- **Script [`create_costmap_from_image.py`](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/costmap/create_costmap_from_image.py)**:
    - **Depth Estimation**: Uses `DepthAnythingV2` (Metric Indoor) to get a dense point cloud.
    - **World Alignment**: Uses RANSAC to calculate the floor plane and rotates the entire world so the floor is flat (Y=0) and walls are vertical. 
    - **Auto-Orientation**: Uses **PCA** on the obstacle points to automatically align the dominant room axis (aisles/walls) with the map grid.
    - **Scaling & Orientation**: Automatically scales to the visible area and ensures objects have positive height regardless of camera pitch.
    - **Projected Costmap**: Slices the aligned 3D world into a 2D grid.
        - **Green (Free)**: Floor area (0 to 0.5m height).
        - **Red (Occupied)**: Obstacles > 0.5m height.
        - **Schematic Processing**: Applies morphological dilation to connect scattered points into solid schematic stand walls.

## How to Run
```bash
python3 sparx_agency/core/mapping/costmap/create_costmap_from_image.py
```

## Result
The system generates a schematic-style map showing the layout of the stands. Notice how the map is now rotated to align the aisles vertically/horizontally.

````carousel
![Depth Map](/home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/costmap_results/depth.jpg)
<!-- slide -->
![Occupancy Map](/home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/costmap_results/costmap.png)
````
