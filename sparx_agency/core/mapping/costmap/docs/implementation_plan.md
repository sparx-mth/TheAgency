# Depth-Based Object Masking Implementation Plan

## Goal
Create a robust pipeline to generate depth from RGB (using Depth Anything), identify and mask out the floor (using RANSAC plane fitting or vertical gradient), and mask out distant walls/background. The result will be a "clean" image containing only relevant objects, suitable for input into a transformer model like OWL-ViT.

## User Review Required
> [!NOTE]
> I will implement a custom `numpy`-based RANSAC for floor detection since `sklearn`/`open3d` appear to be unavailable.
> I will also implement a "vertical gradient" heuristic as requested, but RANSAC is generally more robust for floor planes.

## Proposed Changes

### Core Mapping Depth
#### [NEW] [clean_mask.py](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/depth/clean_mask.py)
- Create a `DepthMasker` class.
- **Intrinsic Estimation**: Simple pinhole model approximation (FOV ~60 deg) if not provided.
- **Point Cloud Generation**: Vectorized `depth_to_points` function.
- **RANSAC Floor Fitting**:
    - Iteratively select 3 random points from the bottom portion of the image.
    - Compute plane equation $ax+by+cz+d=0$.
    - Count inliers (points within small distance to plane).
    - Select best plane.
- **Gradient Filter** (Alternative):
    - Compute vertical gradient of depth map.
    - Threshold simple continuous areas (floor) vs discontinuities (objects).
- **Mask Generation**:
    - `get_object_mask(depth, rgb)`: Returns binary mask of objects.
    - Logic: `(dist_to_floor > threshold) & (depth < max_dist)`.
    - Morphological cleanup (Open/Close) to remove noise.

#### [NEW] [clean_mask_test.py](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/depth/clean_mask_test.py)
- Script to run `DepthMasker` on user's test images (`/home/user1/Pictures/...`).
- Visualizes:
    - Original RGB
    - Depth Map
    - Floor Mask
    - Object Mask
    - Final "Clean" Image (objects on black background)

## Verification Plan

### Manual Verification
- Run `python core/mapping/depth/clean_mask_test.py`
- Inspect the output windows (or saved images) for the set of test images detailed in the user's request context.
- Verify that:
    - Floor is successfully removed.
    - Walls (background) are removed.
    - Objects (chairs, etc.) are preserved.
    - Edges are reasonably clean.
- Tune parameters (RANSAC iterations, distance threshold, height threshold) if necessary.

# Exhibition Occupancy Map Implementation Plan

## Goal
Generate a 2D occupancy/costmap of an exhibition hall from a single high-angle image. The map should distinguish between walkable floor areas (free space) and obstacles (walls, booths, furniture).

## Proposed Changes

### Core Mapping Occupancy
#### [NEW] [create_occupancy_grid.py](file:///home/user1/GIT/TheAgency/sparx_agency/core/mapping/occupancy/create_occupancy_grid.py)
- **Inputs**: RGB Image + Depth Model.
- **Step 1: Depth Estimation**: Use `DepthAnythingV2` to get dense depth.
- **Step 2: Point Cloud Generation**: Convert depth to 3D points $(x, y, z)$.
- **Step 3: Floor Alignment**:
    - Use `DepthMasker.fit_plane_ransac` to find the floor plane equation.
    - Compute the normal vector of the floor.
    - Compute a rotation matrix $R$ to align the floor normal with the target vertical axis (e.g., $Y$ or $Z$).
    - Apply $R$ to all points so the floor becomes horizontal (flat).
    - Translate points so the floor is at $Height = 0$.
- **Step 4: Grid Projection**:
    - Define grid resolution (e.g., 5cm per pixel).
    - Discretize aligned $(x, z)$ coordinates into grid cells.
    - **Free Space**: Cells containing points near height $0$ (floor).
    - **Occupied**: Cells containing points with height $> threshold$ (walls, objects).
    - **Unknown**: Cells with no points (occluded or out of view).
- **Step 5: Visualization**:
    - Output an image representing the map (e.g., Green=Free, Red=Occupied, Black=Unknown).

## Verification Plan
- Use the provided test image (`uploaded_image_...`).
- Run the script and save the generated occupancy map.
- Visually verify that the aisles and open areas are marked as free space, and the stand walls are marked as obstacles.
