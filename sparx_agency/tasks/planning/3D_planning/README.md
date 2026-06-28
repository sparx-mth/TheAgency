# Gibson Tiny Dataset – Quick Start

This README explains how to download the **Gibson Tiny** dataset, how scenes are structured,
and how the interactive 3D RRT* planning demo is organized.

The goal is to work with a **complete house / apartment**, not a single room.

---

## 1. Download Gibson Tiny

Gibson Tiny contains multiple **full indoor buildings** (houses / apartments).

```bash
mkdir -p gibson
cd gibson
```

### Download
```bash
wget https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz
````

(or)

```bash
curl -L https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz -o gibson_tiny.tar.gz
```

### Extract

```bash
mkdir -p extracted
tar -xzf gibson_tiny.tar.gz -C extracted
```

After extraction:

```
extracted/gibson_tiny/
├── Benevolence/
├── Shelbyville/
├── Noxapater/
├── ...
```

Each folder represents **one complete building**.

---

## 2. Scene Files

Inside each scene directory (example: `Benevolence/`):

```
Benevolence/
├── mesh.obj
├── mesh_z_up.obj
├── textures/
```

### mesh_z_up.obj (IMPORTANT)

* Same geometry as `mesh.obj`
* Rotated so **Z is the vertical axis**
* Always use this file for point clouds, voxelization, and mapping

---

## 3. Generated Geometry Files

After processing a scene, two main geometry artifacts are commonly produced.

### 1️⃣ `<scene>_pointcloud.ply`

Example:

```
Benevolence_pointcloud.ply
```

* Point cloud sampled from the house mesh
* Points lie on walls, floors, ceilings, and furniture
* Represents **surface geometry only**
* Does NOT encode free vs occupied space

Used for:

* Geometry inspection
* Mapping / SLAM input
* Voxelization

---

### 2️⃣ `<scene>_voxel_centers.ply`

Example:

```
Benevolence_voxel_centers.ply
```

* Centers of voxels created from the point cloud
* Each point represents an **occupied surface voxel**
* Not a full occupancy map
* Empty space is not explicitly represented

Used for:

* Understanding scene scale and structure
* Debugging voxel resolution
* Intermediate step before occupancy mapping

---

## 4. Code Structure (Interactive RRT* Planner)

The interactive demo is split into small files, each with a **single responsibility**.

```
3D_planning/
├── main.py
├── logging_utils.py
├── tube.py
├── voxelmap.py
├── gibson_io.py
├── interaction.py
├── final_window.py
├── benchmark_bitstar_standalone.py   # Run benchmarks
├── analyze_benchmark.py              # Statistical analysis & plots
├── visualize_paths.py                # 3D path visualization
```

### main.py

* Entry point of the demo
* Loads the scene and point cloud
* Builds the voxel map and collision model
* Handles START / GOAL selection flow
* Launches the final planning window

---

### logging_utils.py

* Simple logging helpers:

  * `pinfo`, `pok`, `pwarn`, `perr`
* Keeps all console output consistent and readable

---

### tube.py

* Generates a **thick 3D tube** from a polyline path
* Used to visualize the planned RRT* path reliably
* Avoids platform-dependent line-width issues in Open3D

---

### voxelmap.py

* Builds a voxel-based collision map from a point cloud
* Inflates obstacles to handle thin walls
* Integrates mesh raycasting for:

  * Distance-to-surface checks
  * Inside/outside mesh validation
* Provides `is_free()` and clearance queries for the planner

---

### gibson_io.py

* Scene I/O utilities
* Loads Gibson meshes (`mesh_z_up.obj`)
* Samples point clouds from the mesh surface

---

### interaction.py

* Interactive UI for user input
* Handles:

  * Point picking (Shift + Click)
  * Keyboard-based 3D adjustment (WASD / arrows)
* Used for precise START and GOAL placement

---

### final_window.py

* Final visualization and planning window
* Triggers OMPL RRT* planning
* Draws the resulting path as a thick black tube
* Allows stepping along the path with a movable marker

---

## 5. Representation Summary

| Representation  | Meaning                                |
| --------------- | -------------------------------------- |
| Mesh            | Exact surface geometry                 |
| Point cloud     | Sampled surface points                 |
| Surface voxels  | Discretized surface geometry           |
| Occupancy logic | Collision + clearance checks (runtime) |

---


## 6. Installation & Running the Demo

### Install Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
````

(Optional but recommended) Using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Run the Interactive Planner

From the `3D_planning` directory:

```bash
python3 main.py
```

(Optional) Select planner type:

```bash
python3 main.py --planner bitstar
python3 main.py --planner informed_rrtstar
```

Here's the section to add at the end of the README:


---

## 7. BIT* Benchmarking & Analysis

A set of scripts for benchmarking BIT* planner performance and analyzing path optimization over time.

### Scripts Overview


---

### benchmark_bitstar_standalone.py

Runs BIT* planning on multiple start-goal pairs and tracks how paths improve over time.

**What it does:**
* Samples random valid points from Floor 1 (start) and Floor 3 (goal)
* Runs BIT* with iterative solving to capture intermediate solutions
* Records time-to-first-solution, path length improvements, and all waypoints
* Saves results to JSON for later analysis

**Usage:**
```bash
# Default: 3 pairs, 30s timeout
python3 benchmark_bitstar_standalone.py

# Custom configuration
python3 benchmark_bitstar_standalone.py --num-pairs 100 --timeout 60

# All options
python3 benchmark_bitstar_standalone.py --num-pairs 50 --timeout 30 --seed 42 --poll-interval 0.1
```

**Output:** `results/bitstar_benchmark_YYYYMMDD_HHMMSS.json`

---

### analyze_benchmark.py

Generates statistics and plots from benchmark results.

**What it shows:**
* Success rate and timing statistics
* Path length distributions (first vs final solution)
* Improvement over time curves
* Path efficiency (path length / euclidean distance)

**Usage:**
```bash
# Analyze latest results
python3 analyze_benchmark.py

# Save plots to files
python3 analyze_benchmark.py --save-plots

# Analyze specific pair in detail
python3 analyze_benchmark.py --pair-id 42
```

---

### visualize_paths.py

Visualizes planned paths in the 3D Gibson environment using Open3D.

**Usage:**
```bash
# View first 5 paths from latest results
python3 visualize_paths.py

# View all paths
python3 visualize_paths.py --all

# View specific pair with evolution (all intermediate solutions)
python3 visualize_paths.py --pair-id 42

# Show first vs final solution comparison
python3 visualize_paths.py --show-first

# Save image instead of interactive view
python3 visualize_paths.py --save output.png
```

**Controls:**
* Mouse drag: Rotate
* Scroll: Zoom
* Shift + drag: Pan
* Q: Quit

---

### Research Goals

The benchmark suite is designed to answer key questions about BIT* anytime behavior:

| Question | Metric |
|----------|--------|
| How fast is the first solution? | `time_to_first_solution_s` |
| How much does the path improve? | First vs final path length |
| When do most improvements happen? | Improvement-over-time curve |
| Is more time worth it? | Diminishing returns analysis |
| How efficient are the paths? | Path length / euclidean distance ratio |

**Expected findings:**
* First solution found within 1-5s for most pairs
* Rapid improvement in first few seconds, then diminishing returns
* 10-30% path length reduction over 30s of optimization
* Trade-off point where additional time yields minimal improvement

This data helps determine optimal timeout values for real-time applications.
```