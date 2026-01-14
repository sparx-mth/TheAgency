# Gibson Tiny Dataset – Quick Start

This README explains how to download the **Gibson Tiny** dataset and what the two main generated files represent.

The goal is to work with a **complete house / apartment**, not a single room.

---

## 1. Download Gibson Tiny

Gibson Tiny contains multiple **full indoor buildings** (houses / apartments).

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

## 3. Generated Files

After processing a scene, two main files are produced.

### 1️⃣ `<scene>_pointcloud.ply`

Example:

```
Benevolence_pointcloud.ply
```

* A **point cloud sampled from the house mesh**
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
* **Not a full occupancy map**
* Empty space is not explicitly represented

Used for:

* Understanding scene scale and structure
* Debugging voxel size
* Intermediate step before occupancy mapping

---

## 4. Concept Summary

| Representation   | Meaning                                  |
| ---------------- | ---------------------------------------- |
| Mesh             | Exact surface geometry                   |
| Point cloud      | Sampled surface points                   |
| Surface voxels   | Discretized surface geometry             |
| Occupancy voxels | Free / occupied / unknown (not included) |

---
