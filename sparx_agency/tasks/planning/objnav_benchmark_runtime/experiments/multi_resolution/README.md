# Multi-Resolution FALCON

`run_coarse_falcon.sh` launches one bounded HM3D ObjectNav scene with FALCON,
1.0 m occupancy cells, half the default angular viewpoint samples, and at most
two retained viewpoints. The run locks exploration to the episode's starting
floor using navmesh stair connectors. It requires the existing HM3D episode and
scene paths plus a Habitat-Sim-compatible Python environment.

```bash
HABITAT_PYTHON=/path/to/habitat/python \
HM3D_EPISODES_DIR=/path/to/objectnav/hm3d/v2/val \
HM3D_SCENES_DIR=/path/to/data/scene_datasets \
HM3D_SCENE=00800-TEEsavR23oF \
bash sparx_agency/tasks/planning/objnav_benchmark_runtime/experiments/multi_resolution/run_coarse_falcon.sh \
  --preflight
```

The 1.0 m profile is an experimental coarse pass, not a fine-resolution map or
a claim of doorway-scale clearance. Compare it with a 0.1 m run using the same
scene, episode, seed, and target; keep the target's ground-truth pose evaluator-
only. HM3D's published ObjectNav vocabulary does not contain pen or scissors,
so this launcher exercises the coarse planner on existing episode goals. A
randomized small-object fixture and ground-truth visibility score are a separate
experiment adapter, not inferred from HM3D ObjectNav labels.