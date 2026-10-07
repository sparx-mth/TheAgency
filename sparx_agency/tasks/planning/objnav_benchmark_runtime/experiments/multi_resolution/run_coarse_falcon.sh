#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
python_bin="${HABITAT_PYTHON:-python3}"
episodes_dir="${HM3D_EPISODES_DIR:?Set HM3D_EPISODES_DIR to an HM3D ObjectNav split}"
scenes_dir="${HM3D_SCENES_DIR:?Set HM3D_SCENES_DIR to the Habitat scene datasets directory}"
scene="${HM3D_SCENE:?Set HM3D_SCENE to one scene folder or stem}"
episodes_per_scene="${HM3D_EPISODES_PER_SCENE:-1}"
version="${HM3D_VERSION:-v2}"
split="${HM3D_SPLIT:-val}"
output="${FALCON_OUTPUT:-${repo_root}/results/multi_resolution/coarse-1m}"
config="${repo_root}/sparx_agency/tasks/planning/objnav_benchmark_runtime/experiments/multi_resolution/coarse_falcon.json"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run \
  --version "${version}" \
  --split "${split}" \
  --episodes-dir "${episodes_dir}" \
  --scenes "${scene}" \
  --episodes-per-scene "${episodes_per_scene}" \
  --scenes-dir "${scenes_dir}" \
  --explorer falcon \
  --policy-config "${config}" \
  --output "${output}" \
  "$@"