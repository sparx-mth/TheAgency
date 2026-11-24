#!/usr/bin/env bash
# Full HM3D v1 val (2,000 episodes). Preview by default; --run is for the workstation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
exec python3 -B -m sparx_agency.tasks.planning.objnav_benchmark_runtime.workstation_4090 hm3d "$@"
