#!/usr/bin/env bash
# Full MP3D v1 val (2,195 episodes). Preview by default; --run is for the workstation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
exec python3 -B -m sparx_agency.tasks.planning.objnav_benchmark_runtime.workstation_4090 mp3d "$@"
