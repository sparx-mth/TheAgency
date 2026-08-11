#!/bin/bash
# ============================================================
# entrypoint.sh — source ROS + the vendor interfaces, export PYTHONPATH for
# the repo, print what the model registry can already see, then exec.
#
# Does NOT set ROS_DOMAIN_ID / RMW_IMPLEMENTATION / CYCLONEDDS_URI -- those
# come from devcontainer.json's containerEnv, exactly like the existing
# run_*.sh wrappers set them on the host. Keeping them out of this script
# means one entrypoint works whether you're pointed at Sphera (domain 9) or
# anything else, without editing the image.
# ============================================================
set -euo pipefail

source /opt/ros/"${ROS_DISTRO:-humble}"/setup.bash
if [ -f /opt/rooster_ws/install/setup.bash ]; then
  source /opt/rooster_ws/install/setup.bash
fi

export PYTHONPATH="${PYTHONPATH:-}:/home/${USER}/GIT/TheAgency"

echo "[entrypoint] ROS_DISTRO=${ROS_DISTRO:-humble} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} " \
     "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>} CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"

if python3 -m sparx_agency.tasks.common.model_registry.cli path \
    --model da3_metric_large --role depth_only --precision fp16 --resolution 546x364 \
    --no-download > /tmp/.model_registry_check 2>&1; then
  echo "[entrypoint] model registry: $(tail -1 /tmp/.model_registry_check)"
else
  echo "[entrypoint] model registry: DA3 engine not found yet (warning, not fatal) -- " \
       "$(tail -1 /tmp/.model_registry_check 2>/dev/null || echo '(no detail)')"
fi

exec "$@"
