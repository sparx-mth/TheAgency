#!/usr/bin/env bash
# Install Pegasus Simulator (patched for Isaac Sim 6.0.1) and build PX4-Autopilot
# SITL, inside the isaac-sim container. See robots/PEGASUS/README.md for context
# on why each step exists.
#
# Usage (run inside the isaac-sim container):
#   robots/PEGASUS/setup/install.sh /tmp/dev
set -euo pipefail

DEV_ROOT="${1:?usage: install.sh <dev-root-dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEGASUS_COMMIT="e13dc659686b09fffb05275988b70e5dc66983da"  # main, 2026-07-24
PX4_TAG="v1.14.3"  # the version Pegasus Simulator was developed/tested against

mkdir -p "$DEV_ROOT"
cd "$DEV_ROOT"

# --- Pegasus Simulator, pinned + patched -----------------------------------
if [ ! -d PegasusSimulator ]; then
    git clone --quiet https://github.com/PegasusSimulator/PegasusSimulator.git
fi
git -C PegasusSimulator checkout --quiet "$PEGASUS_COMMIT"
git -C PegasusSimulator apply --check "$SCRIPT_DIR/pegasus_isaac6_compat.patch" 2>/dev/null \
    || echo "note: patch already applied or does not apply cleanly to $PEGASUS_COMMIT -- check manually"
git -C PegasusSimulator apply "$SCRIPT_DIR/pegasus_isaac6_compat.patch" 2>/dev/null || true

# Drop into Isaac Sim's extsUser/ so it is discovered locally at startup
# (dynamically adding an ext-folder at runtime falls back to the online
# extension registry and fails -- see robots/PEGASUS/README.md).
ln -sfn "$DEV_ROOT/PegasusSimulator/extensions/pegasus.simulator" /isaac-sim/extsUser/pegasus.simulator

# --- PX4-Autopilot SITL ------------------------------------------------------
if [ ! -d PX4-Autopilot ]; then
    git clone --quiet --recursive --branch "$PX4_TAG" --depth 1 https://github.com/PX4/PX4-Autopilot.git
fi

pip3 install --break-system-packages --quiet -r PX4-Autopilot/Tools/setup/requirements.txt
pip3 install --break-system-packages --quiet 'empy==3.3.4'  # >=3.3 resolves to 4.x, which breaks the PX4 build

# GCC 13 (Ubuntu 24.04, this container's toolchain) is stricter than the
# Ubuntu 20.04/22.04 + GCC 9-11 PX4 v1.14.3 was built against. Two known,
# narrowly-scoped fixes:
if ! grep -q '#include <cstdint>' PX4-Autopilot/platforms/posix/src/px4/common/px4_daemon/pxh.cpp; then
    sed -i '/#include <string>/i #include <cstdint>' \
        PX4-Autopilot/platforms/posix/src/px4/common/px4_daemon/pxh.cpp
fi
sed -i 's/^\t\t-Warray-bounds$/\t\t-Wno-error=array-bounds\t# GCC 13 false positive on matrix::Matrix 1x1 specialization/' \
    PX4-Autopilot/cmake/px4_add_common_flags.cmake

cd PX4-Autopilot
make px4_sitl_default none
echo "PX4 SITL built: $DEV_ROOT/PX4-Autopilot/build/px4_sitl_default/bin/px4"
