#!/bin/bash
# Make FALCON say WHERE it could not plan to.
#
# Upstream's planning failure is a bare line:
#
#   [ExplorationManager] planTrajToView: No path to next viewpoint using default A*
#
# which is the least useful possible version of that sentence. Every question it
# raises -- is the viewpoint inside the known free space? how far away is it? is
# the aircraft itself somewhere sane? -- needs the two positions, and neither is
# printed anywhere. Debugging an exploration that will not start therefore means
# reading the C++ rather than the log.
#
# This adds the aircraft's position, the viewpoint's, and the distance between
# them to both messages. It changes no behaviour.
set -e

MANAGER=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_manager.cpp
[ -f "$MANAGER" ] || { echo "exploration_manager.cpp not found at $MANAGER"; exit 1; }

python3 - "$MANAGER" <<'PYEOF'
import re
import sys

path = sys.argv[1]
source = open(path).read()

ARGS = ('pos[0], pos[1], pos[2], next_pos_local[0], next_pos_local[1], '
        'next_pos_local[2], (pos - next_pos_local).norm()')
DETAIL = (' from (%.2f, %.2f, %.2f) to viewpoint (%.2f, %.2f, %.2f), '
          '%.2f m apart')

replacements = 0
for profile in ("default", "coarse"):
    old = ('"[ExplorationManager] planTrajToView: No path to next viewpoint '
           'using %s A*"' % profile)
    new = ('"[ExplorationManager] planTrajToView: No path%s using %s A*", %s'
           % (DETAIL, profile, ARGS))
    if old not in source:
        print("WARNING: could not find the %s A* failure message; upstream may "
              "have reworded it" % profile)
        continue
    source = source.replace(old, new)
    replacements += 1

open(path, "w").write(source)
print("patched %d planning-failure messages to name the positions" % replacements)
PYEOF
