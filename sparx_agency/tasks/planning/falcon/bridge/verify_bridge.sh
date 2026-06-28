#!/bin/bash
# ============================================================
# ros_bridge_docker/verify_bridge.sh
#
# Checks that the 3 FALCON-critical topics are actually
# flowing through the bridge (not just listed).
# ============================================================

PASS=0; FAIL=0
green() { echo -e "\033[32m  OK   $1\033[0m"; }
red()   { echo -e "\033[31m  FAIL $1\033[0m"; }

echo ""
echo "════════════════════════════════════════════════"
echo "  FALCON Bridge Verification"
echo "════════════════════════════════════════════════"

for c in roscore ros1_bridge; do
    if docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
        green "'${c}' running"; ((PASS++))
    else
        red "'${c}' NOT running"; ((FAIL++))
    fi
done

if [[ ${FAIL} -gt 0 ]]; then
    echo "  Start missing containers first."
    exit 1
fi

check_hz() {
    local topic="$1"
    local hz_output
    hz_output=$(docker exec ros1_bridge bash -c \
        "source /opt/ros/noetic/setup.bash && \
         timeout 5 rostopic hz ${topic} 2>&1 | tail -3")
    if echo "${hz_output}" | grep -q "no new messages"; then
        red "${topic} — 0 Hz"
        ((FAIL++))
    elif echo "${hz_output}" | grep -qE "average rate:"; then
        local rate
        rate=$(echo "${hz_output}" | grep "average rate:" | tail -1 | awk '{print $3}')
        green "${topic} — ${rate} Hz"
        ((PASS++))
    else
        red "${topic} — unreachable"
        ((FAIL++))
    fi
}

echo ""
echo "── Messages flowing through bridge ──"
check_hz /simple_drone/gt_pose
check_hz /simple_drone/front_depth/depth/camera_info
check_hz /simple_drone/front_depth/depth/image_raw

echo ""
echo "════════════════════════════════════════════════"
echo "  ${PASS} passed, ${FAIL} failed"
if [[ ${FAIL} -eq 0 ]]; then
    echo -e "\033[32m  Bridge healthy — FALCON should build a map\033[0m"
fi
echo "════════════════════════════════════════════════"
exit ${FAIL}