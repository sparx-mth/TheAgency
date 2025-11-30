# raw servo values for direct actuator controls
#
# about groups, mixing and channels:
# https://dev.px4.io/v1.10/en/concept/mixing.html

# constant for mixer group
uint8 PX4_MIX_FLIGHT_CONTROL = 0
uint8 PX4_MIX_FLIGHT_CONTROL_VTOL_ALT = 1
uint8 PX4_MIX_GIMBAL = 2
uint8 PX4_MIX_MANUAL_PASSTHROUGH = 3
#uint8 PX4_MIX_FC_MC_VIRT = 4
#uint8 PX4_MIX_FC_FW_VIRT = 5
uint8 PX4_MIX_FIRST_PAYLOAD = 6

std_msgs/Header header
uint8 group_mix
float32[8] controls