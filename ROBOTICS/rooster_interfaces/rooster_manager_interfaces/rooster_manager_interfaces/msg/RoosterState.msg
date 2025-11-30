std_msgs/Header header

uint16 runtime

uint8 speed_mode
uint8 SPEED_MODE_TURTLE=0
uint8 SPEED_MODE_DEER=1


#UAV State
bool valid_position

float32 ranger


float32 roll
float32 pitch
float32 azimuth

uint8 flight_mode
uint8 FLIGHT_MODE_NONE=0
uint8 FLIGHT_MODE_GROUND_ROLL=1
uint8 FLIGHT_MODE_MANUAL=2
uint8 FLIGHT_MODE_POSITION=3
uint8 FLIGHT_MODE_ALTITUDE=4
uint8 FLIGHT_MODE_ACRO=5
uint8 FLIGHT_MODE_STABILIZED=6
uint8 FLIGHT_MODE_TAKEOFF=7
uint8 FLIGHT_MODE_LAND=8

bool armed

#uav_state_valid - taken from UAVState.connected 
bool uav_state_valid

bool airborne

bool armable


#Battery
float32 voltage
float32 current
float32 percentage #[0-1]

#is_fcu_connected - when the first message received from the fcu - Battery/Ranger etc
bool is_fcu_connected

#is_ready - when the the manager finished Boot state machine
bool is_ready

#ack for reboot commnad from keep_alive msg
bool got_reboot_command

uint8 MANUAL_CONTROL = 0
uint8 AUTO_CONTROL_SET = 1
uint8 AUTO_CONTROL_ACTIVE = 2
uint8 control_mode


