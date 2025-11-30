std_msgs/Header header

uint8 num_cameras # number of available cameras
video_handler_interfaces/VideoStreamInfo[2] stream_info # data per camera

uint32 free_space # kBytes

uint8 OP_IDLE=0
uint8 OP_DOWNLOAD=1
uint8 OP_CLEANUP=2
uint8 OP_MISSION=3
uint8 operation
uint8 progress # disk operations like Download or Clenaup progress


