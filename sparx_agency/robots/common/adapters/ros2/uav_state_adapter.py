from typing import Optional
import numpy as np
from sparx_agency.core.common.spatial_math import euler_to_rot_zyx
from sparx_agency.core.common.types.perception import PoseSE3
from fcu_driver_interfaces.msg import UAVState

class UAVStateToPoseAdapter:
    def __init__(self):
        self._latest_pose: Optional[PoseSE3] = None

    def update(self, msg: UAVState):
        # Convert raw UAV ROS message to standard PoseSE3
        R = euler_to_rot_zyx(float(msg.roll), float(msg.pitch), float(msg.azimuth))
        t = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        self._latest_pose = PoseSE3(R=R, t=t)

    def get_pose(self) -> Optional[PoseSE3]:
        return self._latest_pose