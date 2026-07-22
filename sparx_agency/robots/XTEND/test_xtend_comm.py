# tasks / demo usage
import asyncio
from sparx_agency.robots.XTEND.adapters.xtend_robot_adapter import XtendRobotAdapter

async def main():
    robot = XtendRobotAdapter(host="192.0.0.15", port=8000, robot_uid="drn120ea1b0", frequency_hz=30.0)
    await robot.start()

    await robot.control.disarm()
    await asyncio.sleep(1.0)
    await robot.control.arm()
    await asyncio.sleep(1.0)
    await robot.control.takeoff()
    await asyncio.sleep(2.0)
    await robot.control.land()
    await asyncio.sleep(1.0)
    await robot.control.disarm()

    await robot.stop()

asyncio.run(main())
