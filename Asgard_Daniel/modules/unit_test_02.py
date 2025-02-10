import time
from bridge_test_asgard import bridge_test
import numpy as np

def main():
    bridge_obj = bridge_test()
    speed = 10

    bridge_obj.takeoff(height=8.0, speed=speed)
    t1 = time.time()

    # bridge_obj.rotateDroneViewPitch(90)
    time.sleep(2)

    bridge_obj.rotateDroneYaw(90)
    time.sleep(3.5)

    bridge_obj.rotateDroneYaw(45)
    time.sleep(3.5)

    bridge_obj.rotateDroneYaw(-45)
    time.sleep(3.5)

    bridge_obj.go_to_direction_relative(direction='RIGHT', speed=speed, duration=4)
    time.sleep(3.5)

    bridge_obj.go_to_direction_relative(direction='FORWARD', speed=speed, duration=4)
    time.sleep(3.5)

    bridge_obj.go_to_direction_relative(direction='BACKWARD', speed=speed, duration=4)
    time.sleep(3.5)

    bridge_obj.go_to_direction_relative(direction='LEFT', speed=speed, duration=4)
    time.sleep(3.5)

    bridge_obj.go_to_direction_relative(direction='RIGHT', speed=speed, duration=4)
    time.sleep(3.5)

    # bridge_obj.land()
    t2 = time.time()
    print(f'Execution time: {np.round(t2-t1, 2)} seconds')

if __name__ == '__main__':
    main()
