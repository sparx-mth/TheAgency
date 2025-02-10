import time
from bridge_test_asgard import bridge_test


def main():
    bridge_obj = bridge_test()

    bridge_obj.takeoff(height=50.0, speed=10)
    bridge_obj.rotateDronePitch(90)
    time.sleep(2)
    bridge_obj.rotateDroneYaw(90)
    time.sleep(6)
    bridge_obj.rotateDroneYaw(45)
    time.sleep(6)
    bridge_obj.rotateDroneYaw(-45)
    time.sleep(6)
    bridge_obj.go_to_direction_relative(direction='RIGHT', speed=15, duration=4)

    bridge_obj.go_to_direction_relative(direction='FORWARD', speed=15, duration=4)
    bridge_obj.go_to_direction_relative(direction='BACKWARD', speed=15, duration=4)
    bridge_obj.hover()
    bridge_obj.go_to_direction_relative(direction='LEFT', speed=15, duration=4)
    bridge_obj.go_to_direction_relative(direction='RIGHT', speed=15, duration=4)
    bridge_obj.hover()


if __name__ == '__main__':
    main()
