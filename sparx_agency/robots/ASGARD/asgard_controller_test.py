import time
from sparx_agency.robots.ASGARD.asgard_controller import AsgardController


def main():
    controller = AsgardController()

    controller.takeoff(height=50.0, speed=10)
    controller.rotateDronePitch(90)
    time.sleep(2)
    controller.rotateDroneYaw(90)
    time.sleep(6)
    controller.rotateDroneYaw(45)
    time.sleep(6)
    controller.rotateDroneYaw(-45)
    time.sleep(6)
    controller.go_to_direction_relative(direction='RIGHT', speed=15, duration=4)

    controller.go_to_direction_relative(direction='FORWARD', speed=15, duration=4)
    controller.go_to_direction_relative(direction='BACKWARD', speed=15, duration=4)
    controller.hover()
    controller.go_to_direction_relative(direction='LEFT', speed=15, duration=4)
    controller.go_to_direction_relative(direction='RIGHT', speed=15, duration=4)
    controller.hover()


if __name__ == '__main__':
    main()