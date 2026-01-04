#!/usr/bin/env python3

from sphera_common_interfaces.srv import SpheraSet
import rclpy
from rclpy.node import Node
import getpass
import json
import os
import random

FetchLengthOptions = [50, 100, 300, 500, 750, 1000, 1500]

WeatherTypes = [
    "light rain",
    "rain",
    "thunderstorm",
    "light snow",
    "snow",
    "blizzard",
    "dust storm",
    "dust",
    "foggy",
    "clear skies",
    "partly cloudy",
    "cloudy",
    "overcast",
]


class OceanWeatherDemo(Node):

    def __init__(self):
        super().__init__("ocean_weather_demo")
        user = "user1" # should be SPHERA host user, not user in docker.

        self.cli = self.create_client(SpheraSet, "/sphera/" + user + "/set")

        self.timer_ = self.create_timer(10.0, self.timer_callback)
        self.get_logger().info(
            "Changing weather every 10 seconds with random parameters"
        )

    def timer_callback(self):
        if not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Spawn service [{self.cli.srv_name}] not available, waiting...")

        time_of_day = self.get_random_time_of_day()
        weather = random.choice(list(WeatherTypes))
        wind_speed = random.randint(1, 35)

        self.get_logger().info("resseting weather conditions:")
        self.get_logger().info(f"  time_of_day:    {time_of_day}")
        self.get_logger().info(f"  weather:        {weather}")
        self.get_logger().info(f"  wind_speed:     {wind_speed}")
        self.send_request(
            time_of_day, weather, wind_speed
        )

    def send_request(
        self,
        time_of_day: str,
        weather: str,
        wind_speed: int,
    ):
        req = SpheraSet.Request()
        data = {
            "environment": {
                "time_of_day": time_of_day, # <hour>:<minute>:<second>
                "weather": weather,
                "wind_speed": wind_speed, 
            }
        }
        req.data = json.dumps(data)
        self.cli.call_async(req)

    def get_random_time_of_day(self):
        hours = random.randint(6, 18)  # leaving out the darkest hours
        minutes = random.randint(0, 59)
        seconds = random.randint(0, 59)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def main(args=None):
    rclpy.init(args=args)

    demo = OceanWeatherDemo()
    rclpy.spin(demo)

    demo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
