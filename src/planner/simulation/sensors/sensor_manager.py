class SensorManager:
    def __init__(self):
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def sense_all(self, pos, facing, env):
        all_observations = []
        for sensor in self.sensors:
            observations = sensor.sense(pos, facing, env)
            all_observations.extend(observations)
        return all_observations
