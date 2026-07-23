import time
import json
import random
from sparx_agency.robots.ASGARD.asgard_controller import Controller
from sparx_agency.robots.ASGARD.geometric import Geometric
import numpy as np
import ast


class AsgardPlanner:
    """
    Researcher wraps all agent abilities and behaviors.
    """
    def __init__(self, config: dict, drone, agent_location, t_start, ip, port):
        self.config = config
        self.drone = drone
        self.agent_location = {'x': agent_location[0], 'y': agent_location[1], 'z': agent_location[2]}
        self.t_start = t_start
        self.ip = ip
        self.port = port

        self.flying_height = self.config['flying_height']
        self.speed = self.config['speed']
        self.liftoff = self.flying_height / self.speed
        self.scan_degrees = self.config['scan_degrees']
        self.threshold = self.config['threshold']
        self.sleep_factor = self.config['sleep_factor']

        self.controller = Controller(self.config, self.drone, self.ip, self.port)
        self.state = 'random_scan'
        self.RandomFlight = True

        self.drone_data_init = json.loads(self.controller.getDroneData())
        self.my_loc_init = np.array(ast.literal_eval(self.drone_data_init['position']))
        self.ground_height = self.my_loc_init[1]

        self.random_speed = None
        self.drone_data = None
        self.pest_list = None
        self.drones_list = None
        self.my_loc = None
        self.agents_locs = None

    def go_to_starting_location(self):
        distance = 1000
        while distance > 2:
            self.controller.moveToPosition(self.agent_location, 10)
            self.get_drone_data()
            drone_pos = self.controller.extractPosition(self.drone_data['position'])
            distance = Geometric.calculate_distance(drone_pos, self.agent_location)
        print(self.drone['droneID'] + ' Arrived to starting location!')

    def loop(self):
        """
        Infinite agent loop.
        """
        self.controller.LiftOffAndStop(self.speed, self.liftoff / self.config['run_speed_factor'])
        self.controller.setPitchRotation(90)
        self.go_to_starting_location()

        while time.time() < self.t_start:
            continue

        print(self.drone['droneID'] + ' starting loop!')

        while self.RandomFlight:
            try:
                if self.state != 'finish_mission':
                    if self.get_drone_data():
                        # self.random_speed = random.randint(2, 10)  # KM/S
                        self.random_speed = self.config['drones_speed']  # KM/S
                        self.detect()
                        if len(self.pest_list) > 0:  # Target Found (find and destroy)
                            print(self.drone['droneID'] + ' detect target')
                            self.awareness_smart() if self.config['smart'] else self.awareness_default()

                match self.state:

                    case 'random_scan':
                        self.state_random_scan()

                    case 'attack':
                        self.state_attack()

                    case 'finish_mission':
                        print('Mission finished!')
                        self.RandomFlight = False
            except Exception as e:
                print(e)
                self.state = 'finish_mission'

    def state_random_scan(self):
        """
        Researcher state - Perform random scan to detect pests.
        """
        random_direction = self.controller.get_random_direction()
        self.controller.rotateDrone(random_direction)
        time.sleep(2 / self.controller.time_step)
        self.controller.moveDrone(random_direction, self.random_speed)
        self.controller.setYawRotation(self.scan_degrees)
        self.scan_degrees = self.scan_degrees * (-1)
        time.sleep(5 / self.controller.time_step)  # 2 seconds delay for random movement

    def state_attack(self):
        """
        Researcher state - Perform attack.
        """
        while True:
            if self.targetPestsAndDestroy():
                print('Target pest and destroy')
                self.state = 'finish_mission'
                break
            if self.get_drone_data():
                self.detect()
                if len(self.pest_list) == 0:
                    break
            else:
                break

    def get_drone_data(self):
        try:
            self.drone_data = json.loads(self.controller.getDroneData())
        except:
            self.drone_data = None
            print('No drone data')
            self.state = 'finish_mission'
            return False
        if not self.drone_data['isActive']:
            self.state = 'finish_mission'
            print('Drone is not active')
            return False
        return True

    def detect(self):
        if self.config['network']:
            self.pest_list = self.controller.extract_pests_from_LOS_using_network(self.drone_data)
        else:
            self.pest_list, self.drones_list = self.controller.extract_pests_from_LOS(self.drone_data)

    def back_to_scan(self):
        self.my_loc = np.array(ast.literal_eval(self.drone_data['position']))
        liftoff = (self.ground_height + self.flying_height - self.my_loc[1]) / self.speed
        if liftoff > 0:
            self.controller.LiftOffAndStop(self.speed, liftoff)
        self.state = 'random_scan'

    def check_there_are_pests(self):
        if self.get_drone_data():
            self.detect()
            if len(self.pest_list) == 0:
                print(self.drone['droneID'] + ' lost the target or may it got destroyed')
                self.back_to_scan()
                return False
            return True
        return False

    def awareness_default(self):
        self.state = 'attack'

    def awareness_smart(self):
        self.get_agents_locations()
        while True:
            if not self.check_there_are_pests():
                break

            self.get_agents_locations()

            if self.check_drones_down():
                print(self.drone['droneID'] + ' keep in random_scan')
                self.back_to_scan()
                break
            else:
                print(self.drone['droneID'] + ' start to go to target')
                self.go_to_target()

                if not self.check_there_are_pests():
                    break

                self.get_agents_locations()

                if self.check_drones_down():
                    print(self.drone['droneID'] + ' keep in random_scan')
                    self.back_to_scan()
                    break
                else:
                    if self.check_another_drones_on_target():
                        print(self.drone['droneID'] + ' detect another drones on target')
                        time.sleep((self.sleep_factor / self.config['run_speed_factor']) * random.randint(1, 5))
                    else:
                        print(self.drone['droneID'] + ' keep to attack')
                        self.state = 'attack'
                        break

    def get_agents_locations(self):
        self.my_loc = np.array(ast.literal_eval(self.drone_data['position']))
        if self.drones_list:
            self.agents_locs = np.array([ast.literal_eval(d['position']) for d in self.drones_list])
        else:
            self.agents_locs = None

    def check_drones_down(self):
        if self.agents_locs is None:
            return False
        height_differences = Geometric.calculate_height_differences(self.my_loc, self.agents_locs)
        return any(difference > self.threshold for difference in height_differences)

    def check_another_drones_on_target(self):
        if self.agents_locs is None:
            return False
        height_differences = Geometric.calculate_height_differences(self.my_loc, self.agents_locs)
        return any(abs(difference) < self.threshold for difference in height_differences)

    def go_to_target(self):
        t_level_down = self.config['t_go_to_target'] / self.config['run_speed_factor']
        drone_pos = self.controller.extractPosition(self.drone_data['position'])
        target_pest_pos = None
        target_pest = None
        min_distance = 1000
        for pest in self.pest_list:
            pest_post = self.controller.extractPosition(pest['position'])
            distance = Geometric.calculate_distance(drone_pos, pest_post)
            if min_distance > distance:
                min_distance = distance
                target_pest_pos = pest_post
                target_pest = pest
        self.controller.StartToMoveToPosition(target_pest_pos, self.random_speed, t_level_down)

    def targetPestsAndDestroy(self):
        drone_pos = self.controller.extractPosition(self.drone_data['position'])
        target_pest_pos = None
        target_pest = None
        min_distance = 1000
        for pest in self.pest_list:
            pest_post = self.controller.extractPosition(pest['position'])
            distance = Geometric.calculate_distance(drone_pos, pest_post)
            if min_distance > distance:
                min_distance = distance
                target_pest_pos = pest_post
                target_pest = pest
            if distance < self.drone['exterminationRadius']:
                # print("[targetPestAndDestroy] drone", self.drone['droneID'], "exterminating", "pest", target_pest["ID"])
                self.controller.exterminateDrone()
                return True
        if self.config['network']:
            self.transmitPestHandle(target_pest)
        if self.controller.moveToPosition(target_pest_pos, self.random_speed) == 'Failed':
            return True
        return False

    def transmitPestHandle(self, pest):
        lv2_data_package = {"state": pest["ID"]}
        self.controller.transmitPackage(2, lv2_data_package)