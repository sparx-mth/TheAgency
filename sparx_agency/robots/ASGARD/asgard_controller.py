import requests
import json
import random
import sys
import re
import time
import numpy as np

# Initialize direction variables
directionUp = {"x": 0.0, "y": 1.0, "z": 0.0}
directionDown = {"x": 0.0, "y": -1.0, "z": 0.0}
directionForward = {"x": 0.0, "y": 0.0, "z": 1.0}
directionBackward = {"x": 0.0, "y": 0.0, "z": -1.0}
directionRightBackward = {"x": 1.0, "y": 0.0, "z": -1.0}
directionLeftBackward = {"x": -1.0, "y": 0.0, "z": -1.0}
directionRight = {"x": 1.0, "y": 0.0, "z": 0.0}
directionLeft = {"x": -1.0, "y": 0.0, "z": 0.0}
stop = {"x": 0.0, "y": 0.0, "z": 0.0}

directionList = [directionBackward, directionForward, directionRight, directionLeft]
#################################################

class AsgardController:
    def __init__(self, config: dict = None, drone: str = None, ip='http://127.0.0.1', port='8081'):
        self.config = config
        self.drone = drone
        self.ip = ip
        self.port = port
        try:
            self.droneid = self.drone['droneID']
        except:
            self.droneid = None
        self.simulation_port = None
        self.time_step = None
        self.url = None
        self.drone_id_list = []
        self.drone_list = []
        self.connect_simulation()
        self.start()
        self.droneid = self.drone_list[0]['droneID']

    def connect_simulation(self):
        if len(sys.argv) < 2:
            simulation_port = self.port
            time_step = 1
        else:
            simulation_port = sys.argv[1]
            time_step = int(sys.argv[2])
        self.time_step = time_step
        self.url = self.ip + ':' + simulation_port

    def start(self):
        connected = False
        while not connected:
            try:
                # Try to establish a connection with the port
                r = requests.get(self.url + '/')  ## Server Root Check
                if r.status_code == 200:
                    # If the connection is successful, enable the buttons
                    connected = True
                    print("mission planner connected")
                    self.connect_button_click()
            except:
                # If the connection fails, disable the buttons
                connected = False
            # time.sleep(10)## retrying every 10 sec

    def exterminateDrone(self):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid}
        response = requests.post(self.url + "/Extermination", data=json.dumps(data), headers=headers)
        print(response.text)
        # print(response.text, "DroneId", self.droneId)

    def connectDrone(self, droneId):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': droneId}
        response = requests.post(self.url + "/Connect", data=json.dumps(data), headers=headers)
        print(response.text)

    def getDroneParams(self, droneId):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': droneId}
        response = requests.post(self.url + "/DroneParamsByDroneID", data=json.dumps(data), headers=headers)
        return response.text

    def getDroneData(self):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid}
        response = requests.post(self.url + "/DroneDataByDroneID", data=json.dumps(data), headers=headers)
        return response.text

    def transmitPackage(self, level, data):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'Level': level, 'Data': data}
        response = requests.post(self.url + "/TransmitPackage", data=json.dumps(data), headers=headers)
        return response.text

    def moveDrone(self, direction, speed):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'Direction': direction, 'Speed': speed}
        response = requests.post(self.url + "/Move", data=json.dumps(data), headers=headers)
        return response.text

    def moveToPosition(self, targetPosition, speed):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'toPosition': targetPosition, 'Speed': speed}
        response = requests.post(self.url + "/MoveTo", data=json.dumps(data), headers=headers)
        return response.text

    def get_current_rotation(self):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid}
        response = requests.post(self.url + "/DroneDataByDroneID", data=json.dumps(data), headers=headers)
        json_response = response.json()
        rotation = json_response["rotation"]
        return rotation

    def rotateDrone(self, direction):
        current_rotation = self.get_current_rotation()
        current_rotation = float(current_rotation)
        new_rotation = 0

        if direction == directionForward:
            new_rotation = current_rotation
        elif direction == directionBackward:
            new_rotation = current_rotation + 180
        elif direction == directionRight:
            new_rotation = current_rotation - 90
        elif direction == directionLeft:
            new_rotation = current_rotation + 90
        elif direction == directionRightBackward:
            new_rotation = current_rotation + 135
        elif direction == directionLeftBackward:
            new_rotation = current_rotation + 45
        side = "RIGHT"
        headers = {'Content-type': 'application/json'}
        if new_rotation > 360:
            new_rotation = new_rotation - 360
            side = 'LEFT'

        data = {'DroneID': self.droneid, 'rotation': side, 'Degrees': new_rotation}
        response = requests.post(self.url + "/SetRotationByDroneID", data=json.dumps(data), headers=headers)

    def setDroneSpeed(self, speed):
        if speed == None or speed == '':
            return
        speed = float(speed)
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'Speed': speed}
        response = requests.post(self.url + "/SetSpeedByDroneID", data=json.dumps(data), headers=headers)

    def getAllDrones(self):
        headers = {'Content-type': 'application/json'}
        response = requests.get(self.url + "/GetDrones", headers=headers)
        return response.text

    def setPitchRotation(self, degree):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'Degrees': degree}
        response = requests.post(self.url + "/SetDronePitch", data=json.dumps(data), headers=headers)

    def setYawRotation(self, degree):
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'Degrees': degree}
        response = requests.post(self.url + "/SetDroneYaw", data=json.dumps(data), headers=headers)

    def connect_button_click(self):
        res = self.getAllDrones()
        json_data = json.loads(res)
        self.drone_id_list = json_data['DroneIDList']
        for droneid in self.drone_id_list:
            drone_data = json.loads(self.getDroneParams(droneid))
            self.drone_list.append(drone_data)
            self.connectDrone(droneid)

    @staticmethod
    def extract_pests_from_LOS_using_network(drone_data):
        target_list = drone_data['lineOfSightTargets']['Targets']
        drone_packages = drone_data["networkPackage"]["packages"]  ## pest that allready tranmited from another drone
        ignore_list = []
        for package in drone_packages:
            ignore_list.append(package["2"].split("State: ")[1])
        pest_list = []
        pest_types = ['Roaming', 'Static', 'Evasive']
        for target in target_list:
            if target['ID'] in ignore_list:
                continue
            if target['Type'] in pest_types:
                if target['Alive']:
                    pest_list.append(target)
        return pest_list

    def extract_pests_from_LOS(self, drone_data):
        target_list = drone_data['lineOfSightTargets']['Targets']
        pest_list = []
        drones_list = []
        pest_types = ['Roaming', 'Static', 'Evasive']
        for target in target_list:
            if target['ID'] == self.config['pest_id_to_ignore']:
                continue
            if target['Type'] in pest_types:
                if target['Alive']:
                    pest_list.append(target)
            else:
                if target['Alive']:
                    drones_list.append(target)
        return pest_list, drones_list

    @staticmethod
    def extractPosition(pos_string):
        # Extract numbers using regex
        numbers = re.findall(r'\d+\.\d+', pos_string)

        # Convert the extracted numbers to float
        numbers = [float(num) for num in numbers]
        position_vector = {"x": numbers[0], "y": numbers[1], "z": numbers[2]}

        return position_vector

    def LiftOffAndStop(self, speed, liftoff):
        self.moveDrone(directionUp, speed)
        time.sleep(liftoff)
        self.moveDrone(stop, speed)

    def StartToMoveToPosition(self, target_pest_pos, speed, t_level_down):
        self.moveToPosition(target_pest_pos, speed)
        time.sleep(t_level_down)
        self.moveDrone(stop, speed)

    @staticmethod
    def get_random_direction():
        return random.choice(directionList)

    def takeoff(self, height, speed):
        self.LiftOffAndStop(speed, height / speed)

    def go_to_direction_absolute(self, direction, speed, duration):
        if direction == 'FORWARD':
            self.moveDrone(directionForward, speed)
        elif direction == 'BACKWARD':
            self.moveDrone(directionBackward, speed)
        elif direction == 'LEFT':
            self.moveDrone(directionLeft, speed)
        elif direction == 'RIGHT':
            self.moveDrone(directionRight, speed)
        elif direction == 'UP':
            self.moveDrone(directionUp, speed)
        elif direction == 'DOWN':
            self.moveDrone(directionDown, speed)
        if duration:
            time.sleep(duration)

    def go_to_direction_relative(self, direction, speed, duration):
        current_rotation = float(self.get_current_rotation())
        print(current_rotation)
        current_rotation_radians = current_rotation * np.pi / 180
        R = [[np.cos(current_rotation_radians), 0, np.sin(current_rotation_radians)],
            [0, 1, 0],
            [-np.sin(current_rotation_radians), 0, np.cos(current_rotation_radians)]]
        if direction == 'FORWARD':
            vec = np.dot(np.array(R), np.array([0, 0, 1]))
            direction = {"x": vec[0], "y": vec[1], "z": vec[2]}
            self.moveDrone(direction, speed)
        elif direction == 'BACKWARD':
            vec = np.dot(np.array(R), np.array([0, 0, -1]))
            direction = {"x": vec[0], "y": vec[1], "z": vec[2]}
            self.moveDrone(direction, speed)
        elif direction == 'LEFT':
            vec = np.dot(np.array(R), np.array([-1, 0, 0]))
            direction = {"x": vec[0], "y": vec[1], "z": vec[2]}
            self.moveDrone(direction, speed)
        elif direction == 'RIGHT':
            vec = np.dot(np.array(R), np.array([1, 0, 0]))
            direction = {"x": vec[0], "y": vec[1], "z": vec[2]}
            self.moveDrone(direction, speed)
        elif direction == 'UP':
            self.moveDrone(directionUp, speed)
        elif direction == 'DOWN':
            self.moveDrone(directionDown, speed)
        if duration:
            time.sleep(duration)
        self.hover()

    def hover(self):
        self.moveDrone(stop, speed=1)

    def rotateDroneYaw(self, angle):
        new_rotation = angle
        side = "RIGHT"
        headers = {'Content-type': 'application/json'}
        data = {'DroneID': self.droneid, 'rotation': side, 'Degrees': new_rotation}
        response = requests.post(self.url + "/SetRotationByDroneID", data=json.dumps(data), headers=headers)

    def rotateDroneViewPitch(self, angle):
        self.setPitchRotation(angle)