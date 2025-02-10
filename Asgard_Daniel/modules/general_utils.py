import math
import random
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import json
from modules.controller import Controller
import ast
import numpy as np
import os
import yaml


class PreRun:
    def plot_random_starting_area(self, pest_position, h, r1, r2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pest_position[0], pest_position[1], pest_position[2], color='red', label='pest_position')
        for i in range(500):
            updated_drone_position = self.update_point(pest_position, h, r1, r2)
            ax.scatter(updated_drone_position[0], updated_drone_position[1], updated_drone_position[2], color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.axis('equal')
        plt.show()

    @staticmethod
    def update_point(pest_position, h, r1, r2):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(r1, r2)
        new_x = pest_position[0] + radius * math.cos(angle)
        new_y = pest_position[1] + h
        new_z = pest_position[2] + radius * math.sin(angle)
        return (new_x, new_y, new_z), angle * (180 / math.pi)

    @staticmethod
    def are_vectors_intersecting(p1, q1, p2, q2):
        line1 = LineString([p1, q1])
        line2 = LineString([p2, q2])
        return line1.intersects(line2)

    def random_starting_locations(self, config, ip, port):
        with open(config['config_path'], 'r') as f:
            data = json.load(f)

        controller = Controller(None, None, ip, port)
        controller.start()
        N_drones = len(controller.drone_list)
        drones_positions = []
        for i in range(N_drones):
            controller_ = Controller(None, controller.drone_list[i], ip, port)
            drone_position = json.loads(controller_.getDroneData())['position']
            drone_position = ast.literal_eval(drone_position)
            drones_positions.append(drone_position)
        pest_position = data['pestsDetails'][config['pest_idx_to_attack']]['worldPosition']
        pest_position = np.array([pest_position[i] for i in pest_position.keys()])
        h = config['flying_height']
        r1 = config['r1']
        r2 = config['r2']
        # plot_random_starting_area(pest_position, h, r1, r2)
        agents_locations_ = []
        angle_0 = None
        for i in range(N_drones):
            if angle_0 is None:
                updated_drone_position, angle_0 = self.update_point(pest_position, h, r1, r2)
            else:
                updated_drone_position, angle_1 = self.update_point(pest_position, h, r1, r2)
                while np.min([angle_0 - angle_1, angle_1 - angle_0]) + 360 < config['min_deg_diff']:
                    updated_drone_position, angle_1 = self.update_point(pest_position, h, r1, r2)
            agents_locations_.append(updated_drone_position)
        if self.are_vectors_intersecting([agents_locations_[0][0], agents_locations_[0][2]], [drones_positions[0][0],
                                    drones_positions[0][2]], [agents_locations_[1][0], agents_locations_[1][2]],
                                    [drones_positions[1][0], drones_positions[1][2]]):
            temp = agents_locations_[1]
            agents_locations_[1] = agents_locations_[0]
            agents_locations_[0] = temp
        return agents_locations_


class Results:
    @staticmethod
    def get_folders(path):
        folders = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        return folders

    @staticmethod
    def get_most_recent_folder(path):
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not dirs:
            return None
        return max(dirs, key=lambda d: os.path.getctime(os.path.join(path, d)))

    def get_results_combined(self):
        with open('./config/config.yml') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        results_path = cfg['results_path']
        most_recent_folder = self.get_most_recent_folder(results_path)

        if most_recent_folder:
            print("The most recent folder in", results_path, "is:", most_recent_folder)
        else:
            print("No folders found in", results_path)

        path = results_path + most_recent_folder + '/Combined Results.json'

        with open(path, 'r') as f:
            data = json.load(f)

        data_drones = [data[i]['drones'] for i in data.keys()]
        data_pests = [data[i]['pests'] for i in data.keys()]

        N = len(data_drones)
        counter = 0

        for data_drone, data_pest in zip(data_drones, data_pests):
            if data_drone[0]['status'] == 1 and data_drone[1]['status'] == 3:
                counter += 1
            elif data_drone[0]['status'] == 3 and data_drone[1]['status'] == 1:
                counter += 1
            elif not data_pest[0]['isDetected'] and not data_pest[1]['isDetected']:
                N -= 1
            else:
                continue

        grade = (counter / N) * 100
        print(f'{counter}/{N}')
        print(f'{grade}%')
