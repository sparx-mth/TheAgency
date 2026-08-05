import math
import numpy as np
import math
import random
from shapely.geometry import LineString
import json
import ast
import os
import yaml


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
