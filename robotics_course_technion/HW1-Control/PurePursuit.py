import numpy as np
import math
from utils import Trajectory
import car_consts
import matplotlib.pyplot as plt
from car_simulator import State, States, Simulator



class PurePursuit_Controller(object):
    def __init__(self, cx, cy, k=0.1, Lfc=0.5, Kp =1.0, WB =0.335, MAX_ACCEL=1.0, MAX_SPEED=3, MIN_SPEED=-3, MAX_STEER=np.deg2rad(27.0), MAX_DSTEER=np.deg2rad(150.0) ):
        self.k = k  # look forward gain
        self.Lfc = Lfc   # [m] look-ahead distance
        self.Kp = Kp  # speed proportional gain
        self.WB = WB # wheelbase
        self.old_nearest_point_index = None
        self.cx = cx # trajectory x coords
        self.cy = cy # trajectory y coords
        self.MAX_ACCEL = MAX_ACCEL
        self.MAX_SPEED  = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.MAX_STEER = MAX_STEER
        self.MAX_DSTEER = MAX_DSTEER

    def pure_pursuit_steer_control(self, state: State, trajectory: Trajectory,  dt):
        '''
        input: 
        current state,
        trajectory,
        dt
        Return the target index, look ahead distance, and closest index of path coords
        '''
        # TODO
        # ind, Lf, closest_index = self.search_target_index(state)
        # return delta, ind, closest_index
    
    def proportional_control_acceleration(self, target_speed, current_speed, dt=0.1):
        '''
        returns updated linear speed
        '''
        # TODO
        pass
        # return linear_velocity
        

    def search_target_index(self, state: State):
        '''
        input: current state
        output:
        target index, 
        updated look-ahead distance,
        index of closest coords on trajectory 
        '''
        # TODO
        pass
        # return ind, Lf, self.old_nearest_point_index
    
    def calc_distance(self, rear_x, rear_y, point_x, point_y):
        '''
        calculates the distance between two coords
        '''
        # TODO
        pass


def plot_error(closest_path_coords, states:States, trajectory:Trajectory):
    fig = plt.figure()
    ax = fig.add_subplot()
    total_tracking_error = 0
    for i in range(len(states.x)-1):
        tracking_error = ((closest_path_coords[i][0] -states.rear_x[i])**2 + (closest_path_coords[i][1] - states.rear_y[i])**2)**0.5
        total_tracking_error += tracking_error
        ax.scatter(i, tracking_error, c='b')
    print(f'average tracking error {total_tracking_error/(i+1)}')
    ax.set_xlabel('itr')
    ax.set_ylabel('Tracking Error')
    ax.grid()
    plt.show()

def main():
    #  hyper-parameters
    k = 0.1  # look forward gain
    Lfc = 1.0  # [m] look-ahead distance
    Kp = 1.0  # speed proportional gain
    dt = 0.1  # [s] time tick
    target_speed = 1.0  # [m/s]
    T = 100.0  # max simulation time
    WB = car_consts.wheelbase 
    MAX_STEER = car_consts.max_steering_angle_rad  # maximum steering angle [rad]
    MAX_DSTEER = car_consts.max_dt_steering_angle  # maximum steering speed [rad/s]
    MAX_SPEED = car_consts.max_linear_velocity  # maximum speed [m/s]
    MIN_SPEED = car_consts.min_linear_velocity  # minimum speed [m/s]
    MAX_ACCEL = 1.0  # maximum accel [m/ss]

    path = np.load('path_maze_meter_sim.npy')
    trajectory = Trajectory(dl=0.1, path =path, TARGET_SPEED=target_speed)
    state = State(x=trajectory.cx[0], y=trajectory.cy[0], yaw=trajectory.cyaw[0], v=0.0)
    lastIndex = len(trajectory.cx) - 1
    clock = 0.0
    states = States()
    states.append(clock, state)
    pp = PurePursuit_Controller(trajectory.cx, trajectory.cy, k, Lfc, Kp, WB, MAX_ACCEL, MAX_SPEED, MIN_SPEED, MAX_STEER, MAX_DSTEER)
    target_ind, _, nearest_index = pp.search_target_index(state)
    simulator = Simulator(trajectory, dt)
    closest_path_coords = []
    while T >= clock and lastIndex > target_ind:
        state.v = pp.proportional_control_acceleration(target_speed, state.v, dt)
        delta, target_ind, closest_index = pp.pure_pursuit_steer_control(state, trajectory, dt)
        state.predelta = delta
        state = simulator.update_state(state, delta)  # Control vehicle
        clock += dt
        states.append(clock, state, delta)
        closest_path_coords.append([trajectory.cx[closest_index], trajectory.cy[closest_index]])
    simulator.show_simulation(states, closest_path_coords)
    plot_error(closest_path_coords, states, trajectory)


if __name__ == '__main__':
    main()
