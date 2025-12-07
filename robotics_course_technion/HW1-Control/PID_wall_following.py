import matplotlib.pyplot as plt
from car_simulator import State, States, Simulator
from utils import Trajectory
import car_consts
import numpy as np


class PID(object):
    def __init__(self, current_time, kp, ki, kd):

        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._start_time = current_time
        self._errors = []
        self._dts = []
        self._integral = 0.0
        self._prev_time = current_time

    def pid_step(self, ref, current, clock):
        error = current - ref
        dt = clock - self._prev_time

        self._integral += error * dt
        derivative = 0 if len(self._errors) == 0 else (error - self._errors[-1]) / dt

        pid = -(self._kp * error + self._ki * self._integral + self._kd * derivative)

        self._dts.append(dt)
        self._errors.append(error)

        return pid



class WALL_FOLLOWING(object):

    def __init__(self, localization_error):

        self.theta = 70
        self.theta_rad = self.theta * np.pi / 180
        self.L = 1.0
        self.localiztion_error = localization_error

    def get_predicted_dist2wall(self, state: State):

        x_err, y_err = np.random.rand(2) * self.localiztion_error
        xr = 0
        yr = -state.x / np.tan(state.yaw) + state.y
        right_beam = ((xr - (state.x + x_err)) ** 2 + (yr - (state.y + y_err)) ** 2) ** 0.5
        xo = 0
        yo = -state.x / np.tan(state.yaw + self.theta_rad) + state.y
        offset_beam = ((xo - (state.x + x_err)) ** 2 + (yo - (state.y + y_err)) ** 2) ** 0.5
        alpha = np.arctan((offset_beam * np.cos(self.theta_rad) - right_beam) / (offset_beam * np.sin(self.theta_rad)))
        dist2wall = right_beam * np.cos(alpha)
        predicted_dist2wall = dist2wall + self.L * np.sin(alpha)
        return predicted_dist2wall


def plot_error(errors, times):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(times, errors)
    ax.grid()
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Error (m)')

    print(f'Average_error: {sum(abs(np.array(errors))) / len(errors)}')
    plt.show()


def main():
    pid = PID(current_time=0, kp=1.0, kd=2.0, ki=0.0)
    STEERING_ERROR_PRECENTAGE = 0.0
    STEERING_ERROR_CONST = 0.0
    LOCALIZATION_ERROR_METER = 0.0

    # ----- Do not change the code below ----------------
    dt = 0.1  # [s] time tick
    SIMULATION_TIME = 15.0  # max simulation time
    MAX_STEER = car_consts.max_steering_angle_rad  # maximum steering angle [rad]
    keep_distance_from_wall = 1.0
    line_coords = []
    for i in range(15):
        line_coords.append([0, i])

    trajectory = Trajectory(dl=0.1, path=np.array(line_coords), TARGET_SPEED=1.0)
    wall_following = WALL_FOLLOWING(LOCALIZATION_ERROR_METER)
    state = State(x=-2.5, y=0, yaw=np.pi / 2 - np.pi / 4, v=1.0)
    clock = 0.0
    states = States()
    states.append(clock, state)
    simulator = Simulator(trajectory=trajectory, DT=dt)
    errors = []
    times = []
    while SIMULATION_TIME >= clock:
        clock += dt
        predicted_dist2wall = wall_following.get_predicted_dist2wall(state)

        delta = pid.pid_step(ref=keep_distance_from_wall, current=predicted_dist2wall, clock=clock)
        times.append(clock)
        errors.append(state.x + keep_distance_from_wall)
        state.predelta = delta
        state = simulator.update_state(state, delta + np.random.rand(
            1) * MAX_STEER * STEERING_ERROR_PRECENTAGE + STEERING_ERROR_CONST)
        states.append(clock, state, delta)
    simulator.show_simulation(states)
    plot_error(errors, times)

if __name__ == '__main__':
    main()
