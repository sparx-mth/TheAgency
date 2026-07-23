import yaml
from sparx_agency.robots.ASGARD.asgard_planner import AsgardPlanner
from multiprocessing import Process
from sparx_agency.robots.ASGARD.asgard_controller import Controller
import subprocess
import time
from sparx_agency.robots.ASGARD.results import Results
import subprocess
import threading


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

class Wow:
    """
    Wow wraps all wrappers instances.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.wrappers = []

    def init_wrappers(self, ips, ports):
        """
        Initialize instance for each wrapper.
        """
        for ip, port in zip(ips, ports):
            self.wrappers.append(RunWrapper(ip, port))
            if self.debug:
                break

    def start(self):
        """
        Initialize independent process for each instance.
        """
        if self.debug:
            self.wrappers[0].loop()
        else:
            for wrapper in self.wrappers:
                Process(target=wrapper.run_wrapper).start()


class Wrapper:
    """
    Wrapper wraps all entities instances.
    """
    def __init__(self, drones, debug: bool = False):
        self.drones = drones
        self.debug = debug
        self.agents = []

    def init_agents(self, config: dict, agents_locations, ip, port):
        """
        Initialize instance for each entity.
        """
        for i, drone in enumerate(self.drones):
            self.agents.append(AsgardPlanner(config, drone, agents_locations[i],
                    time.time() + config['trigger_time'] / config['run_speed_factor'], ip, port))
            if self.debug:
                break

    def start(self):
        """
        Initialize independent process for each instance.
        """
        if self.debug:
            self.agents[0].loop()
        else:
            for agent in self.agents:
                Process(target=agent.loop).start()


def main(ip, port):
    with open('./config/config.yml') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

    f = True
    while f:
        try:
            agents_locations = PreRun().random_starting_locations(cfg, ip, port)
            f = False
        except:
            print(ip)
            print(port)
            continue


    controller = Controller(None, None, ip, port)
    controller.start()
    drones = controller.drone_list

    wrapper = Wrapper(drones, cfg['debug'])
    wrapper.init_agents(cfg, agents_locations, ip, port)
    wrapper.start()

    quit()


def read_output(stream, prefix):
    for line in stream:
        print(f"{prefix}: {line.decode().strip()}")


class RunWrapper:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def run_wrapper(self):
        while True:
            # # Run the external script using subprocess
            # process = subprocess.Popen(['python', 'wrapper.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # process = subprocess.Popen(['python', 'wrapper.py', ip, port], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Run the external function using subprocess
            # process = subprocess.Popen(['python', '-c', 'import wrapper; wrapper.main()'],
            #                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process = subprocess.Popen(['python', '-c', f'import wrapper; wrapper.main("{self.ip}", "{self.port}")'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            time.sleep(6)


            # # Start threads to read stdout and stderr
            # stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "stdout"))
            # stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "stderr"))
            # stdout_thread.start()
            # stderr_thread.start()
            #
            # # # Wait for the process to finish
            # # return_code = process.wait()
            # # Wait for the process to finish
            # stdout, stderr = process.communicate()
            #
            # # Check the return code to see if the process finished successfully
            # return_code = process.returncode
            #
            # # Join threads to ensure they finish properly
            # stdout_thread.join()
            # stderr_thread.join()


            # Wait for the process to finish
            stdout, stderr = process.communicate()

            # Check the return code to see if the process finished successfully
            return_code = process.returncode
            if return_code == 0:
                print("External function finished successfully.")
            else:
                print("External function finished with errors.")
                print("Error message:", stderr.decode())
            # process.kill()
            process.terminate()
            time.sleep(1)


if __name__ == "__main__":
    ips = ['http://127.0.0.1'] * 5
    # ips = ['http://172.16.17.9'] * 5
    ports = ['8081', '8082', '8083', '8084', '8085']

    # ips = ['http://127.0.0.1'] * 5 + ['http://172.16.17.9'] * 5
    # ports = ports + ports

    wow = Wow()
    wow.init_wrappers(ips, ports)
    wow.start()

    # main()

    # Results().get_results_combined()

