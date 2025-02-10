import yaml
from modules.researcher import Researcher
from multiprocessing import Process
from modules.controller import Controller
import subprocess
import time
from modules.general_utils import PreRun, Results
import subprocess
import threading


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
            self.agents.append(Researcher(config, drone, agents_locations[i],
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
