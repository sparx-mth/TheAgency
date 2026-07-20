"""Where the demo's machines keep things, and how to enter their shells.

Every path here is on a *remote* machine -- the Jetson on the drone, or the
operator's PC -- so none of them can be derived from this checkout's location.
They are gathered in one module instead: moving the workspace is then one edit
rather than a hunt through sixteen command strings.
"""
from __future__ import annotations

#: Default SSH target for the drone's Jetson.
JETSON_SSH_DEFAULT = "user@192.0.0.89"

#: The repo checkout on the Jetson.
JETSON_REPO = "/home/user/agency_ws"
#: The repo checkout on the operator's PC.
PC_REPO = "/home/user1/GIT/TheAgency"

#: FALCON's task directory on the Jetson: the mission scripts run from here.
FALCON_DIR = JETSON_REPO + "/sparx_agency/tasks/planning/falcon"

#: The Jetson's local screen. The FALCON container's windows (RViz, the BEV
#: viewer, the object list) are X clients, and a bare `ssh jetson` has no
#: DISPLAY, so every windowed command sets it explicitly.
JETSON_DISPLAY = ":0"

#: The ROS domain both machines must agree on, or no topic crosses between them.
ROS_DOMAIN_ID = 5

JETSON_ENV = f"""
cd {JETSON_REPO}
source /opt/ros/humble/setup.bash
source {JETSON_REPO}/venv/bin/activate
export ROS_DOMAIN_ID={ROS_DOMAIN_ID}
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:${{LD_LIBRARY_PATH}}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:{JETSON_REPO}:{JETSON_REPO}/sparx_agency:${{PYTHONPATH}}
"""

PC_ENV = f"""
cd {PC_REPO}
source /opt/ros/jazzy/setup.bash
source {PC_REPO}/venv/bin/activate
export ROS_DOMAIN_ID={ROS_DOMAIN_ID}
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}:/opt/ros/jazzy/lib
export PYTHONPATH=/usr/lib/python3.12/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages:{PC_REPO}:${{PYTHONPATH}}
"""


def normalize_command(text: str) -> str:
    """Strip trailing whitespace from every line and blank lines from the ends."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def wrap_with_env(machine: str, command: str) -> str:
    """Prefix a command with the shell setup of the machine that will run it.

    Args:
        machine: ``"jetson"`` or ``"pc"``. Anything else gets the PC's setup,
            which is what a locally-run command needs.
        command: The command text.

    Returns:
        The environment setup followed by the command.
    """
    env = JETSON_ENV if machine == "jetson" else PC_ENV
    return normalize_command(env) + "\n" + normalize_command(command)
