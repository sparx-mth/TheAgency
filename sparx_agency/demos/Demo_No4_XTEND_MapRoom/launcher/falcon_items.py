"""The planner side of the demo: the object mission, NavDP, and the viewers.

These are the commands ``tasks/planning/falcon/run_object_mission.sh`` documents,
declared here so the launcher starts exactly what that script does rather than a
copy of it that drifts. The parameter screens come from the same three files the
script itself reads -- ``config/mission.yaml``, ``config/mission_config.py`` and
``adapter/launch/object_mission.launch`` -- so every knob the mission has is
reachable, at the default a plain run would use.

**The two-terminal workflow.** Loading the YOLO-World TensorRT engines is the
slow part of a start; FALCON is the part worth iterating on. So they are two
commands, not one:

* *FALCON A* runs only the detector sidecar and holds it up.
* *FALCON B* runs only the bridge and the FALCON container, reusing the sidecar
  already running and never touching it.

Start A once and leave it; restart B as often as you like, in seconds, with no
engine reload. B refuses to start if no sidecar is up -- a mission whose
detector never publishes looks perfectly healthy while being unable to see
anything, which is far worse than not starting.

The nav modes other than ``astar`` also need the NavDP inference server, which
must run on the FALCON *host*: the Noetic container ships no TensorRT and reaches
the server over ``--network host`` loopback.
"""
from __future__ import annotations

from sparx_agency.tasks.common.launch_params.discovery import Source
from sparx_agency.tasks.common.launch_params.spec import (CLI, ENV, SLOT,
                                                          ParamSpec)

from .environments import FALCON_DIR, JETSON_DISPLAY, JETSON_REPO
from .item import LaunchItem

#: The three files the mission script reads, repo-relative.
_FALCON = "sparx_agency/tasks/planning/falcon"
MISSION_YAML = _FALCON + "/config/mission.yaml"
MISSION_SCHEMA = _FALCON + "/config/mission_config.py"
MISSION_LAUNCH = _FALCON + "/adapter/launch/object_mission.launch"
NAVDP_SERVER = "sparx_agency/tasks/planning/navdp/server/navdp_trt_server.py"

#: The launch arguments run_object_mission.sh fills in ITSELF, from its
#: environment variables and positionals -- see its LAUNCH_ARGS block. They must
#: not also be offered as `key:=value` overrides: the script would then pass each
#: one to roslaunch twice, and, worse, the two knobs disagree. `nav_mode` is the
#: case that bites: the launch file defaults it to `fallback` while mission.yaml
#: sets NAV_MODE to `astar`, so an operator who picked `fallback` from the launch
#: side would change nothing and fly `astar` -- with no NavDP rescue -- believing
#: otherwise. Set these through the env parameters (NAV_MODE, MAP, ...) instead.
SCRIPT_OWNED_LAUNCH_ARGS = ("map_name", "selection_mode", "seed", "nav_mode",
                            "objects_file", "target_object")

#: Parameter sources for a command that runs the whole mission.
#:
#: Order is the mission's own precedence. ``mission.yaml`` first, because a value
#: set there is what a plain run already uses -- it is the default this screen
#: must show and the one "reset" must return to. The launch file second and
#: marked as not defining defaults: it is authoritative about which arguments
#: EXIST (it declares far more than the config re-defaults) but not about what
#: they are set to, since the config was read after it.
MISSION_SOURCES = (
    Source("yaml", MISSION_YAML, env_schema_from=MISSION_SCHEMA),
    Source("roslaunch", MISSION_LAUNCH, defines_defaults=False,
           suppress=SCRIPT_OWNED_LAUNCH_ARGS),
)

#: OBJECTS_FILE is commented out in mission.yaml (the script derives it from
#: OBJECTS_DIR), so it is not discovered -- but it is the override you need when
#: the catalog is not at the default name. Empty means "let the script derive it".
_OBJECTS_FILE_PARAM = ParamSpec(
    name="OBJECTS_FILE", default="", syntax=ENV, section="OBJECT CATALOG",
    doc="Path to the objects JSON. Empty derives <objects_dir>/objects.json. Must "
        "be visible inside the container: under OBJECTS_DIR, or under the repo "
        "mounted at /opt/sparx_agency.")

#: Slot for the directory the mission scripts are run from.
_FALCON_DIR_PARAM = ParamSpec(
    name="falcon_dir", default=FALCON_DIR, syntax=SLOT,
    section="Where it runs",
    doc="Directory holding run_object_mission.sh on the machine that runs it.")

_CONTAINER_PARAMS = (
    ParamSpec(name="container", default="falcon", syntax=SLOT,
              section="Where it runs",
              doc="Name of the running FALCON container to attach to."),
    ParamSpec(name="display", default=JETSON_DISPLAY, syntax=SLOT,
              section="Where it runs",
              doc="X display for the window. A bare ssh has none, so it is set "
                  "explicitly; :0 is the Jetson's own screen."),
)


FALCON_ITEMS: list[LaunchItem] = [
    LaunchItem(
        name="11. NavDP TensorRT inference server (host)",
        machine="jetson",
        tmux_name="navdp_server",
        description=(
            "The point-goal policy every nav_mode except 'astar' calls. Runs on the "
            "FALCON HOST, not in the container: the Noetic image ships no TensorRT, and "
            "FALCON reaches this over --network host loopback (127.0.0.1:8888).\n\n"
            "Engines are built per device and are NOT portable, so engine-dir must name "
            "the tag of the machine running it (orin_sm87 on the Jetson). Start it in the "
            "same power mode the engines were built in — MAXN + jetson_clocks — or the "
            "timings will not match.\n\n"
            "Start this BEFORE the mission if nav_mode is fallback/hybrid/combination/navdp; "
            "'astar' is the one mode that needs no server."
        ),
        command=(
            "cd {workspace}\n"
            "export NAVDP_REPO={navdp_repo}\n"
            "PYTHONPATH=$PWD {python} \\\n"
            "  -m sparx_agency.tasks.planning.navdp.server.navdp_trt_server {params}"
        ),
        template=True,
        enabled_by_default=False,
        param_sources=(Source("argparse", NAVDP_SERVER),),
        params=(
            ParamSpec(name="workspace", default=JETSON_REPO, syntax=SLOT,
                      section="Where it runs",
                      doc="The checkout holding the BUILT engines — not necessarily "
                          "the one you develop in."),
            ParamSpec(name="python", default="python3", syntax=SLOT,
                      section="Where it runs",
                      doc="Interpreter with tensorrt + pycuda: the env that BUILT the "
                          "engines. The algorithms venv has neither and will not import."),
            ParamSpec(name="navdp_repo", default="/home/user/GIT/NavDP/baselines/navdp",
                      syntax=SLOT, section="Where it runs",
                      doc="Dir containing policy_network.py. Required even on the trt "
                          "backend: image preprocessing is imported from it."),
            ParamSpec(name="engine-dir", default="sparx_agency/tasks/planning/navdp/"
                                                 "engines/orin_sm87",
                      syntax=CLI, pinned=True, section="Engines",
                      doc="Per-device engine dir; the name is the hardware tag. "
                          "Engines do not transfer between machines."),
            ParamSpec(name="port", default="8888", syntax=CLI, pinned=True,
                      section="Engines",
                      doc="navdp_click_node and the FALCON planners expect 8888."),
        ),
    ),
    LaunchItem(
        name="12. FALCON A: YOLO detector sidecar only (leave running)",
        machine="jetson",
        tmux_name="falcon_detector",
        description=(
            "TERMINAL A of the two-terminal workflow. Runs ONLY the YOLO-World TensorRT "
            "detector, a ROS2 sidecar on the HOST (the FALCON container has no CUDA/"
            "TensorRT/pycuda). Nothing plans and nothing flies.\n\n"
            "Loading the engines is the expensive part of a start, so this exists to pay "
            "it ONCE and keep it paid: start it, leave it up, and relaunch the mission "
            "next door with item 13 as often as you like.\n\n"
            "It starts on the placeholder prompt below and is RE-PROMPTED by the mission "
            "director the moment you select an object. Stopping this session stops the "
            "detector. Log: /tmp/object_mission/sidecar.log"
        ),
        command="cd {falcon_dir}\n{env}./run_object_mission.sh --detector-only",
        template=True,
        enabled_by_default=False,
        param_sources=(Source("yaml", MISSION_YAML, env_schema_from=MISSION_SCHEMA,
                              only_groups=("detector",)),),
        params=(
            _FALCON_DIR_PARAM,
            ParamSpec(name="ENGINES_DIR", default="", section="Detector engines",
                      syntax=ENV,
                      doc="Host dir with the *.engine files. Empty derives it from the "
                          "detected hardware tag: yolo_world_trt/engines/<tag>."),
            ParamSpec(name="TEXT_WEIGHTS", default="", section="Detector engines",
                      syntax=ENV,
                      doc="Full path to yolov8<model>-worldv2.pt. Empty derives it as "
                          "<weights_dir>/yolov8<model>-worldv2.pt."),
            ParamSpec(name="PYTHON", default="", section="Detector engines",
                      syntax=ENV,
                      doc="Interpreter with tensorrt + pycuda (+ torch/ultralytics). "
                          "Empty uses python3 from the activated environment."),
        ),
    ),
    LaunchItem(
        name="13. FALCON B: mission only — bridge + container (fast relaunch)",
        machine="jetson",
        tmux_name="falcon_mission",
        description=(
            "TERMINAL B of the two-terminal workflow, and the one you re-run. Starts the "
            "ros1<->ros2 bridge and the FALCON container (nav + A*/NavDP + object-approach "
            "+ the mission director), REUSING the detector sidecar already running from "
            "item 12 — it neither starts nor, on exit, stops it, so no engine is reloaded.\n\n"
            "It refuses to start when no sidecar is running: nothing would ever publish a "
            "detection, so the mission could only ever land by A* alone, while looking "
            "perfectly healthy.\n\n"
            "The bridge IS restarted every time and cannot be kept: it is a ROS1 node "
            "against the roscore that roslaunch starts inside the container, so that "
            "master dies with the container. A fresh roscore is wanted anyway — it is what "
            "stops a stale latched goal from pre-arming the planners.\n\n"
            "Every parameter of the mission is on the Parameters tab: the map and how the "
            "target is picked, which planner flies (nav_mode), the staging vantage point, "
            "the five controllers and their speeds, the A* and BEV knobs, path correction, "
            "landing, and the recovery ladder. Only what you change is put on the command "
            "line; everything else stays as config/mission.yaml has it."
        ),
        command=("cd {falcon_dir}\n"
                 "{env}./run_object_mission.sh --falcon-only {MAP} {SELECTION_MODE} {params}"),
        template=True,
        enabled_by_default=False,
        param_sources=MISSION_SOURCES,
        params=(_FALCON_DIR_PARAM, _OBJECTS_FILE_PARAM),
    ),
    LaunchItem(
        name="14. FALCON: full object mission (detector + bridge + container)",
        machine="jetson",
        tmux_name="falcon_full_mission",
        description=(
            "All three processes in one session: detector sidecar, bridge, and the FALCON "
            "container. The single-terminal form — simplest for a one-shot run, but every "
            "restart reloads the TensorRT engines.\n\n"
            "Prefer items 12 + 13 while iterating; use this when you want one command that "
            "brings the whole mission up and takes it all down again on exit.\n\n"
            "Same parameter screen as item 13, plus the detector's own settings."
        ),
        command=("cd {falcon_dir}\n"
                 "{env}./run_object_mission.sh {MAP} {SELECTION_MODE} {params}"),
        template=True,
        enabled_by_default=False,
        param_sources=MISSION_SOURCES,
        params=(_FALCON_DIR_PARAM, _OBJECTS_FILE_PARAM),
    ),
    LaunchItem(
        name="15. FALCON RViz (3D view inside the container)",
        machine="jetson",
        tmux_name="falcon_rviz",
        description=(
            "Opens the pre-configured RViz — BEV map, planned path and odometry already "
            "wired up — from inside the running FALCON container.\n\n"
            "Needs the container up first (item 13 or 14) and a display: it is an X client "
            "inside the container, so DISPLAY is set explicitly rather than inherited from "
            "an ssh session that has none."
        ),
        command=("docker exec -it {container} bash -lc "
                 "'export DISPLAY={display} && source /catkin_ws/devel/setup.bash && "
                 "roslaunch {package} {launch_file}'"),
        template=True,
        enabled_by_default=False,
        params=_CONTAINER_PARAMS + (
            ParamSpec(name="package", default="exploration_manager", syntax=SLOT,
                      section="What it opens",
                      doc="ROS package holding the RViz launch file."),
            ParamSpec(name="launch_file", default="rviz.launch", syntax=SLOT,
                      section="What it opens",
                      doc="Launch file that starts RViz with the demo's config."),
        ),
    ),
    LaunchItem(
        name="16. FALCON BEV click-goal viewer (standalone)",
        machine="jetson",
        tmux_name="falcon_bev_goal",
        description=(
            "The interactive 2D map: occupancy, the planner routes, the flown path, the "
            "system-status HUD, and click-to-goal.\n\n"
            "The mission normally starts this itself (bev_viewer defaults to true), so run "
            "it by hand only when the mission was launched headless with bev_viewer:=false. "
            "Starting a second copy against a mission that already has one is harmless but "
            "pointless."
        ),
        command=("docker exec -it {container} bash -lc "
                 "'export DISPLAY={display} && source /catkin_ws/devel/setup.bash && "
                 "rosrun {package} {node}'"),
        template=True,
        enabled_by_default=False,
        params=_CONTAINER_PARAMS + (
            ParamSpec(name="package", default="falcon_adapter", syntax=SLOT,
                      section="What it opens",
                      doc="ROS package holding the viewer node."),
            ParamSpec(name="node", default="bev_click_goal_node.py", syntax=SLOT,
                      section="What it opens",
                      doc="The viewer node script."),
        ),
    ),
]
