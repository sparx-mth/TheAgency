#!/usr/bin/env python3
"""Create the InternVLA-N1 agent once, before the flight, with the flight's settings.

The model server loads its 7B checkpoint lazily, on the first ``/agent/init`` --
about 27 seconds on this machine. Doing that inside a recording spends the
flight clock on it, and it used to do worse than that: the policy node asks for
the agent at start-up *and* again from any ``step()`` that finds itself
uninitialised, on another thread, so a cold server received two init requests
and constructed two agents. On an 8 GB card the second checkpoint load is a
``torch.OutOfMemoryError`` twenty-seven seconds in, after which the server
answers nothing and the whole recording is a motionless aircraft. That race is
fixed in ``core/planning/vlas/internvla_n1/client.py`` (init is serialised), but
paying the load before the clock starts is still the right way round.

**Pre-warm with the settings the flight would send, or do not pre-warm at all.**
The agent is constructed once, from the ``model_settings`` of the ``/agent/init``
that created it, and every later init against a live agent is a no-op -- so an
agent created with the server's 640x480 / fx 585 defaults will happily fly a
600x600 / fx 390.6 aircraft, projecting every pixel goal through the wrong
camera, while every file on disk says otherwise. This reads the settings out of
the same binding YAML and through the same node method the flight uses, so there
is nothing to keep in step.

Usage, from the repo root with ROS 2 sourced (it imports the node module)::

    python3 -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.prewarm_server
    ... --config-file <other.yaml>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # the client is CPU-only

from sparx_agency.core.planning.vlas.internvla_n1.client import ModelClient
from sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.n1_policy_node import (
    N1PolicyNode,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))
_DEFAULT_CONFIG = os.path.join(
    _REPO_ROOT, "sparx_agency", "robots", "SJTU", "config", "vla", "internvla_n1.yaml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG,
                        help="the binding YAML the flight will use")
    parser.add_argument("--timeout", type=float, default=None,
                        help="override server.timeout_sec, seconds")
    args = parser.parse_args(argv)

    with open(args.config_file, "r") as handle:
        cfg = yaml.safe_load(handle) or {}
    server = cfg.get("server", {})
    settings = N1PolicyNode._model_settings(cfg.get("camera", {}),
                                            cfg.get("policy_params", {}))
    print("[prewarm] %s:%s" % (server.get("host", "127.0.0.1"),
                               server.get("port", 8087)))
    for key in sorted(settings):
        print("[prewarm]   %-24s %s" % (key, settings[key]))

    client = ModelClient(host=server.get("host", "127.0.0.1"),
                         port=int(server.get("port", 8087)),
                         timeout=float(args.timeout
                                       if args.timeout is not None
                                       else server.get("timeout_sec", 30.0)))
    if not client.check_health():
        print("[prewarm] the model server is not answering; start it first.",
              file=sys.stderr)
        return 2
    started = time.time()
    ok = client.init_agent(model_name="internvla_n1", model_settings=settings)
    print("[prewarm] agent %s after %.1f s"
          % ("ready" if ok else "FAILED", time.time() - started))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
