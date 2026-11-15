"""The ObjectNav benchmark harness: run a headless agent through an environment, score it, log it, compare it.

No simulator here and no ROS. The environments come from the benchmark branches
(each implements ``core.planning.objnav.interfaces.ObjNavEnv``); this package
drives any of them the same way, scores every benchmark with the same code, and
writes results outside the repository.

Python 3.8 syntax; the harness itself is standard library only (numpy arrives
with the core types and the fake environment).
"""
