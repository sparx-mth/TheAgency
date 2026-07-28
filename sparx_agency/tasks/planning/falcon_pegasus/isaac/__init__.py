"""The aircraft half of a FALCON-on-Pegasus run.

Everything here runs inside the ``isaac-sim`` container under Isaac Sim's own
Python. Nothing here runs in the FALCON container, and nothing in the FALCON
container imports it -- the two meet only over the socket defined in
:mod:`~sparx_agency.tasks.planning.falcon_pegasus.link`.
"""
