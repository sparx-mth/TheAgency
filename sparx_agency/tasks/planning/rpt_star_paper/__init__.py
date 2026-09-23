"""RPT* paper replication -- the algorithm alone, no pipeline attached.

Reproduces the experiments of

    Yunpeng Lyu, Chao Cao, Ji Zhang, Howie Choset, Zhongqiang Ren.
    *RPT\\*: Global Planning with Probabilistic Terminals for Target Search in
    Complex Environments.* arXiv:2601.12701, January 2026.

against
:mod:`sparx_agency.core.planning.routing.rpt_star`, on the paper's own datasets
and against the paper's own baselines. Nothing here touches ROS, a map, a
robot or a detector: the point is to know whether the algorithm behaves as
advertised before any of that is wired to it.

See ``README.md`` for what the numbers came out as.
"""

