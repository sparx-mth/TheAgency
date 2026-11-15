# Common planner geometry and optional OMPL

The 2D/3D geometry helpers are ROS-free and do not require OMPL. Importing A*,
ObjectNav, interpolation, resampling, or clearance-grid geometry must not import
native sampling-planner bindings.

OMPL is loaded only when explicitly requested:

- `planners` resolves its existing RRT*, BIT* and informed-RRT* exports lazily.
- `common` resolves `ob`, `og`, `OMPL_AVAILABLE` and `OMPL_ERROR` lazily.
- `make_clearance_objective_2d`, `make_clearance_objective_3d` and
  `setup_ompl_space_3d` import bindings when called and retain their explicit
  missing-dependency errors.

Public names and geometry algorithms are unchanged. Python 3.8 module
`__getattr__` provides lazy exports without a new dependency. A wildcard import
requests the optional exports too; use explicit geometry imports for an
OMPL-free process.

This boundary matters on machines where importing OMPL causes an allocator
abort at interpreter shutdown, even when no OMPL planner was used. It prevents
unrelated A*/ObjectNav runs from loading those bindings; it does not repair an
installed OMPL library used by an actual sampling planner.

## Validation

From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests/test_runtime_boundaries.py -q
```

The boundary tests use fresh subprocesses. They verify that pure navigation
never attempts an OMPL import, optional public exports still resolve, missing
bindings fail explicitly, and the process exits normally.

