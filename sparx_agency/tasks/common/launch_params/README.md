# launch_params

Give any launchable command a parameter screen, built from whatever declares its
parameters — so the list can never drift from the code.

A launcher normally hard-codes each command as a string, which means the only
knobs an operator can reach are the ones whoever wrote that string thought to
spell out. In this repo that gap is wide: the localization node declares 37
parameters and its launch command names 6; the FALCON object mission has around
290 and names 4. The rest are invisible, and reaching them means editing the
launcher.

## The three steps

```python
from sparx_agency.tasks.common.launch_params import command, discovery, Source

found  = discovery.discover((Source("ros2_node", "sparx_agency/.../localization_node.py"),))
parsed = command.parse(item.command)      # what the command already names
found.params.extend(parsed.params)        # ...which is what a plain start runs
text   = command.render(parsed, found.params)
```

1. **Discover** (`discovery.py`, `sources/`) — read the full parameter set out of
   its declaration site, with the comments that explain it.
2. **Edit** (`spec.py`, `editor.py`) — show them grouped and searchable, each
   against its own default, with one click to put any (or all) of them back.
3. **Render** (`command.py`) — write the command back out carrying only what was
   deliberately set.

`store.py` keeps the operator's overrides between sessions.

## Sources

| Kind | Reads | Used for |
| --- | --- | --- |
| `ros2_node` | `self.declare_parameter(...)` | ROS2 nodes |
| `argparse` | `p.add_argument(...)` | plain Python scripts |
| `roslaunch` | `<arg name= default=>` | ROS1 launch files |
| `yaml` | a hand-commented YAML config | `falcon/config/mission.yaml` |

The Python readers use `ast`, so a call spanning several lines or a default that
is a negative number is no harder than the one-liner case. All four harvest the
surrounding comments: a heading becomes a section, the block above a declaration
becomes its documentation, a trailing note becomes its one-line summary.

## The two rules that matter

**Defaults follow precedence, not file order.** `Source(..., defines_defaults=False)`
marks a source that is authoritative about which parameters *exist* but not about
what they are set to. The object mission needs this: `object_mission.launch`
declares every argument, but `mission.yaml` is read after it and decides what a
plain run actually uses — so that is the default the screen shows and the value
"reset" returns to.

**Only deliberate values are emitted.** A parameter renders when it is `pinned`
(the command spelled it out, so it is part of that command's identity) or when it
has been moved off its default. That is what keeps a 290-knob mission startable
as a one-line command.

## Syntaxes

A parameter knows how it is written, so one command can mix them:

```
NAV_MODE=hybrid ./run_object_mission.sh --falcon-only office gui vel_x:=0.3
└─ env ─┘                                                      └ roslaunch ┘
```

`ros2` (`-p k:=v`), `roslaunch` (`k:=v`), `cli` (`--k v`), `flag` (`--k`),
`env` (`K=v`), and `slot` — a value a command *template* substitutes itself, for
commands whose parameters sit inside quotes (`docker exec ... bash -lc '...'`)
and cannot be parsed back out.

## Tests

```bash
.venv/bin/python -m pytest sparx_agency/tasks/common/launch_params/tests/ -q
```

## Consumer

`demos/Demo_No4_XTEND_MapRoom/launcher/` — the XTEND pipeline launcher.
