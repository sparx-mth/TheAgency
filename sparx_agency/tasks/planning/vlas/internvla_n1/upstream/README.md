# InternNav upstream patches

Patched copies of the InternVLA-N1 **agent** and **server** that this stack
deploys into an InternNav checkout, plus the one deeper model edit they depend
on. Upstream: <https://github.com/InternRobotics/InternNav>. The checkout is
external and not vendored; copy these over it (or apply the model edit below)
before starting the server.

```
<internnav>/code/internnav/agent/internvla_n1_agent.py   <- internvla_n1_agent.py (here)
<internnav>/code/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py  <- edit below
```

Restart the server after applying — the agent and model are imported once at
`/agent/init`.

## Why these exist

Stock InternVLA-N1 discretizes System 1's trajectory into VLN-CE actions
(`STOP`/`FORWARD`/`TURN_LEFT`/`TURN_RIGHT`) and returns only the first action. It
also does not surface the System-2 pixel goal. Two patch sets fix that:

* **PATCH 1–4 (pixel goal).** Carry S2's pixel goal through to the HTTP response.
* **PATCH 5 (continuous trajectory).** Carry S1's **continuous body-frame
  trajectory** through to the HTTP response, so a trajectory follower can fly the
  curve directly instead of the coarse discrete action — the NavDP way.

## The continuous trajectory is already there

It is not an addition to the model, only an exposure of what it already computes:

* `model/utils/vln_utils.py::traj_to_actions(dp_actions, use_discrate_action=False)`
  returns the integrated mean path `[T+1, 2]` in **metres, body FLU** (x forward,
  y left) — the same curve it otherwise discretizes.
* `model/utils/vln_utils.py::S1Output` already has a `trajectory` field.

So the model edit is one line filling that field, and the agent edit carries it
to the wire. `core/planning/vlas/internvla_n1` parses it and prefers it over the
discrete action automatically.

## The one model edit (not vendored as a whole file)

`code/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`,
`s1_step_latent`:

```python
# before
if self.continuous_traj:
    action_list = traj_to_actions(dp_actions)
...
output = S1Output(idx=action_list[:4])

# after
trajectory = None
if self.continuous_traj:
    # traj_to_actions unnormalises dp_actions IN PLACE (dp[:, :, :2] /= 4), so
    # compute the trajectory on a clone or the discrete pass divides by 16.
    trajectory = traj_to_actions(dp_actions.clone(), use_discrate_action=False)
    action_list = traj_to_actions(dp_actions)
...
output = S1Output(idx=action_list[:4], trajectory=trajectory)
```

The **in-place `/= 4` on a clone** is the one subtlety — get it wrong and every
distance is scaled by 4 (or 16), which flies but flies wrong.

## What the response looks like after the patches

`POST /agent/internvla_n1/step` returns, per step:

```json
{"action": [{"action": [1], "ideal_flag": true,
             "pixel_goal": [y, x],
             "trajectory": [[0.0, 0.0], [0.24, 0.03], [0.48, 0.07], "..."]}],
 "pixel_goal": [y, x], "pixel_goal_step": 12}
```

`trajectory` is `null` on a pure-S2 discrete step (a turn or look-down, where the
model genuinely emits no curve); the client falls back to the discrete action for
those and uses the curve everywhere else.

