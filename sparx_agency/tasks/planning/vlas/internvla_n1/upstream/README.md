# InternNav upstream patches

Patched copies of the InternVLA-N1 **agent** and **server** that this stack
deploys into an InternNav checkout, plus the deeper model edits they depend on.
Upstream: <https://github.com/InternRobotics/InternNav>. The checkout is
external and not vendored; copy these over it and apply the edits below before
starting the server.

```
<internnav>/code/internnav/agent/internvla_n1_agent.py         <- internvla_n1_agent.py (here)
<internnav>/code/internnav/utils/comm_utils/server.py          <- server.py (here)
<internnav>/code/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py  <- edits below
<internnav>/code/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py    <- edit below
<internnav>/code/internnav/model/encoder/__init__.py                            <- edit below
<internnav>/code/checkpoints/depth_anything_v2_metric_hypersim_vits.pth         <- fetch, below
```

(`video_stream.py` here is not part of this deployment — it belongs to the
Sphera/Rooster bring-up described in the parent README, and nothing copies it
into an InternNav checkout.)

Restart the server after applying — the agent and model are imported once at
`/agent/init`.

**Two of the edits below are not features; they are the difference between a
server that answers and a server that lies.** Edits **2** (the 4-bit quantiser)
and **3** (the NextDiT feed-forward) describe failures that return HTTP 200 on
most steps and 500 on exactly the steps that matter, so "the server is up"
proves nothing about System 1.

## Why these exist

Stock InternVLA-N1 discretizes System 1's trajectory into VLN-CE actions
(`STOP`/`FORWARD`/`TURN_LEFT`/`TURN_RIGHT`) and returns only the first action. It
also does not surface the System-2 pixel goal. Two patch sets fix that:

* **PATCH 1–4 (pixel goal).** Carry S2's pixel goal through to the HTTP response.
* **PATCH 5 (continuous trajectory).** Carry S1's **continuous body-frame
  trajectory** through to the HTTP response, so a trajectory follower can fly the
  curve directly instead of the coarse discrete action — the NavDP way.
* **PATCH 6 (look-down).** Say when a look-down has been requested. The action
  index cannot carry it: the agent overwrites the look-down action with `-1`,
  and `-1` is also what an empty System-1 list reports, so on the wire the two
  are indistinguishable. A client that wants to *perform* the look-down — the
  model expects the next frame to be a lower view and computes its pixel goal
  in that frame — has to be told which one it is. One field, `look_down`, added
  to the response dict.

## The continuous trajectory is already there

It is not an addition to the model, only an exposure of what it already computes:

* `model/utils/vln_utils.py::traj_to_actions(dp_actions, use_discrate_action=False)`
  returns the integrated mean path `[T+1, 2]` in **metres, body FLU** (x forward,
  y left) — the same curve it otherwise discretizes.
* `model/utils/vln_utils.py::S1Output` already has a `trajectory` field.

So the model edit is one line filling that field, and the agent edit carries it
to the wire. `core/planning/vlas/internvla_n1` parses it and prefers it over the
discrete action automatically.

## The model edits (not vendored as whole files)

### 1. Carry System 1's continuous trajectory to the wire

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

### What the response looks like after the patches 1-4

`POST /agent/internvla_n1/step` returns, per step:

```json
{"action": [{"action": [1], "ideal_flag": true,
             "pixel_goal": [y, x],
             "trajectory": [[0.0, 0.0], [0.24, 0.03], [0.48, 0.07], "..."],
             "look_down": false,
             "s1_ms": 43.4, "s2_ms": 2641.0}],
 "pixel_goal": [y, x], "pixel_goal_step": 12}
```

Note `look_down` stays **inside `action[0]`**: `server.py` pops and hoists
`pixel_goal` / `pixel_goal_step` to the top level and nothing else, so a
re-implementation that puts it at the top level will be read by this repo's
client (which checks both) but will not match what the agent actually emits.

`trajectory` is `null` on a pure-S2 discrete step (a turn or look-down, where the
model genuinely emits no curve); the client falls back to the discrete action for
those and uses the curve everywhere else.



### 2. Keep System 1 out of the 4-bit quantiser

`code/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`, the
`BitsAndBytesConfig` used by the NF4 load (`INTERNVLA_N1_4BIT=1`, which an 8 GB
card needs — the checkpoint is 16.58 GB at bf16):

```python
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=[
        "traj_dit", "action_encoder", "action_decoder", "cond_projector",
        "rgb_model", "memory_encoder", "rgb_resampler", "latent_queries",
        "lm_head",
    ]),
```

System 1 is 91.4 M parameters — 0.18 GB at bf16, nothing beside System 2's
16.58 — so quantising it saves nothing. It is also **wrong**: `MemoryEncoder`
is an `nn.TransformerEncoder`, and `nn.MultiheadAttention` reads
`out_proj.weight` as a raw tensor inside `F.multi_head_attention_forward`
instead of calling the module. bitsandbytes' `Linear4bit` stores that weight
packed as uint8, so the read reaches `linear(attn_output, out_proj_weight, ...)`
and dies with

```
RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Byte
```

**as an HTTP 500 on every System-1 step**, while the pure System-2 discrete
steps keep answering 200. Symptom seen from the client: the policy never
produces a trajectory and the drone turns on the spot for ever, with a server
that looks perfectly healthy.

### 3. Build the NextDiT feed-forward the size the checkpoint actually is

`code/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`,
`build_traj_dit`:

```python
dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=LatentEmbSize,
                                              ffn_dim_multiplier=2 / 3))
```

Left at the config default (`None`) the installed diffusers builds the
feed-forward at 1536 where the published checkpoint carries 1024, and the load
dies with `size mismatch for weight: ... [1024, 384] vs [1536, 384]`. Under a
**4-bit** load the shape check is bypassed by the quantiser's own parameter
path, so the same defect shows up not as an error but as a System 1 running on
mis-loaded weights — which is why it only appeared once edit 2 above put System
1 back in bf16.

### 4. Let the encoder package import without its optional submodules

`code/internnav/model/encoder/__init__.py` eagerly imports `image_clip_encoder`,
which imports `internnav.model.basemodel.LongCLIP.model` — a git submodule that
is empty in a shallow checkout. InternVLA-N1 reaches this package only for
`depth_anything`, so wrap the optional siblings:

```python
try:
    from .image_clip_encoder import ImageEncoder
except Exception:  # noqa: BLE001
    ImageEncoder = None
```

(same for `instruction_longCLIP_encoder`, `instruction_roberta_encoder` and
`vision_language_encoder`). Without it `/agent/init` fails with
`ModuleNotFoundError: No module named 'internnav.model.basemodel.LongCLIP.model'`
— nothing to do with LongCLIP being needed, only with it being imported.

## The one weight that is not in the checkpoint

`build_depthanythingv2` loads
`checkpoints/depth_anything_v2_metric_hypersim_vits.pth` (~99 MB) to construct
System 1's DINOv2 backbone before `from_pretrained` overwrites it with the
InternVLA-N1 weights. It is **not** part of the InternVLA-N1 checkpoint and its
absence fails `/agent/init` with a bare `FileNotFoundError`. Fetch it from
<https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small>
and put it (or a symlink) at `<internnav>/code/checkpoints/`.

## Starting the server

```bash
cd <internnav>/code            # cwd matters: start_server.py does sys.path.append('.')
INTERNVLA_N1_4BIT=1 CUDA_VISIBLE_DEVICES=0 \
  python scripts/eval/start_server.py --host 127.0.0.1 --port 8087
```

**Do not pass `--config`.** `start_server.py` overwrites `--port` with
`eval_cfg.agent.server_port`, and the shipped
`scripts/eval/configs/h1_internvla_n1_async_cfg.py` says **8023** while this
stack expects 8087. That config also carries a 640x480 / fx 585 / hfov 79
camera, which is not the SJTU drone's; the client sends the real intrinsics with
`/agent/init`, so the config file is not needed at all.

The model is loaded lazily on the first `/agent/init`, not at startup, so
`GET /openapi.json` answers within seconds while the first init takes ~20 s.
A client with a 30 s timeout that treats a timeout as success (this one does)
will mark itself initialised against a server that never finished — pre-warm
with one `/agent/init` before launching the flight nodes.
