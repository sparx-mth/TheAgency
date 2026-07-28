"""The torch half: the cache must be exact, and the objective must run.

Needs the ``navdp`` conda env, the external NavDP repo (``NAVDP_REPO``) and the
pretrained checkpoint; skipped cleanly anywhere else, so this file is safe in a
plain ``.venv`` collection.

The load-bearing test here is :func:`test_cached_tokens_match_the_live_encoder`.
The whole training speed-up rests on the claim that resuming NavDP's encoder from
cached patch tokens gives the same scene embedding as running it from pixels. If
that drifts, every cached run trains against a slightly wrong representation and
nothing else in the pipeline would notice.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CKPT = Path(os.environ.get("NAVDP_CKPT",
                           Path.home() / "Downloads" / "navdp-cross-modal.ckpt"))
REPO = os.environ.get("NAVDP_REPO",
                      str(Path.home() / "PycharmProjects" / "NavDP" / "baselines" / "navdp"))

pytestmark = pytest.mark.skipif(
    not CKPT.exists() or not Path(REPO, "policy_network.py").exists(),
    reason="needs the NavDP checkpoint and repo (NAVDP_CKPT / NAVDP_REPO)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2
HORIZON = 24


@pytest.fixture(scope="module")
def model():
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.model import (
        WorldGoalModelConfig, WorldGoalNavDP,
    )
    built = WorldGoalNavDP(str(CKPT), REPO, device=DEVICE,
                           config=WorldGoalModelConfig()).to(DEVICE).eval()
    return built


@pytest.fixture(scope="module")
def fields():
    """A 20 m x 20 m corridor ESDF on the device, at the training resolution."""
    from sparx_agency.core.mapping.costmap.sdf import compute_sdf
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
        SceneField, SceneFields,
    )
    occupancy = np.zeros((200, 200), dtype=np.uint8)
    occupancy[:60, :] = 1
    occupancy[140:, :] = 1
    sdf = compute_sdf(occupancy, 0.1)
    return SceneFields([SceneField(sdf, 0.1, 0.0, 0.0, DEVICE)])


def synthetic_batch():
    generator = torch.Generator().manual_seed(7)
    return {
        "images": torch.rand((BATCH, 8, 3, 224, 224), generator=generator),
        "depth": torch.rand((BATCH, 1, 224, 224), generator=generator) * 4.0 + 0.5,
        "goal": torch.tensor([[4.0, 0.5, 0.0], [2.0, -1.0, 0.0]]),
        "action": torch.rand((BATCH, HORIZON, 3), generator=generator) * 0.6,
        "pose": torch.tensor([[5.0, 10.0, 0.0], [8.0, 10.0, 0.3]]),
        "goal_world": torch.tensor([[12.0, 10.0], [14.0, 11.0]]),
        "scene": torch.zeros(BATCH, dtype=torch.long),
    }


# ------------------------------------------------------------------- the cache
@torch.no_grad()
def test_cached_tokens_match_the_live_encoder(model):
    """Resuming from cached patch tokens must equal encoding from pixels.

    The entire ~30x training speed-up rests on this identity. Tolerance is loose
    enough for float32 non-determinism in attention and tight enough that a real
    divergence (a missing positional embedding, a wrong reshape) fails.
    """
    batch = synthetic_batch()
    images = batch["images"].to(DEVICE)
    depth = batch["depth"].to(DEVICE)

    live = model.encode(images, depth)
    rgb_tokens = model.tokenize_rgb(images.reshape(-1, 3, 224, 224)).reshape(
        BATCH, 8, 256, 384)
    depth_tokens = model.tokenize_depth(depth)
    cached = model.encode_tokens(rgb_tokens, depth_tokens)

    assert cached.shape == (BATCH, 128, 384)
    assert torch.allclose(live, cached, atol=2e-4, rtol=2e-3), (
        f"max abs difference {float((live - cached).abs().max()):.2e}")


@torch.no_grad()
def test_float16_cache_round_trip_stays_close(model):
    """The cache is stored as float16; that must not move the embedding much."""
    batch = synthetic_batch()
    images, depth = batch["images"].to(DEVICE), batch["depth"].to(DEVICE)
    rgb = model.tokenize_rgb(images.reshape(-1, 3, 224, 224)).reshape(BATCH, 8, 256, 384)
    depth_tokens = model.tokenize_depth(depth)
    exact = model.encode_tokens(rgb, depth_tokens)
    halved = model.encode_tokens(rgb.half().float(), depth_tokens.half().float())
    assert float((exact - halved).abs().max()) < 5e-2


# -------------------------------------------------------------- freeze policy
def test_freeze_policy_matches_what_is_documented(model):
    counts = model.param_counts()
    assert counts["total"] == pytest.approx(135_730_000, rel=0.01)
    assert counts["trainable"] == pytest.approx(44_500_000, rel=0.02), (
        "expected the Q-Former + the 16 decoder layers + the heads; a larger "
        "number usually means a prefix picked up NavDP's unused prototype "
        "decoder layer, which never receives a gradient")
    frozen_rgb = [name for name, p in model.policy.named_parameters()
                  if name.startswith("rgbd_encoder.rgb_model") and p.requires_grad]
    assert not frozen_rgb, "the RGB trunk must never be trainable"
    assert not model.depth_encoder_trainable


def test_trainable_checkpoint_round_trips(model):
    state = model.trainable_state_dict()
    assert 0 < len(state) < len(model.policy.state_dict())
    with torch.no_grad():
        for value in state.values():
            value.add_(0.01)
    assert model.load_trainable(state, strict=True) > 0


def test_loading_an_unrelated_checkpoint_raises(model):
    """A silent no-op load would report the baseline as the fine-tuned result."""
    with pytest.raises(RuntimeError, match="changed nothing"):
        model.load_trainable({"not.a.real.parameter": torch.zeros(1)}, strict=True)


# --------------------------------------------------------------------- geometry
def test_scene_field_matches_the_numpy_sampler(fields):
    from sparx_agency.core.mapping.costmap.sdf import compute_sdf
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
        bilinear_sample,
    )
    occupancy = np.zeros((200, 200), dtype=np.uint8)
    occupancy[:60, :] = 1
    occupancy[140:, :] = 1
    sdf = compute_sdf(occupancy, 0.1)

    points = np.array([[[3.0, 8.0], [5.0, 10.0], [7.0, 12.5]]])
    expected = bilinear_sample(sdf, 0.1, 0.0, 0.0, points[0, :, 0], points[0, :, 1])
    got = fields.sample(torch.tensor(points, dtype=torch.float32, device=DEVICE),
                        torch.zeros(1, dtype=torch.long, device=DEVICE))
    assert np.allclose(got.cpu().numpy()[0], expected, atol=1e-4)


def test_action_waypoint_round_trip():
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
        body_to_world, decode_action, encode_waypoints,
    )
    action = torch.rand((3, HORIZON, 3)) * 0.5
    waypoints = decode_action(action)
    assert torch.allclose(decode_action(encode_waypoints(waypoints)), waypoints,
                          atol=1e-5)
    pose = torch.tensor([[1.0, 2.0, 0.4], [0.0, 0.0, 0.0], [-3.0, 1.0, -1.2]])
    world = body_to_world(waypoints, pose)
    assert world.shape == waypoints.shape
    # Waypoint 0 of a zero-yaw pose is just the offset from the pose.
    assert torch.allclose(world[1, 0], waypoints[1, 0], atol=1e-6)


# ------------------------------------------------------------------- objective
def test_objective_runs_and_every_term_is_finite(model, fields):
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
        WorldGoalLoss, WorldGoalLossConfig,
    )
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.train import (
        forward_step,
    )
    loss_fn = WorldGoalLoss(WorldGoalLossConfig()).to(DEVICE)
    total, parts = forward_step(model, loss_fn, synthetic_batch(), fields, DEVICE)
    assert torch.isfinite(total)
    for name in ("raw/act", "raw/waypoint", "raw/clearance", "raw/goal", "raw/critic"):
        assert np.isfinite(parts[name]), name
    for name in ("metric/min_clear_m", "metric/collide_frac", "metric/goal_gap_m"):
        assert name in parts


def test_gradients_reach_the_head_and_not_the_frozen_trunk(model, fields):
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
        WorldGoalLoss, WorldGoalLossConfig,
    )
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.train import (
        forward_step,
    )
    model.zero_grad(set_to_none=True)
    loss_fn = WorldGoalLoss(WorldGoalLossConfig()).to(DEVICE)
    total, _ = forward_step(model, loss_fn, synthetic_batch(), fields, DEVICE)
    total.backward()

    named = dict(model.policy.named_parameters())
    assert named["action_head.weight"].grad is not None
    assert float(named["action_head.weight"].grad.abs().sum()) > 0
    assert named["rgbd_encoder.former_query.position_embedding.weight"].grad is not None
    assert named["rgbd_encoder.rgb_model.pos_embed"].grad is None
    model.zero_grad(set_to_none=True)


def test_critic_negatives_are_actually_different(model, fields):
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
        WorldGoalLoss, WorldGoalLossConfig, decode_action,
    )
    loss_fn = WorldGoalLoss(WorldGoalLossConfig(critic_negatives=3)).to(DEVICE)
    expert = decode_action(synthetic_batch()["action"].to(DEVICE))
    wrong = loss_fn.negatives(expert)
    assert wrong.shape == (BATCH, 3, HORIZON, 2)
    for index in range(3):
        assert float((wrong[:, index] - expert).abs().mean()) > 0.05


def test_inference_is_reproducible_under_a_seed(model):
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.infer import (
        NavDPRunner,
    )
    batch = synthetic_batch()
    with torch.no_grad():
        rgbd = model.encode(batch["images"].to(DEVICE), batch["depth"].to(DEVICE))
    runner = NavDPRunner(model, sample_num=4)
    first = runner.run(rgbd, batch["goal"].to(DEVICE), seed=11)
    second = runner.run(rgbd, batch["goal"].to(DEVICE), seed=11)
    other = runner.run(rgbd, batch["goal"].to(DEVICE), seed=12)
    assert first.trajectory.shape == (BATCH, HORIZON, 2)
    assert torch.allclose(first.trajectory, second.trajectory, atol=1e-5)
    assert not torch.allclose(first.trajectory, other.trajectory, atol=1e-3)
