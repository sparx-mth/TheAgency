"""Tests for the planner's memory of obstacles the sensors cannot see."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)
from sparx_agency.core.planning.environment.blockage_memory import (
    BlockageMemory,
    BlockageMemoryParams,
)

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)


def _grid(w=40, h=40, res=0.1):
    """An all-free BEV-convention grid with its origin at the world origin."""
    data = np.zeros((h, w), dtype=np.int16)
    return OccupancyGrid2D(
        data, OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=0.0),
        values=BEV)


def test_a_reported_blockage_becomes_occupied_in_the_map():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.3))
    mem.add(2.0, 2.0, 0.0)
    grid = _grid()
    assert mem.stamp(grid) > 0
    gx, gy = grid.world_to_grid(2.0, 2.0)
    assert grid.is_occupied(gx, gy)


def test_repeat_reports_at_the_same_spot_do_not_stack():
    """Grinding against one wall is one fact, not twenty."""
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.35))
    assert mem.add(1.0, 1.0, 0.0) is True
    for _ in range(20):
        assert mem.add(1.05, 1.05, 1.0) is False
    assert len(mem) == 1


def test_distinct_spots_are_remembered_separately():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.2))
    assert mem.add(1.0, 1.0, 0.0) is True
    assert mem.add(3.0, 3.0, 0.0) is True
    assert len(mem) == 2


def test_memory_is_capped_dropping_the_oldest():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.1, max_entries=3))
    for i in range(6):
        mem.add(float(i), 0.0, 0.0)
    assert len(mem) == 3
    assert mem.points == [(3.0, 0.0), (4.0, 0.0), (5.0, 0.0)]


def test_blockages_survive_a_fresh_map():
    """The BEV is rebuilt from depth every frame; an unseen wall reads free again."""
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.3))
    mem.add(2.0, 2.0, 0.0)
    for _ in range(5):
        fresh = _grid()                    # a brand-new, all-free map
        mem.stamp(fresh)
        gx, gy = fresh.world_to_grid(2.0, 2.0)
        assert fresh.is_occupied(gx, gy)


def test_ttl_expires_a_blockage_but_zero_means_forever():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.2, ttl_s=10.0))
    mem.add(1.0, 1.0, 100.0)
    assert mem.prune(105.0) == 0 and len(mem) == 1
    assert mem.prune(111.0) == 1 and len(mem) == 0

    forever = BlockageMemory(BlockageMemoryParams(radius_m=0.2, ttl_s=0.0))
    forever.add(1.0, 1.0, 0.0)
    assert forever.prune(1e9) == 0 and len(forever) == 1


def test_confidence_is_forced_to_certain():
    """Without this the planner treats the cell as cheap and routes through it."""
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.3))
    mem.add(2.0, 2.0, 0.0)
    grid = _grid()
    conf = np.zeros((grid.height, grid.width), dtype=np.float32)
    assert mem.stamp_confidence(conf, grid) > 0
    gx, gy = grid.world_to_grid(2.0, 2.0)
    assert conf[gy, gx] == pytest.approx(1.0)


def test_stamping_outside_the_map_is_harmless():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.3))
    mem.add(99.0, 99.0, 0.0)
    assert mem.stamp(_grid()) == 0


def test_clear_forgets_everything():
    mem = BlockageMemory(BlockageMemoryParams(radius_m=0.2))
    mem.add(1.0, 1.0, 0.0)
    mem.clear()
    assert len(mem) == 0 and mem.stamp(_grid()) == 0


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        BlockageMemoryParams(radius_m=0.0)
    with pytest.raises(ValueError):
        BlockageMemoryParams(ttl_s=-1.0)
    with pytest.raises(ValueError):
        BlockageMemoryParams(max_entries=0)
