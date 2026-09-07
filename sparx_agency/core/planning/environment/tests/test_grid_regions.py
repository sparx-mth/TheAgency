"""Connected regions of a boolean grid: correctness against a brute-force BFS."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from sparx_agency.core.planning.environment.grid_regions import (
    connected_regions,
    flood_region,
    largest_enclosed_region,
)


def _reference_flood(mask, gy, gx, connectivity):
    """A plain BFS, written for obviousness rather than speed."""
    out = np.zeros(mask.shape, dtype=bool)
    if not mask[gy, gx]:
        return out
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = mask.shape
    queue = deque([(gy, gx)])
    out[gy, gx] = True
    while queue:
        y, x = queue.popleft()
        for dy, dx in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((ny, nx))
    return out


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(12))
def test_flood_matches_a_brute_force_bfs(connectivity, seed):
    rng = np.random.RandomState(seed)
    mask = rng.rand(24, 31) > 0.42
    ys, xs = np.nonzero(mask)
    if not ys.size:
        pytest.skip("empty draw")
    gy, gx = int(ys[0]), int(xs[0])
    got = flood_region(mask, gy, gx, connectivity)
    want = _reference_flood(mask, gy, gx, connectivity)
    assert np.array_equal(got, want)


def test_four_and_eight_connectivity_differ_at_a_diagonal_pinch():
    #  X .        the two set cells touch only at a corner
    #  . X
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = mask[1, 1] = True
    assert flood_region(mask, 0, 0, 4).sum() == 1
    assert flood_region(mask, 0, 0, 8).sum() == 2


def test_flood_of_an_unset_or_out_of_bounds_seed_is_empty():
    mask = np.ones((4, 4), dtype=bool)
    mask[2, 2] = False
    assert not flood_region(mask, 2, 2, 8).any()
    assert not flood_region(mask, -1, 0, 8).any()
    assert not flood_region(mask, 0, 99, 8).any()


def test_connected_regions_partitions_the_mask_largest_first():
    mask = np.zeros((9, 9), dtype=bool)
    mask[0:2, 0:2] = True          # 4 cells
    mask[4:8, 4:8] = True          # 16 cells
    mask[8, 0] = True              # 1 cell
    regions = connected_regions(mask, connectivity=4)
    assert [int(r.sum()) for r in regions] == [16, 4, 1]
    union = np.zeros_like(mask)
    for region in regions:
        assert not (union & region).any(), "components must be disjoint"
        union |= region
    assert np.array_equal(union, mask)


def test_connected_regions_of_nothing_is_empty():
    assert connected_regions(np.zeros((5, 5), dtype=bool)) == []


def test_largest_enclosed_region_is_the_room_not_the_field_around_it():
    # A walled room in the middle of open ground. The open ground runs off the
    # edge of the grid; the room does not.
    free = np.ones((20, 20), dtype=bool)
    free[5:15, 5:15] = False          # the wall block...
    free[6:14, 6:14] = True           # ...hollowed out into a room
    room = largest_enclosed_region(free, connectivity=4)
    assert room is not None
    assert int(room.sum()) == 8 * 8
    assert room[6, 6] and not room[0, 0]


def test_largest_enclosed_region_ignores_a_smaller_sealed_void():
    free = np.ones((30, 30), dtype=bool)
    free[4:26, 4:26] = False
    free[5:25, 5:25] = True           # the big room, 20x20
    free[10:14, 10:14] = False        # a solid block inside it...
    free[11:13, 11:13] = True         # ...with a sealed 2x2 cavity
    room = largest_enclosed_region(free, connectivity=4)
    assert room is not None
    assert int(room.sum()) == 20 * 20 - 4 * 4
    assert not room[11, 11], "the sealed cavity is not part of the floor"


def test_largest_enclosed_region_is_none_when_everything_reaches_the_edge():
    assert largest_enclosed_region(np.ones((6, 6), dtype=bool)) is None


def test_connectivity_must_be_four_or_eight():
    with pytest.raises(ValueError):
        flood_region(np.ones((3, 3), dtype=bool), 0, 0, connectivity=6)


def test_mask_must_be_two_dimensional():
    with pytest.raises(ValueError):
        flood_region(np.ones(5, dtype=bool), 0, 0)
