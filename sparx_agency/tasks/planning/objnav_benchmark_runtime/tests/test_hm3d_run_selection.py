"""Episode selection and development flags of the HM3D entrypoint: the same contract Gibson's has."""
from types import SimpleNamespace

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import parser
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    episodes_of_category, explicit_episodes, select_episodes)


def test_shuffle_is_a_seeded_sample_fixed_after_sharding():
    ids = tuple("00877-4ok3usBNeis/%06d" % i for i in range(30))
    first = select_episodes(ids, limit=3, shuffle_seed=17)
    assert first == select_episodes(ids, limit=3, shuffle_seed=17) and set(first) <= set(ids)
    assert first != ids[:3] and select_episodes(ids, limit=3, shuffle_seed=18) != first
    assert sorted(select_episodes(ids, shuffle_seed=17)) == sorted(ids)
    assert set(select_episodes(ids, limit=2, shards=2, shard_index=1, shuffle_seed=17)) <= set(ids[1::2])


def test_without_a_seed_selection_keeps_split_order():
    ids = tuple("S/%06d" % i for i in range(10))
    assert select_episodes(ids, limit=3) == ids[:3]


def test_explicit_ids_run_as_given_and_must_lie_in_the_pool():
    pool = tuple("S/%06d" % i for i in range(20))
    assert explicit_episodes(["S/000007", "S/000002"], pool) == ("S/000007", "S/000002")
    for bad in (["S/000007", "S/000007"], ["T/000001"]):
        with pytest.raises(ValueError):
            explicit_episodes(bad, pool)


def test_category_filter_keeps_goal_in_split_order():
    episodes = {"S/%06d" % i: SimpleNamespace(category="toilet" if i % 3 == 0 else "chair") for i in range(9)}
    ids = tuple(episodes)
    assert episodes_of_category(ids, episodes, None, CATEGORIES) == ids
    assert episodes_of_category(ids, episodes, "toilet", CATEGORIES) == ("S/000000", "S/000003", "S/000006")
    with pytest.raises(ValueError, match="No published bed episode"):
        episodes_of_category(ids, episodes, "bed", CATEGORIES)
    with pytest.raises(ValueError, match="Unknown category"):
        episodes_of_category(ids, episodes, "sink", CATEGORIES)


def test_parser_exposes_the_development_flags():
    args = parser().parse_args(["--shuffle", "--limit", "3", "--category", "toilet", "--lean",
                                "--episode-ids", "S/000001", "--progress-file", "p.json"])
    assert (args.shuffle, args.category, args.lean, args.episode_ids) == (True, "toilet", True, ["S/000001"])
