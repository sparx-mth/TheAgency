import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_spin_scans import TURN_DEG, spin_methods, spin_offsets


def test_a_spin_of_k_turns_has_k_plus_one_views_each_one_turn_further_round():
    assert spin_offsets(3) == [0, 30, 60, 90]
    assert spin_offsets(10)[-1] == 300 and len(spin_offsets(10)) == 11


def test_spin_methods_are_prefixes_of_the_longest_spin_and_cost_their_turns():
    methods, actions = spin_methods((3, 6, 10))
    assert methods["spin3"] == [0, 30, 60, 90] and methods["spin6"] == spin_offsets(6) and methods["spin0"] == [0]
    assert actions == {"spin0": 0, "spin3": 3, "spin6": 6, "spin10": 10}
    assert TURN_DEG == 30 and np.isclose(np.deg2rad(TURN_DEG), np.pi / 6)
