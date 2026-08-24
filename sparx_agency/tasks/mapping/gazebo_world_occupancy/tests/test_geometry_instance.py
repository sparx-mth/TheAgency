"""Tests for the placed-shape vocabulary the parser produces."""
from __future__ import annotations

import numpy as np

from sparx_agency.tasks.mapping.gazebo_world_occupancy.geometry_instance import (
    BOX_KIND,
    COLLISION_SOURCE,
    GeometryInstance,
)


def _instance():
    """One box instance, with the ndarray fields a real parse produces."""
    return GeometryInstance(
        model_name="Ward",
        link_name="link",
        source=COLLISION_SOURCE,
        kind=BOX_KIND,
        transform=np.eye(4),
        scale=np.ones(3),
        size=np.ones(3),
    )


def test_two_instances_are_distinct_without_comparing_their_arrays():
    """A frozen dataclass of ndarrays gets an __eq__ that cannot return a bool."""
    first, second = _instance(), _instance()
    assert first == first
    assert first != second


def test_an_instance_can_go_in_a_set():
    """The synthesised __hash__ hashes an ndarray, which is unhashable."""
    first, second = _instance(), _instance()
    assert len({first, second, first}) == 2
